from __future__ import annotations

"""Phase-aware adversarial training utilities (Phase-AT).

We generate adversarial examples by taking update steps
in the frequency domain (phase/amplitude/complex), optionally masked by a spectral
saliency map, and provide a simple per-epoch curriculum scheduler.
"""

from typing import Optional, Tuple, Callable

import torch
from torch import nn

DEBUG_PREFIX = "[PHASE-AT DEBUG]"


# -------------------------------------------------------------------------
# Color-space helpers (RGB <-> YCbCr)
# -------------------------------------------------------------------------

def _mean_std_tensors(
    data_mean: Tuple[float, float, float],
    data_std: Tuple[float, float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create broadcastable mean/std tensors for image normalization.

    Args:
        data_mean: Per-channel mean (R, G, B).
        data_std: Per-channel std (R, G, B).
        device: Target device.
        dtype: Target dtype.

    Returns:
        (mean, std) shaped as (1, 3, 1, 1) each, suitable for broadcasting.
    """
    mean = torch.tensor(data_mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std = torch.tensor(data_std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean, std


def _rgb_to_ycbcr_01(x_rgb: torch.Tensor, offset: float = 0.5) -> torch.Tensor:
    """Convert RGB->YCbCr for inputs in [0, 1] (full-range style).

    Args:
        x_rgb: (B, 3, H, W) float tensor (typically in [0,1]).
        offset: Chroma offset (0.5 for [0,1] range).

    Returns:
        ycbcr: (B, 3, H, W) tensor.
    """
    if x_rgb.dim() != 4 or x_rgb.size(1) != 3:
        raise ValueError(f"_rgb_to_ycbcr_01 expects (B,3,H,W), got {tuple(x_rgb.shape)}")

    r = x_rgb[:, 0:1]
    g = x_rgb[:, 1:2]
    b = x_rgb[:, 2:3]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + offset
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + offset

    return torch.cat([y, cb, cr], dim=1)


def _ycbcr_to_rgb_01(x_ycbcr: torch.Tensor, offset: float = 0.5) -> torch.Tensor:
    """Convert YCbCr->RGB for tensors produced by :func:`_rgb_to_ycbcr_01`.

    Args:
        x_ycbcr: (B, 3, H, W) tensor.
        offset: Chroma offset used in the forward transform.

    Returns:
        x_rgb: (B, 3, H, W) tensor.
    """
    if x_ycbcr.dim() != 4 or x_ycbcr.size(1) != 3:
        raise ValueError(f"_ycbcr_to_rgb_01 expects (B,3,H,W), got {tuple(x_ycbcr.shape)}")

    y = x_ycbcr[:, 0:1]
    cb = x_ycbcr[:, 1:2] - offset
    cr = x_ycbcr[:, 2:3] - offset

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    return torch.cat([r, g, b], dim=1)


# -------------------------------------------------------------------------
# Basic FFT helpers
# -------------------------------------------------------------------------

def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Compute a 2D Fourier transform over the last two dimensions.

    Args:
        x: Real-valued tensor shaped (B, C, H, W).

    Returns:
        Complex-valued FFT of x with shape (B, C, H, W).
    """
    return torch.fft.fft2(x)


def ifft2c(X: torch.Tensor) -> torch.Tensor:
    """Compute the inverse 2D Fourier transform and return the real component.

    Args:
        X: Complex-valued tensor shaped (B, C, H, W).

    Returns:
        Real-valued inverse FFT with shape (B, C, H, W).
    """
    return torch.fft.ifft2(X).real


# -------------------------------------------------------------------------
# Spectral saliency and Masks
# -------------------------------------------------------------------------

def spectral_saliency(G_freq: torch.Tensor) -> torch.Tensor:
    """Compute a simple spectral saliency score from a frequency-domain gradient.

    This implementation uses the gradient magnitude as saliency.
    """
    return torch.abs(G_freq)


def spectral_mask(
    S: torch.Tensor, mask_type: str = "soft", topk_frac: float = 0.05
) -> torch.Tensor:
    """Build a spectral mask from a saliency tensor.

    Args:
        S: Spectral saliency, shape (B, C, H, W).
        mask_type: 'hard' -> binary top-k, 'soft' -> normalized saliency in [0,1],
            'none' -> all-ones mask.
        topk_frac: Fraction of frequencies to keep in 'hard' mode.

    Returns:
        Mask tensor shaped like S.
    """
    B, C, H, W = S.shape
    N = H * W

    if mask_type == "none":
        # No masking: all frequencies are equally weighted.
        return torch.ones_like(S)

    if mask_type == "hard":
        # Binary mask: keep only top-k frequencies.
        k = max(1, int(topk_frac * N))
        S_flat = S.view(B, C, -1)
        _, topk_idx = torch.topk(S_flat, k, dim=-1)
        M_flat = torch.zeros_like(S_flat)
        M_flat.scatter_(dim=-1, index=topk_idx, value=1.0)
        return M_flat.view(B, C, H, W)

    if mask_type == "soft":
        # Pure normalized saliency: strictly [0,1], no floor.
        S_flat = S.view(B, C, -1)
        min_v = S_flat.min(dim=-1, keepdim=True)[0]
        max_v = S_flat.max(dim=-1, keepdim=True)[0]
        eps = 1e-8
        S_norm = (S_flat - min_v) / (max_v - min_v + eps)
        return S_norm.view(B, C, H, W)

    raise ValueError(f"Unknown mask_type: {mask_type}")


# -------------------------------------------------------------------------
# Implicit phase gradient
# -------------------------------------------------------------------------

def implicit_phase_grad(G_freq: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Compute an implicit gradient w.r.t. phase.

    The phase gradient is derived from the interaction between the frequency-domain
    gradient and the conjugate of the current frequency representation.

    Args:
        G_freq: Gradient w.r.t. X, complex tensor.
        X: Current frequency representation, complex tensor.

    Returns:
        Real-valued tensor representing an implicit phase gradient.
    """
    interaction = G_freq * torch.conj(X)
    grad_phi = interaction.imag
    return grad_phi


# -------------------------------------------------------------------------
# One Phase-AT step (frequency domain)
# -------------------------------------------------------------------------

def phase_at_step(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    step_size_phi: float,
    mask_type: str,
    topk_frac: float,
    update_mode: str = "phase",
    direction_mode: str = "adversarial",
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    reconstruct_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    debug: bool = False,
    debug_step_idx: Optional[int] = None,
    debug_tag: str = "",
) -> torch.Tensor:
    """Perform a single frequency-domain update step for Phase-AT.

    This function takes the current frequency representation X, computes gradients
    of the task loss with respect to X, optionally applies a spectral mask, and
    then updates phase/amplitude/complex components depending on `update_mode`.

    Args:
        model: The classifier.
        X: Current frequency representation (complex tensor).
        y: Target labels.
        criterion: Loss function.
        step_size_phi: Step size in the chosen update space.
        mask_type: 'soft' | 'hard' | 'none'.
        topk_frac: Top-k fraction used for 'hard' masking.
        update_mode: 'phase' | 'amplitude' | 'complex'.
        direction_mode: 'adversarial' (use gradient sign) | 'random'.
        clamp_min/clamp_max: Optional clamping bounds for spatial-domain inputs.
        reconstruct_fn: Optional function mapping X -> spatial-domain adversarial input.
        debug: If True, prints step statistics.
        debug_step_idx: Step index used for debug print formatting.
        debug_tag: Extra tag string for debug prints.

    Returns:
        Updated frequency representation X (detached).
    """
    if update_mode not in ("phase", "amplitude", "complex"):
        raise ValueError(f"Unknown update_mode: {update_mode}")
    if direction_mode not in ("adversarial", "random"):
        raise ValueError(f"Unknown direction_mode: {direction_mode}")

    X = X.detach().clone().requires_grad_(True)

    # 1) Convert current spectrum to spatial domain (or custom reconstruction).
    x_adv = ifft2c(X) if reconstruct_fn is None else reconstruct_fn(X)

    # 2) Optionally clamp the adversarial input.
    if clamp_min is not None and clamp_max is not None:
        x_adv = x_adv.clamp(clamp_min, clamp_max)

    # 3) Forward + loss.
    logits = model(x_adv)
    loss = criterion(logits, y)

    # 4) Gradient w.r.t. the frequency representation.
    (G_freq,) = torch.autograd.grad(loss, X, create_graph=False)

    # 5) Build an update (phase/amplitude/complex) and take one step.
    step_tensor: torch.Tensor

    if update_mode == "phase":
        # Phase-direction gradient (used for both sign direction and phase-saliency mask).
        grad_phi = implicit_phase_grad(G_freq, X)

        # Mask from |grad_phi| (phase-saliency).
        S = torch.abs(grad_phi)
        M = spectral_mask(S, mask_type=mask_type, topk_frac=topk_frac)

        if direction_mode == "random":
            rnd = torch.empty_like(grad_phi).uniform_(-1.0, 1.0)
            phase_dir = rnd.sign()
        else:
            phase_dir = grad_phi.sign()

        phase_step = step_size_phi * M * phase_dir
        step_tensor = phase_step
        X_next = X * torch.exp(1j * phase_step)

    elif update_mode == "amplitude":
        amp = torch.abs(X)
        phase = torch.angle(X)

        # Compute amplitude-gradient for saliency; direction may still be random.
        grad_amp = (G_freq * torch.exp(-1j * phase)).real
        S = torch.abs(grad_amp)
        M = spectral_mask(S, mask_type=mask_type, topk_frac=topk_frac)

        if direction_mode == "random":
            rnd = torch.empty_like(amp).uniform_(-1.0, 1.0)
            amp_dir = rnd.sign()
        else:
            amp_dir = grad_amp.sign()

        amp_step = step_size_phi * M * amp_dir
        step_tensor = amp_step
        amp_next = (amp + amp_step).clamp(min=0.0)
        X_next = amp_next * torch.exp(1j * phase)

    else:  # update_mode == "complex"
        # Saliency from |G_freq|.
        S = spectral_saliency(G_freq)
        M = spectral_mask(S, mask_type=mask_type, topk_frac=topk_frac)

        if direction_mode == "random":
            rnd_r = torch.empty_like(X.real).uniform_(-1.0, 1.0).sign()
            rnd_i = torch.empty_like(X.imag).uniform_(-1.0, 1.0).sign()
            direction = rnd_r.to(dtype=X.dtype) + 1j * rnd_i.to(dtype=X.dtype)
        else:
            direction = torch.sign(G_freq.real).to(dtype=X.dtype) + 1j * torch.sign(G_freq.imag).to(dtype=X.dtype)

        complex_step = (step_size_phi * M).to(dtype=X.dtype) * direction
        step_tensor = complex_step
        X_next = X + complex_step

    if debug:
        step_str = f"step={debug_step_idx}" if debug_step_idx is not None else "step=?"
        prefix = f"{DEBUG_PREFIX} {debug_tag} {step_str}"
        with torch.no_grad():
            mean_step = step_tensor.abs().mean().item()
            max_step = step_tensor.abs().max().item()
            print(f"{prefix} Step Magnitude: Mean={mean_step:.4e}, Max={max_step:.4e}")

    return X_next.detach()


# -------------------------------------------------------------------------
# Attack loop
# -------------------------------------------------------------------------

def phase_at_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    num_steps: int = 3,
    step_size_phi: float = 0.1,
    mask_type: str = "soft",
    topk_frac: float = 0.05,
    update_mode: str = "phase",
    direction_mode: str = "adversarial",
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    random_init_phase: bool = False,
    debug: bool = False,
    debug_steps: int = 1,
    debug_tag: str = "",
    color_mode: str = "rgb",
    data_mean: Optional[Tuple[float, float, float]] = None,
    data_std: Optional[Tuple[float, float, float]] = None,
) -> torch.Tensor:
    """Generate Phase-AT adversarial examples for a batch.

    The perturbation is applied in the frequency domain, optionally operating in
    RGB or YCbCr space.

    Args:
        model: The classifier.
        x: Normalized input images, shape (B, 3, H, W).
        y: Target labels.
        criterion: Loss function.
        num_steps: Number of frequency-domain steps.
        step_size_phi: Step size used by :func:`phase_at_step`.
        mask_type/topk_frac: Spectral masking configuration.
        update_mode: 'phase' | 'amplitude' | 'complex'.
        direction_mode: 'adversarial' | 'random'.
        clamp_min/clamp_max: Optional clamping bounds in normalized space.
        random_init_phase: If True, starts from a random phase initialization.
        debug/debug_steps/debug_tag: Debug printing controls.
        color_mode: 'rgb' | 'ycbcr' | 'ycbcr_all'.
        data_mean/data_std: Required when using YCbCr modes to move between
            normalized space and [0,1] RGB space.

    Returns:
        Adversarial images in the same normalized space as x.
    """
    if color_mode not in ("rgb", "ycbcr", "ycbcr_all"):
        raise ValueError(f"Unknown color_mode for Phase-AT: {color_mode}")

    reconstruct_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    if color_mode == "rgb":
        X = fft2c(x)
        finalize = lambda X_in: ifft2c(X_in)
    else:
        # YCbCr-space perturbations. We need mean/std to move between normalized
        # model space and [0,1] RGB space where the color conversion is meaningful.
        if data_mean is None or data_std is None:
            raise ValueError(
                "phase_at_attack(color_mode in {'ycbcr','ycbcr_all'}) requires data_mean and data_std"
            )

        mean_t, std_t = _mean_std_tensors(data_mean, data_std, device=x.device, dtype=x.dtype)

        # Convert normalized -> RGB([0,1]) then to YCbCr.
        x_rgb = (x * std_t + mean_t).clamp(0.0, 1.0)
        ycbcr = _rgb_to_ycbcr_01(x_rgb)

        if color_mode == "ycbcr":
            y_chan = ycbcr[:, 0:1]
            cbcr = ycbcr[:, 1:3].detach()  # keep chroma fixed

            X = fft2c(y_chan)

            def reconstruct_fn(X_in: torch.Tensor) -> torch.Tensor:
                """Reconstruct normalized RGB inputs from an adversarial Y-channel spectrum."""
                y_adv = ifft2c(X_in)
                ycbcr_adv = torch.cat([y_adv, cbcr], dim=1)
                x_rgb_adv = _ycbcr_to_rgb_01(ycbcr_adv).clamp(0.0, 1.0)
                x_norm_adv = (x_rgb_adv - mean_t) / std_t
                return x_norm_adv

        else:  # color_mode == "ycbcr_all"
            X = fft2c(ycbcr)

            def reconstruct_fn(X_in: torch.Tensor) -> torch.Tensor:
                """Reconstruct normalized RGB inputs from an adversarial YCbCr spectrum."""
                ycbcr_adv = ifft2c(X_in)
                x_rgb_adv = _ycbcr_to_rgb_01(ycbcr_adv).clamp(0.0, 1.0)
                x_norm_adv = (x_rgb_adv - mean_t) / std_t
                return x_norm_adv

        finalize = lambda X_in: reconstruct_fn(X_in)

    if random_init_phase:
        # Phase noise has same shape as the real part of the spectrum.
        noise = torch.empty_like(X.real, dtype=torch.float32).uniform_(-step_size_phi, step_size_phi)
        X = X * torch.exp(1j * noise.to(X.device))

    was_training = model.training
    model.eval()

    with torch.enable_grad():
        for step_idx in range(num_steps):
            step_debug = debug and (step_idx < debug_steps)
            X = phase_at_step(
                model=model,
                X=X,
                y=y,
                criterion=criterion,
                step_size_phi=step_size_phi,
                mask_type=mask_type,
                topk_frac=topk_frac,
                update_mode=update_mode,
                direction_mode=direction_mode,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                reconstruct_fn=reconstruct_fn,
                debug=step_debug,
                debug_step_idx=step_idx,
                debug_tag=debug_tag,
            )

    if was_training:
        model.train()

    x_adv = finalize(X)

    if clamp_min is not None and clamp_max is not None:
        x_adv = x_adv.clamp(clamp_min, clamp_max)

    return x_adv


# -------------------------------------------------------------------------
# Curriculum scheduler for Phase-AT (per-epoch)
# -------------------------------------------------------------------------

def phase_scheduler(
    epoch: int,
    warmup_epochs: int,
    rampup_epochs: int,
    step_size_max: float,
    lambda_adv_max: float,
    step_size_init: float = 0.02,
    lambda_adv_init: float = 0.1,
) -> Tuple[float, float]:
    """Compute a per-epoch curriculum schedule for Phase-AT.

    Phases:
        1) epoch < warmup_epochs:
           - step_size = 0, lambda_adv = 0 (clean training)

        2) warmup_epochs <= epoch < warmup_epochs + rampup_epochs:
           - linear ramp from (step_size_init, lambda_adv_init)
             to (step_size_max, lambda_adv_max)

        3) epoch >= warmup_epochs + rampup_epochs:
           - step_size = step_size_max, lambda_adv = lambda_adv_max

    Args:
        epoch: Current epoch index (0-based).
        warmup_epochs: Number of initial epochs without adversarial training.
        rampup_epochs: Number of epochs over which to ramp up.
        step_size_max: Maximum step size.
        lambda_adv_max: Maximum weight on adversarial loss.
        step_size_init: Initial step size at the start of ramp-up.
        lambda_adv_init: Initial adversarial weight at the start of ramp-up.

    Returns:
        (current_step_size, current_lambda_adv)
    """
    if epoch < warmup_epochs:
        return 0.0, 0.0

    if epoch >= warmup_epochs + rampup_epochs:
        return step_size_max, lambda_adv_max

    progress = (epoch - warmup_epochs) / float(rampup_epochs)
    cur_step = step_size_init + progress * (step_size_max - step_size_init)
    cur_lambda = lambda_adv_init + progress * (lambda_adv_max - lambda_adv_init)
    return float(cur_step), float(cur_lambda)
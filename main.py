import os
import ssl
import random
from typing import Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from tqdm import tqdm

from wilds import get_dataset

from args import parse_args
from phase_at import phase_at_attack, phase_scheduler


# =============================================================================
# Repro helpers
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set Python/NumPy/PyTorch random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def seed_worker(worker_id: int) -> None:
    """Seed each dataloader worker deterministically for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# Camelyon17 helpers
# =============================================================================

class TransformedWildsSubset(Dataset):
    """Subset wrapper for a WILDS dataset that applies a torchvision transform.

    WILDS datasets return (x, y, metadata). This wrapper selects a subset of
    indices and applies `transform` to x.
    """

    def __init__(self, dataset, indices, transform=None):
        """Initialize the subset.

        Args:
            dataset: A WILDS dataset instance.
            indices: Iterable of integer indices into the dataset.
            transform: Optional torchvision-style transform applied to x.
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        """Return the idx-th sample from the subset as (x, y, metadata)."""
        real_idx = self.indices[idx]
        x, y, metadata = self.dataset[real_idx]

        if self.transform:
            x = self.transform(x)

        return x, y, metadata

    def __len__(self):
        """Return the subset size."""
        return len(self.indices)


def print_hospital_counts_indices(dataset, indices, title: str) -> None:
    """Print counts per hospital for a list of dataset indices."""
    idx_t = torch.as_tensor(indices, dtype=torch.long)
    hosp_ids = dataset.metadata_array[idx_t, 0]
    unique_h, counts = torch.unique(hosp_ids, return_counts=True)

    pairs = [(int(h.item()), int(c.item())) for h, c in zip(unique_h, counts)]
    pairs.sort(key=lambda x: x[0])

    print(f"\n{title}")
    print("-" * 60)
    print(f"Total samples: {len(indices)}")
    for h, c in pairs:
        print(f"Hospital {h}: {c} samples")
    print("-" * 60)


def subsample_indices_by_hospital(
    dataset,
    indices,
    fraction: float,
    seed: int,
    min_per_hospital: int = 1,
):
    """Subsample indices per hospital to keep a fixed fraction from each domain.

    Args:
        dataset: WILDS Camelyon17 dataset (must have `metadata_array`).
        indices: List of indices to subsample from.
        fraction: Fraction (0,1] to keep per hospital. If >=1.0, returns indices.
        seed: Base RNG seed.
        min_per_hospital: Minimum samples to keep per hospital.

    Returns:
        Sorted list of selected indices.
    """
    if fraction >= 1.0:
        return indices

    idx_t = torch.as_tensor(indices, dtype=torch.long)
    hosp_ids = dataset.metadata_array[idx_t, 0].cpu().numpy()

    selected_indices = []
    unique_hosp = np.unique(hosp_ids)
    unique_hosp.sort()

    for h in unique_hosp:
        local_pos = np.where(hosp_ids == h)[0]
        n = len(local_pos)
        if n == 0:
            continue

        n_keep = int(round(fraction * n))
        n_keep = max(min_per_hospital, n_keep)
        n_keep = min(n_keep, n)

        rng = np.random.default_rng(seed + int(h) * 10007)
        chosen_local = rng.choice(local_pos, size=n_keep, replace=False)
        global_chosen = [indices[i] for i in chosen_local]
        selected_indices.extend(global_chosen)

    if not selected_indices:
        raise RuntimeError("Subsampling produced 0 samples.")

    return sorted(selected_indices)


# =============================================================================
# Data loaders
# =============================================================================

def get_transforms():
    """Create train/eval transforms using the global IMG_SIZE/normalization."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEAN, DATA_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEAN, DATA_STD),
        ]
    )
    return train_transform, eval_transform


def get_dataloaders_camelyon17(seed: int):
    """Load Camelyon17 via WILDS, apply a custom hospital split, and build loaders."""
    if BYPASS_SSL:
        ssl._create_default_https_context = ssl._create_unverified_context

    print("Loading Camelyon17 (Full dataset)...")
    dataset = get_dataset(
        dataset="camelyon17",
        download=WILDS_DOWNLOAD,
        root_dir=WILDS_ROOT_DIR,
    )

    # Metadata col 0 is the hospital ID (0, 1, 2, 3, 4).
    hospital_ids = dataset.metadata_array[:, 0].long().numpy()

    train_indices: List[int] = []
    val_indices: List[int] = []

    print(f"Splitting Logic -> Val Hospitals: {VAL_HOSPITALS} | Train Hospitals: Rest")

    for idx, hosp_id in enumerate(hospital_ids):
        if int(hosp_id) in VAL_HOSPITALS:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    # Optional subsampling.
    if HOSPITAL_SUBSAMPLE_FRACTION < 1.0:
        print(f"Subsampling Train/Val by {HOSPITAL_SUBSAMPLE_FRACTION}...")
        train_indices = subsample_indices_by_hospital(
            dataset,
            train_indices,
            HOSPITAL_SUBSAMPLE_FRACTION,
            seed,
            HOSPITAL_SUBSAMPLE_MIN_PER_HOSPITAL,
        )
        val_indices = subsample_indices_by_hospital(
            dataset,
            val_indices,
            HOSPITAL_SUBSAMPLE_FRACTION,
            seed + 1,
            HOSPITAL_SUBSAMPLE_MIN_PER_HOSPITAL,
        )

    # Print split stats.
    print_hospital_counts_indices(dataset, train_indices, "Camelyon17 CUSTOM TRAIN")
    print_hospital_counts_indices(dataset, val_indices, "Camelyon17 CUSTOM VAL")

    # Apply transforms.
    train_t, eval_t = get_transforms()
    train_set_wrapped = TransformedWildsSubset(dataset, train_indices, transform=train_t)
    val_set_wrapped = TransformedWildsSubset(dataset, val_indices, transform=eval_t)

    # DataLoaders.
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set_wrapped,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_set_wrapped,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
    )

    return dataset, train_loader, val_loader


# =============================================================================
# Model + train/eval
# =============================================================================

def get_model(num_classes: int) -> nn.Module:
    """Create a DenseNet-121 model and replace the classifier head."""
    model = densenet121(weights=WEIGHTS)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model.to(DEVICE)


def unpack_batch(batch):
    """Unpack a batch from WILDS-style datasets.

    WILDS yields (x, y, metadata). We ignore metadata for standard ERM training.
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        x, y, _meta = batch
        return x, y
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        # Kept for robustness: some datasets/loaders may yield (x, y).
        x, y = batch
        return x, y
    raise ValueError(f"Unexpected batch format: {type(batch)}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch_index: int,
    use_attack: bool,
    cur_step_size: float,
    cur_lambda: float,
) -> Tuple[float, float]:
    """Train the model for a single epoch (optionally with Phase-AT)."""
    model.train()

    total = 0
    correct = 0
    running_loss_sum = 0.0

    loop = tqdm(loader, desc=f"Epoch {epoch_index+1} [Train]", leave=False)

    for batch in loop:
        inputs, targets = unpack_batch(batch)
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_attack:
            # Clean loss.
            logits_clean = model(inputs)
            loss_clean = criterion(logits_clean, targets)

            # Adversarial loss (Phase-AT examples).
            adv_inputs = phase_at_attack(
                model=model,
                x=inputs,
                y=targets,
                criterion=criterion,
                num_steps=PHASE_NUM_STEPS,
                step_size_phi=cur_step_size,
                mask_type=PHASE_MASK_TYPE,
                topk_frac=PHASE_TOPK_FRAC,
                clamp_min=PHASE_CLAMP_MIN,
                clamp_max=PHASE_CLAMP_MAX,
                random_init_phase=PHASE_RANDOM_INIT_PHASE,
                debug=False,
                color_mode=PHASE_COLOR_MODE,
                data_mean=DATA_MEAN,
                data_std=DATA_STD,
                update_mode=PHASE_UPDATE_MODE,
                direction_mode=PHASE_DIRECTION_MODE,
            ).detach()

            logits_adv = model(adv_inputs)
            loss_adv = criterion(logits_adv, targets)
            loss = loss_clean + (cur_lambda * loss_adv)
            logits_for_acc = logits_clean
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
            logits_for_acc = logits

        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        running_loss_sum += loss.item() * bs
        total += bs
        pred = logits_for_acc.argmax(dim=1)
        correct += (pred == targets).sum().item()
        loop.set_postfix(acc=100.0 * correct / max(1, total))

    return running_loss_sum / max(1, total), 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    epoch_index: int,
    split_name: str,
) -> Tuple[float, float]:
    """Evaluate model loss/accuracy on a given dataloader."""
    model.eval()
    total = 0
    correct = 0
    running_loss_sum = 0.0

    loop = tqdm(loader, desc=f"Epoch {epoch_index+1} [Eval {split_name}]", leave=False)

    for batch in loop:
        inputs, targets = unpack_batch(batch)
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        bs = targets.size(0)
        running_loss_sum += loss.item() * bs
        total += bs
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        loop.set_postfix(acc=100.0 * correct / max(1, total))

    return running_loss_sum / max(1, total), 100.0 * correct / max(1, total)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Entry point: parse args, build loaders/model, then run training."""
    args = parse_args()

    # -------------------------------------------------------------------------
    # Bind all configuration into this module's globals (minimal code changes)
    # -------------------------------------------------------------------------
    global SEED, CUDA_DEVICE, DEVICE
    global BATCH_SIZE, LEARNING_RATE, EPOCHS, NUM_WORKERS
    global IMG_SIZE, DATA_MEAN, DATA_STD
    global WILDS_ROOT_DIR, WILDS_DOWNLOAD, BYPASS_SSL
    global VAL_HOSPITALS, HOSPITAL_SUBSAMPLE_FRACTION, HOSPITAL_SUBSAMPLE_MIN_PER_HOSPITAL
    global USE_IMAGENET_PRETRAIN, WEIGHTS
    global USE_PHASE_AT, PHASE_COLOR_MODE, PHASE_NUM_STEPS, PHASE_MASK_TYPE, PHASE_TOPK_FRAC, PHASE_RANDOM_INIT_PHASE
    global PHASE_UPDATE_MODE, PHASE_DIRECTION_MODE
    global PHASE_CLAMP_MIN, PHASE_CLAMP_MAX
    global PHASE_WARMUP_EPOCHS, PHASE_RAMPUP_EPOCHS
    global PHASE_STEP_SIZE_INIT, PHASE_STEP_SIZE_MAX
    global PHASE_LAMBDA_INIT, PHASE_LAMBDA_MAX
    global SAVE_PATH

    SEED = int(args.seed)
    CUDA_DEVICE = str(args.device)
    if CUDA_DEVICE.lower() == "cpu":
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = int(args.batch_size)
    LEARNING_RATE = float(args.lr)
    EPOCHS = int(args.epochs)
    NUM_WORKERS = int(args.num_workers)

    IMG_SIZE = int(args.img_size)
    DATA_MEAN = tuple(args.data_mean)
    DATA_STD = tuple(args.data_std)

    WILDS_ROOT_DIR = str(args.wilds_root_dir)
    WILDS_DOWNLOAD = bool(args.wilds_download)
    BYPASS_SSL = bool(args.bypass_ssl)

    VAL_HOSPITALS = list(args.val_hospitals)
    HOSPITAL_SUBSAMPLE_FRACTION = float(args.hospital_subsample_fraction)
    HOSPITAL_SUBSAMPLE_MIN_PER_HOSPITAL = int(args.hospital_subsample_min_per_hospital)

    USE_IMAGENET_PRETRAIN = bool(args.use_imagenet_pretrain)
    WEIGHTS = DenseNet121_Weights.IMAGENET1K_V1 if USE_IMAGENET_PRETRAIN else None

    USE_PHASE_AT = bool(args.use_phase_at)
    PHASE_COLOR_MODE = str(args.color_mode)
    PHASE_NUM_STEPS = int(args.num_steps)
    PHASE_MASK_TYPE = str(args.mask_type)
    PHASE_TOPK_FRAC = float(args.topk_frac)
    PHASE_RANDOM_INIT_PHASE = bool(args.random_init_phase)

    PHASE_UPDATE_MODE = str(args.update_mode)
    PHASE_DIRECTION_MODE = str(args.direction_mode)

    PHASE_CLAMP_MIN = float(args.clamp_min)
    PHASE_CLAMP_MAX = float(args.clamp_max)

    PHASE_WARMUP_EPOCHS = int(args.warmup_epochs)
    PHASE_RAMPUP_EPOCHS = int(args.rampup_epochs)
    PHASE_STEP_SIZE_INIT = float(args.step_size_init)
    PHASE_STEP_SIZE_MAX = float(args.step_size_max)
    PHASE_LAMBDA_INIT = float(args.lambda_init)
    PHASE_LAMBDA_MAX = float(args.lambda_max)

    # Save path suffixing.
    SAVE_PATH = str(args.save_path)
    if args.run_name:
        save_dir = os.path.dirname(SAVE_PATH) or "."
        base_name = os.path.basename(SAVE_PATH)
        root, ext = os.path.splitext(base_name)
        if ext == "":
            ext = ".pth"
        SAVE_PATH = os.path.join(save_dir, f"{root}_{args.run_name}{ext}")

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    print("Dataset: camelyon17")
    print(f"Val Hospitals: {VAL_HOSPITALS}")

    wilds_dataset, train_loader, val_loader = get_dataloaders_camelyon17(SEED)
    num_classes = int(wilds_dataset.n_classes)

    model = get_model(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Monitor Val Accuracy.
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10)

    print("\n=== Phase-AT settings ===")
    print(f"USE_PHASE_AT={USE_PHASE_AT}")
    print(f"PHASE_COLOR_MODE={PHASE_COLOR_MODE}")
    print(f"PHASE_UPDATE_MODE={PHASE_UPDATE_MODE}")
    print(f"PHASE_DIRECTION_MODE={PHASE_DIRECTION_MODE}")
    print(f"PHASE_MASK_TYPE={PHASE_MASK_TYPE}, PHASE_TOPK_FRAC={PHASE_TOPK_FRAC}")
    print(f"PHASE_NUM_STEPS={PHASE_NUM_STEPS}")
    print(f"PHASE_CLAMP_MIN/MAX=({PHASE_CLAMP_MIN}, {PHASE_CLAMP_MAX})")
    print(f"SAVE_PATH={SAVE_PATH}")
    print("========================\n")

    best_val_acc = -1.0

    for epoch in range(EPOCHS):
        cur_step, cur_lambda = phase_scheduler(
            epoch=epoch,
            warmup_epochs=PHASE_WARMUP_EPOCHS,
            rampup_epochs=PHASE_RAMPUP_EPOCHS,
            step_size_max=PHASE_STEP_SIZE_MAX,
            lambda_adv_max=PHASE_LAMBDA_MAX,
            step_size_init=PHASE_STEP_SIZE_INIT,
            lambda_adv_init=PHASE_LAMBDA_INIT,
        )

        use_attack = bool(USE_PHASE_AT and cur_lambda > 0.0 and cur_step > 0.0)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch_index=epoch,
            use_attack=use_attack,
            cur_step_size=cur_step,
            cur_lambda=cur_lambda,
        )

        # Eval on Val.
        val_loss, val_acc = evaluate(model, val_loader, criterion, epoch, "VAL")
        lr_scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}% | "
            f"PhaseAT(step={cur_step:.4f}, lam={cur_lambda:.3f}) | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Saved new best model (val_acc={best_val_acc:.2f}%) to {SAVE_PATH}")

    print("\nTraining complete.")
    print(f"Best Val Acc: {best_val_acc:.2f}% | Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
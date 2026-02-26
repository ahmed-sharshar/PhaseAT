"""Argument/configuration definitions for `main.py`.

This file holds *all* tunable configuration values ("arguments") and the CLI
parser. `main.py` imports :func:`parse_args` from here and contains the training
logic.

The user-requested interface is supported:
    python main.py --args

`--args` is a no-op flag whose only purpose is to make that exact command valid.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple


# =============================================================================
# Defaults (edit here)
# =============================================================================

SEED: int = 42

# Device
CUDA_DEVICE: str = "cuda:4"

# Training
BATCH_SIZE: int = 128
LEARNING_RATE: float = 1e-4
EPOCHS: int = 20
NUM_WORKERS: int = 4

# Image / normalization
IMG_SIZE: int = 96
DATA_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DATA_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Camelyon17 (WILDS)
WILDS_DOWNLOAD: bool = False
BYPASS_SSL: bool = True

# Validation split configuration
VAL_HOSPITALS: List[int] = [0]
HOSPITAL_SUBSAMPLE_FRACTION: float = 1
HOSPITAL_SUBSAMPLE_MIN_PER_HOSPITAL: int = 1

# Model
USE_IMAGENET_PRETRAIN: bool = False

# Phase-AT (frequency-domain adversarial training)
USE_PHASE_AT: bool = True
PHASE_COLOR_MODE: str = "rgb"  # "rgb" | "ycbcr" | "ycbcr_all"
PHASE_NUM_STEPS: int = 5
PHASE_MASK_TYPE: str = "soft"  # "soft" | "hard" | "none"

# Update/direction modes (ablations)
PHASE_UPDATE_MODE: str = "phase"         # "phase" | "amplitude" | "complex"
PHASE_DIRECTION_MODE: str = "adversarial"  # "adversarial" | "random"

# Input clamp range (normalized space)
PHASE_CLAMP_MIN: float = -2.0
PHASE_CLAMP_MAX: float = 2.6

# Phase-AT curriculum
PHASE_WARMUP_EPOCHS: int = 5
PHASE_RAMPUP_EPOCHS: int = 15
PHASE_STEP_SIZE_INIT: float = 0.1
PHASE_STEP_SIZE_MAX: float = 0.4
PHASE_LAMBDA_INIT: float = 0.10
PHASE_LAMBDA_MAX: float = 0.7

# Output
SAVE_PATH: str = "densenet121_best.pth"


# =============================================================================
# CLI helpers
# =============================================================================

def _parse_int_list(s: str) -> List[int]:
    """Parse a comma-separated string into a list of integers.

    Examples:
        "3" -> [3]
        "0,1,2" -> [0, 1, 2]
        "" -> []

    Args:
        s: Input string.

    Returns:
        Parsed list of ints.
    """
    s = (s or "").strip()
    if s == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser used by `main.py`."""
    parser = argparse.ArgumentParser(
        description="Train Camelyon17 models with optional Phase-AT (frequency-domain adversarial training)."
    )

    # User-requested interface: allow `python main.py --args`.
    parser.add_argument(
        "--args",
        action="store_true",
        help="No-op flag. Provided only so `python main.py --args` is a valid command.",
    )

    # Core
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=CUDA_DEVICE, help="e.g., cuda:0 or cpu")

    # Training
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)

    # Normalization (kept configurable for completeness)
    parser.add_argument("--data_mean", type=str, default=",".join(str(x) for x in DATA_MEAN))
    parser.add_argument("--data_std", type=str, default=",".join(str(x) for x in DATA_STD))

    # WILDS
    parser.add_argument("--wilds_root_dir", type=str, required=True, help="Input path to camelyon17 dataset")
    parser.add_argument("--wilds_download", type=int, default=int(WILDS_DOWNLOAD), help="0 or 1")
    parser.add_argument("--bypass_ssl", type=int, default=int(BYPASS_SSL), help="0 or 1")

    # Camelyon split/subsample
    parser.add_argument(
        "--val_hospitals",
        type=str,
        default=",".join(str(h) for h in VAL_HOSPITALS),
        help="Comma-separated hospital ids for validation (e.g., 3 or 0,1,2,4).",
    )
    parser.add_argument("--hospital_subsample_fraction", type=float, default=HOSPITAL_SUBSAMPLE_FRACTION)
    parser.add_argument(
        "--hospital_subsample_min_per_hospital",
        type=int,
        default=HOSPITAL_SUBSAMPLE_MIN_PER_HOSPITAL,
    )

    # Model
    parser.add_argument("--use_imagenet_pretrain", type=int, default=int(0), help="0 or 1")

    # Phase-AT toggles
    parser.add_argument("--use_phase_at", type=int, default=int(USE_PHASE_AT), help="0 or 1")
    parser.add_argument("--color_mode", type=str, default=PHASE_COLOR_MODE, choices=["rgb", "ycbcr", "ycbcr_all"])
    parser.add_argument("--num_steps", type=int, default=PHASE_NUM_STEPS)
    parser.add_argument("--mask_type", type=str, default=PHASE_MASK_TYPE, choices=["soft", "hard", "none"])

    parser.add_argument("--update_mode", type=str, default=PHASE_UPDATE_MODE, choices=["phase", "amplitude", "complex"])
    parser.add_argument("--direction_mode", type=str, default=PHASE_DIRECTION_MODE, choices=["adversarial", "random"])

    parser.add_argument("--clamp_min", type=float, default=PHASE_CLAMP_MIN)
    parser.add_argument("--clamp_max", type=float, default=PHASE_CLAMP_MAX)

    # Curriculum
    parser.add_argument("--warmup_epochs", type=int, default=PHASE_WARMUP_EPOCHS)
    parser.add_argument("--rampup_epochs", type=int, default=PHASE_RAMPUP_EPOCHS)
    parser.add_argument("--step_size_init", type=float, default=PHASE_STEP_SIZE_INIT)
    parser.add_argument("--step_size_max", type=float, default=PHASE_STEP_SIZE_MAX)
    parser.add_argument("--lambda_init", type=float, default=PHASE_LAMBDA_INIT)
    parser.add_argument("--lambda_max", type=float, default=PHASE_LAMBDA_MAX)

    # Output
    parser.add_argument("--save_path", type=str, default=SAVE_PATH)
    parser.add_argument("--run_name", type=str, default="", help="Optional suffix for save_path (e.g., ERM_C3).")

    return parser


def parse_args() -> argparse.Namespace:
    """Parse CLI args and return a Namespace with post-processed fields."""
    parser = build_parser()
    args = parser.parse_args()

    # Convert comma-separated mean/std strings into tuples of floats.
    args.data_mean = tuple(float(x.strip()) for x in str(args.data_mean).split(",") if x.strip() != "")
    args.data_std = tuple(float(x.strip()) for x in str(args.data_std).split(",") if x.strip() != "")

    # Convert validation hospitals string to a list[int].
    args.val_hospitals = _parse_int_list(args.val_hospitals)

    # Convert 0/1 toggles into bools.
    args.wilds_download = bool(args.wilds_download)
    args.bypass_ssl = bool(args.bypass_ssl)
    args.use_imagenet_pretrain = bool(args.use_imagenet_pretrain)

    args.use_phase_at = bool(args.use_phase_at)
    args.random_init_phase = bool(args.random_init_phase)

    return args
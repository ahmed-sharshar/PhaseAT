# This is the official implementation of PhaseAT: Fourier Phase Adversarial Training for Medical Image Domain Generalization (MICCAI 2026)
## Camelyon17 Training with Phase-AT <img width="1442" height="514" alt="phase_at" src="https://github.com/user-attachments/assets/665988b1-407d-4778-8a99-3efd47de4fed" />


### Files

- **`main.py`**: training entrypoint (Camelyon17 only for now)
- **`args.py`**: all configuration + CLI arguments
- **`phase_at.py`**: Phase-AT mai algorithms attack

### Requirements

Python 3.9+ recommended.

Install dependencies:

```bash
pip install torch torchvision tqdm wilds numpy
````

> If you use CUDA, install the correct PyTorch build for your CUDA version (recommended via the official PyTorch install command).

### Dataset setup (WILDS Camelyon17)

You can either:

1. **Download using WILDS**, or
2. **Point to an existing WILDS dataset directory**

The code uses `--wilds_root_dir` to locate data.

### Run

The repository supports the exact command:

```bash
python main.py --args
```

### Common run examples

#### 1) Run with an existing dataset folder

```bash
python main.py --args --wilds_root_dir /path/to/wilds_data
```

#### 2) Download Camelyon17 via WILDS (if your machine has internet access)

```bash
python main.py --args --wilds_download 1 --wilds_root_dir ./wilds_data
```

#### 3) Choose device

```bash
python main.py --args --device cuda:0
# or
python main.py --args --device cpu
```

#### 4) Change training hyperparameters

```bash
python main.py --args --epochs 20 --batch_size 128 --lr 1e-4
```

#### 5) Change validation hospitals (domain split)

`--val_hospitals` is a comma-separated list of hospital IDs (0–4).

```bash
python main.py --args --val_hospitals 0
python main.py --args --val_hospitals 3
python main.py --args --val_hospitals 0,1
```

## Phase-AT 

Phase-AT is enabled by default. To disable it (clean ERM training):

```bash
python main.py --args --use_phase_at 0
```

Key Phase-AT parameters (optional overrides):

```bash
python main.py --args \
  --use_phase_at 1 \
  --color_mode ycbcr \
  --update_mode phase \
  --direction_mode adversarial \
  --num_steps 5 \
  --mask_type soft \
```

## Output

By default, the best checkpoint (based on validation accuracy) is saved to:

* `densenet121_best.pth`

You can change it:

```bash
python main.py --args --save_path checkpoints/best.pth
```

Optionally add a run name suffix:

```bash
python main.py --args --run_name EXP1
# saves as: checkpoints/best_EXP1.pth (or densenet121_best_EXP1.pth)
```


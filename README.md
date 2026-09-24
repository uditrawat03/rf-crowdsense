# RF CrowdSense

RF CrowdSense is a GPU-oriented research repo for estimating **aggregate RF activity** from synthetic or explicitly authorized IQ data. It uses PyTorch, torchvision and an optional TensorFlow nightly path for Python 3.14.

It does not decode communications, recover subscriber identifiers, track individual phones, or bypass cellular security.

## v0.3 stack

- Python 3.14 (latest 3.14.x selected by `uv`)
- PyTorch 2.14.0
- torchvision 0.29.0
- CUDA 13.2 PyTorch wheels
- TensorFlow `tf-nightly` 2.22 for Python 3.14 (optional)
- NumPy-only IQ to spectrogram preprocessing
- Custom RF CNN and torchvision ResNet-18
- CUDA AMP training
- Aggregate activity-range classification and normalized activity regression

## Windows quick start with uv

Open PowerShell in the extracted project:

```powershell
cd C:\working\Projects\Python\rf-crowdsense

uv python install 3.14
uv python pin 3.14
uv sync --extra pytorch --extra dev
```

Or run the included setup script:

```powershell
.\scripts\setup-windows.ps1 -Reset
```

`-Reset` removes an old `.venv` before syncing. Omit it if this is a fresh extraction.

Verify the environment:

```powershell
uv run python --version
uv run rfcrowd doctor
```

For an NVIDIA CUDA 13.2 machine you should see PyTorch CUDA available and `torchvision` 0.29.x.

## Why Python starts at 3.14.2

`torchvision` 0.29 excludes Python 3.14.1. The repo therefore requires:

```toml
requires-python = ">=3.14.2,<3.15"
```

The `.python-version` file contains `3.14`, so `uv` selects the latest available 3.14 patch release rather than an old 3.14.1 interpreter.

## Generate synthetic RF data

```powershell
uv run rfcrowd generate-dataset `
  --output .\data\synthetic-v0.2 `
  --samples 2000 `
  --seed 42
```

The generator varies transmitter count, burst timing, carrier offset, fading and SNR. Every sample contains complex IQ plus aggregate labels.

The activity classes are:

```text
0-5
6-20
21-50
51-100
101+
```

## Prepare a cached spectrogram dataset

Milestone 1 adds a disk-backed spectrogram cache and deterministic train/validation/test splits. This avoids recomputing the STFT for every sample on every epoch.

```powershell
uv run rfcrowd prepare-dataset `
  --dataset .\data\synthetic-v0.2 `
  --seed 42
```

By default this creates:

```text
data/synthetic-v0.2/cache/
├── cache.json
├── spectrograms.npy
├── scores.npy
├── counts.npy
├── classes.npy
└── splits.npz
```

`spectrograms.npy` is written in NumPy `.npy` format and opened with memory mapping during training. The split indices are deterministic for the same seed. If the source manifest or preprocessing settings change, rebuild explicitly:

```powershell
uv run rfcrowd prepare-dataset `
  --dataset .\data\synthetic-v0.2 `
  --overwrite
```

## Train the custom PyTorch CNN

```powershell
uv run rfcrowd train-pytorch `
  --dataset .\data\synthetic-v0.2 `
  --model cnn `
  --epochs 10 `
  --batch-size 64
```

CUDA AMP is enabled automatically when CUDA is available. Disable it with `--no-amp`. If `data/synthetic-v0.2/cache/cache.json` exists, training uses the cached spectrograms automatically and reports train/validation/test split sizes.

## Train torchvision ResNet-18

This is why `torchvision` is included in the project.

```powershell
uv run rfcrowd train-pytorch `
  --dataset .\data\synthetic-v0.2 `
  --model resnet18 `
  --epochs 10 `
  --batch-size 32
```

The standard ResNet-18 first convolution is changed from three image channels to one spectrogram channel. No pretrained weights are downloaded.

## GPU benchmark

```powershell
uv run rfcrowd benchmark-gpu --size 4096 --iterations 20
```

Preprocessing benchmark:

```powershell
uv run rfcrowd benchmark --iterations 200
```

## TensorFlow on Python 3.14

TensorFlow 2.21 stable does not provide Python 3.14 wheels. This repo therefore keeps TensorFlow out of the default install and provides a Python 3.14 `tf-nightly` extra.

Native Windows TensorFlow:

```powershell
.\scripts\setup-tensorflow-nightly-windows.ps1
```

or:

```powershell
uv sync --extra pytorch --extra tensorflow --extra dev
```

Modern TensorFlow on native Windows does not use NVIDIA CUDA. This path is useful for CPU comparison only.

For TensorFlow GPU training, use WSL2 and run:

```bash
./scripts/setup-wsl2.sh
```

That installs the Linux TensorFlow nightly CUDA extra together with the CUDA 13.2 PyTorch build.

## TorchStudio

TorchStudio is not a required Python dependency. See `docs/TORCHSTUDIO.md` for why it is kept separate and how to use the standalone GUI if you want it.

## API

Install API dependencies:

```powershell
uv sync --extra pytorch --extra api --extra dev
```

Run:

```powershell
uv run uvicorn rf_crowdsense.api.app:app --reload
```

## Safety scope

In scope:

- synthetic RF waveforms
- public research datasets with compatible licenses
- RF captures you are authorized to collect
- aggregate spectrum occupancy and activity estimation

Out of scope:

- IMSI or IMEI extraction
- phone number or subscriber identification
- call/message interception
- bypassing cellular security
- locating or tracking specific people or devices

## Milestones

### Milestone 1: cached dataset pipeline

- disk-backed spectrogram cache
- deterministic train/validation/test splits
- automatic cache detection in PyTorch training
- held-out test metrics saved into the checkpoint
- cache reuse/staleness checks based on the source manifest and preprocessing settings

## Suggested next milestones

1. Add calibration for count ranges and confidence intervals.
2. Add ONNX export for the PyTorch models.
3. Benchmark FP32 vs FP16/BF16 on the RTX 5050 Laptop GPU.
4. Add legal/public RF datasets to test synthetic-to-real domain shift.
5. Add experiment tracking and reproducible benchmark reports.

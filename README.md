# RF CrowdSense

RF CrowdSense is a GPU-oriented research repo for estimating **aggregate RF activity** from synthetic or explicitly authorized IQ data. It uses PyTorch, torchvision and an optional TensorFlow nightly path for Python 3.14.

It does not decode communications, recover subscriber identifiers, track individual phones, or bypass cellular security.

## v0.5 stack

- Python 3.14 (latest 3.14.x selected by `uv`)
- PyTorch 2.14.0
- torchvision 0.29.0
- CUDA 13.2 PyTorch wheels
- TensorFlow `tf-nightly` 2.22 for Python 3.14 (optional)
- NumPy-only IQ to spectrogram preprocessing
- Custom RF CNN and torchvision ResNet-18
- CUDA AMP training
- Aggregate activity-range classification and normalized activity regression
- Aggregate transmitter-count estimation derived from normalized activity
- Split-conformal count intervals calibrated on the validation split
- Single-sample PyTorch inference CLI
- ONNX export using the PyTorch `torch.export`-based exporter
- ONNX Runtime CPU/CUDA inference
- PyTorch-vs-ONNX model-inference benchmarking

## Windows quick start with uv

Open PowerShell in the extracted project:

```powershell
cd C:\working\Projects\Python\rf-crowdsense

uv python install 3.14
uv python pin 3.14
uv sync --extra pytorch --extra dev
```

For ONNX export and ONNX Runtime benchmarking, include the `onnx` extra:

```powershell
uv sync --extra pytorch --extra onnx --extra dev
```

Or run the included setup script:

```powershell
.\scripts\setup-windows.ps1 -Reset

# Include ONNX dependencies too
.\scripts\setup-windows.ps1 -Reset -WithOnnx
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
  --batch-size 64 `
  --count-coverage 0.90
```

CUDA AMP is enabled automatically when CUDA is available. Disable it with `--no-amp`. If `data/synthetic-v0.2/cache/cache.json` exists, training uses the cached spectrograms automatically and reports train/validation/test split sizes.

Milestone 2 also reports aggregate transmitter-count MAE. The normalized activity head is converted back to a count using the dataset's configured `max_devices` value. The best checkpoint is then calibrated on the validation split with absolute-residual split conformal prediction. `--count-coverage 0.90` requests a 90% **dataset-level nominal coverage target** for the count interval. It is not a 90% probability statement for an individual sample.


## Predict aggregate transmitter count

After training, run inference against one authorized/synthetic `.npz` IQ sample:

```powershell
uv run rfcrowd predict-pytorch `
  --checkpoint .\artifacts\pytorch_activity.pt `
  --sample .\data\synthetic-v0.2\sample_000000.npz `
  --device cuda
```

Example shape of the JSON output:

```json
{
  "activity_score": 0.31,
  "estimated_active_transmitters": 37.2,
  "activity_class": {
    "index": 2,
    "label": "21-50",
    "probability": 0.81
  },
  "count_interval": {
    "lower": 29.4,
    "upper": 45.0,
    "nominal_coverage": 0.9,
    "method": "split_conformal_absolute_residual"
  }
}
```

The exact values depend on the trained checkpoint. The class softmax probability is reported separately and is not calibrated in this milestone.

### How the count interval works

RF CrowdSense uses the validation split only to estimate the conformal residual radius:

```text
validation truth + validation predictions
                  ↓
          absolute residuals
                  ↓
       conformal residual quantile
                  ↓
 predicted count ± calibrated radius
```

The held-out test split is then used to report count MAE and observed interval coverage. This keeps interval calibration separate from final test evaluation.

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

## Export a PyTorch checkpoint to ONNX

Install the ONNX extra first:

```powershell
uv sync --extra pytorch --extra onnx --extra dev
```

Export a trained checkpoint. The sample is used only to derive the spectrogram shape expected by the model:

```powershell
uv run rfcrowd export-onnx `
  --checkpoint .\artifacts\pytorch_activity.pt `
  --sample .\data\synthetic-v0.2\sample_000000.npz `
  --output .\artifacts\pytorch_activity.onnx `
  --max-batch 256
```

The exporter uses the modern `torch.export`-based ONNX path, validates the generated ONNX model, and by default compares PyTorch and ONNX Runtime CPU outputs before accepting the export. It writes:

```text
artifacts/
├── pytorch_activity.onnx
└── pytorch_activity.onnx.json
```

The JSON sidecar keeps the preprocessing settings, activity labels, count scale, conformal count calibration, checkpoint SHA-256 and ONNX input/output metadata required for reproducible inference. The exported graph has a dynamic batch dimension up to `--max-batch`; spectrogram frequency/time dimensions remain fixed to the preprocessing configuration.

Skip numerical export verification only when diagnosing an environment problem:

```powershell
uv run rfcrowd export-onnx `
  --checkpoint .\artifacts\pytorch_activity.pt `
  --sample .\data\synthetic-v0.2\sample_000000.npz `
  --no-verify
```

## Run ONNX Runtime inference

```powershell
uv run rfcrowd predict-onnx `
  --model .\artifacts\pytorch_activity.onnx `
  --sample .\data\synthetic-v0.2\sample_000000.npz `
  --provider cuda
```

Provider choices are `auto`, `cuda`, and `cpu`. `auto` prefers `CUDAExecutionProvider` when available and otherwise falls back to CPU. On Windows, the runtime tries to reuse the CUDA/cuDNN DLLs already provided by the PyTorch CUDA installation before creating the ONNX Runtime session.

`rfcrowd doctor` now also reports the installed ONNX Runtime version and available execution providers.

## Compare PyTorch and ONNX Runtime inference

The benchmark reuses one already-preprocessed spectrogram, so it measures model inference rather than NumPy STFT preprocessing:

```powershell
uv run rfcrowd benchmark-inference `
  --checkpoint .\artifacts\pytorch_activity.pt `
  --onnx-model .\artifacts\pytorch_activity.onnx `
  --sample .\data\synthetic-v0.2\sample_000000.npz `
  --device cuda `
  --provider cuda `
  --batch-size 1 `
  --warmup 20 `
  --iterations 200
```

The JSON report includes mean, p50 and p95 latency, throughput, and ONNX Runtime mean-latency speedup relative to PyTorch eager execution. Try batch sizes such as `1`, `8`, `32`, and `64`, staying below the `--max-batch` used during export.

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

### Milestone 2: aggregate count calibration

- convert normalized activity predictions back to aggregate transmitter counts
- report validation and held-out test count MAE
- fit split-conformal absolute-residual intervals on the validation split
- save calibration metadata and held-out interval coverage into checkpoints
- add `predict-pytorch` for single-sample count/range inference
- persist preprocessing and count-scale metadata required for reproducible inference

### Milestone 3: ONNX export and runtime benchmarking

- export CNN and ResNet-18 checkpoints through the modern PyTorch ONNX exporter
- validate exported ONNX graphs and check PyTorch/ORT numerical parity
- persist deployment metadata next to each ONNX model
- add ONNX Runtime CPU/CUDA single-sample inference
- reuse PyTorch CUDA libraries when ONNX Runtime initializes on Windows
- add dynamic ONNX batch support up to a configured maximum
- benchmark PyTorch eager vs ONNX Runtime with mean/p50/p95 latency and throughput
- extend `rfcrowd doctor` with ONNX Runtime provider diagnostics

## Suggested next milestones

1. Benchmark and optimize FP32 vs FP16/BF16 on the RTX 5050 Laptop GPU.
2. Add probability calibration for the activity-class head.
3. Add TensorRT export/engine benchmarking after the ONNX baseline is measured.
4. Add legal/public RF datasets to test synthetic-to-real domain shift.
5. Add experiment tracking and reproducible benchmark reports.

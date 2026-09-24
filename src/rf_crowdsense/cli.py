from __future__ import annotations

from pathlib import Path
import json

import typer
from rich import print

from .benchmark.device_info import collect as collect_devices
from .benchmark.preprocessing import gpu_matmul, run as run_preprocessing_benchmark
from .data import build_spectrogram_cache
from .generator.synthetic import GeneratorConfig, generate_dataset

app = typer.Typer(no_args_is_help=True, help="Aggregate RF activity experiments.")


@app.command("generate-dataset")
def generate_dataset_cmd(
    output: Path = typer.Option(Path("data/synthetic-v0.2")),
    samples: int = typer.Option(1000, min=1),
    seed: int = typer.Option(42),
    max_devices: int = typer.Option(120, min=1),
):
    cfg = GeneratorConfig(max_devices=max_devices)
    summary = generate_dataset(output=output, samples=samples, seed=seed, cfg=cfg)
    print(json.dumps(summary, indent=2))


@app.command("prepare-dataset")
def prepare_dataset_cmd(
    dataset: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path | None = typer.Option(None, help="Defaults to <dataset>/cache"),
    seed: int = typer.Option(42),
    train_ratio: float = typer.Option(0.8, min=0.01, max=0.98),
    validation_ratio: float = typer.Option(0.1, min=0.01, max=0.98),
    nperseg: int = typer.Option(256, min=8),
    overlap: float = typer.Option(0.5, min=0.0, max=0.99),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    summary = build_spectrogram_cache(
        dataset,
        output,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        nperseg=nperseg,
        overlap=overlap,
        overwrite=overwrite,
    )
    print(json.dumps(summary, indent=2))


@app.command("devices")
def devices_cmd():
    print(json.dumps(collect_devices(), indent=2))


@app.command("doctor")
def doctor_cmd():
    print(json.dumps(collect_devices(), indent=2))


@app.command("benchmark")
def benchmark_cmd(iterations: int = typer.Option(200, min=1)):
    print(json.dumps(run_preprocessing_benchmark(iterations), indent=2))


@app.command("benchmark-gpu")
def benchmark_gpu_cmd(
    size: int = typer.Option(4096, min=256),
    iterations: int = typer.Option(20, min=1),
):
    print(json.dumps(gpu_matmul(size, iterations), indent=2))


@app.command("train-pytorch")
def train_pytorch_cmd(
    dataset: Path = typer.Option(..., exists=True, file_okay=False),
    epochs: int = typer.Option(10, min=1),
    batch_size: int = typer.Option(32, min=1),
    model: str = typer.Option("cnn", help="cnn or resnet18"),
    amp: bool = typer.Option(True, "--amp/--no-amp"),
    amp_dtype: str = typer.Option("fp16", help="fp16 or bf16 when AMP is enabled"),
    seed: int = typer.Option(42),
    count_coverage: float = typer.Option(
        0.90,
        min=0.5,
        max=0.999,
        help="Nominal split-conformal coverage for aggregate-count intervals",
    ),
    output: Path = typer.Option(Path("artifacts/pytorch_activity.pt")),
    cache: Path | None = typer.Option(
        None, help="Prepared cache directory; defaults to <dataset>/cache when present"
    ),
):
    from .training.pytorch_train import train

    path = train(
        dataset,
        epochs,
        batch_size,
        output,
        model_name=model,
        seed=seed,
        amp=amp,
        amp_dtype=amp_dtype,
        cache=cache,
        count_coverage=count_coverage,
    )
    print(f"saved: {path}")


@app.command("predict-pytorch")
def predict_pytorch_cmd(
    checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    sample: Path = typer.Option(..., exists=True, dir_okay=False),
    device: str = typer.Option("auto", help="auto, cpu, or cuda"),
):
    from .inference.pytorch_predict import predict_sample

    result = predict_sample(checkpoint, sample, device=device)
    print(json.dumps(result, indent=2))


@app.command("export-onnx")
def export_onnx_cmd(
    checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    sample: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(Path("artifacts/pytorch_activity.onnx")),
    opset: int = typer.Option(18, min=17),
    max_batch: int = typer.Option(256, min=2),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
):
    from .deployment.onnx_export import export_checkpoint

    result = export_checkpoint(
        checkpoint,
        sample,
        output,
        opset=opset,
        max_batch=max_batch,
        verify=verify,
    )
    print(json.dumps(result, indent=2))


@app.command("predict-onnx")
def predict_onnx_cmd(
    model: Path = typer.Option(..., exists=True, dir_okay=False),
    sample: Path = typer.Option(..., exists=True, dir_okay=False),
    provider: str = typer.Option("auto", help="auto, cpu, or cuda"),
):
    from .inference.onnx_predict import predict_sample_onnx

    result = predict_sample_onnx(model, sample, provider=provider)
    print(json.dumps(result, indent=2))


@app.command("benchmark-inference")
def benchmark_inference_cmd(
    checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    onnx_model: Path = typer.Option(..., exists=True, dir_okay=False),
    sample: Path = typer.Option(..., exists=True, dir_okay=False),
    device: str = typer.Option("auto", help="PyTorch: auto, cpu, or cuda"),
    provider: str = typer.Option("auto", help="ONNX Runtime: auto, cpu, or cuda"),
    batch_size: int = typer.Option(1, min=1),
    warmup: int = typer.Option(20, min=0),
    iterations: int = typer.Option(100, min=1),
):
    from .benchmark.inference import benchmark_engines

    result = benchmark_engines(
        checkpoint,
        onnx_model,
        sample,
        device=device,
        provider=provider,
        batch_size=batch_size,
        warmup=warmup,
        iterations=iterations,
    )
    print(json.dumps(result, indent=2))


@app.command("benchmark-precision")
def benchmark_precision_cmd(
    checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    sample: Path = typer.Option(..., exists=True, dir_okay=False),
    device: str = typer.Option("cuda", help="auto, cpu, or cuda"),
    batch_size: int = typer.Option(1, min=1),
    warmup: int = typer.Option(20, min=0),
    iterations: int = typer.Option(100, min=1),
    precisions: str = typer.Option(
        "fp32,fp16,bf16",
        help="Comma-separated precision modes to benchmark",
    ),
    output: Path | None = typer.Option(None, help="Optional JSON report path"),
):
    from .benchmark.precision import benchmark_precision_modes

    result = benchmark_precision_modes(
        checkpoint,
        sample,
        device=device,
        batch_size=batch_size,
        warmup=warmup,
        iterations=iterations,
        precisions=precisions,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["report"] = str(output.resolve())
    print(json.dumps(result, indent=2))


@app.command("train-tensorflow")
def train_tensorflow_cmd(
    dataset: Path = typer.Option(..., exists=True, file_okay=False),
    epochs: int = typer.Option(10, min=1),
    batch_size: int = typer.Option(32, min=1),
    output: Path = typer.Option(Path("artifacts/tensorflow_activity.keras")),
):
    from .training.tensorflow_train import train

    path = train(dataset, epochs, batch_size, output)
    print(f"saved: {path}")


if __name__ == "__main__":
    app()

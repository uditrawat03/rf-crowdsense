from __future__ import annotations

from pathlib import Path
import json

import typer
from rich import print

from .benchmark.device_info import collect as collect_devices
from .benchmark.preprocessing import gpu_matmul, run as run_preprocessing_benchmark
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
    output: Path = typer.Option(Path("artifacts/pytorch_activity.pt")),
):
    from .training.pytorch_train import train

    path = train(dataset, epochs, batch_size, output, model_name=model, amp=amp)
    print(f"saved: {path}")


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

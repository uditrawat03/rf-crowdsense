param(
    [switch]$Reset,
    [switch]$WithOnnx
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "Installing the latest Python 3.14 managed by uv..."
uv python install 3.14
uv python pin 3.14

if ($Reset -and (Test-Path ".venv")) {
    Write-Host "Removing existing .venv..."
    Remove-Item -Recurse -Force ".venv"
}

if ($WithOnnx) {
    Write-Host "Syncing PyTorch CUDA + ONNX export/runtime dependencies..."
    uv sync --extra pytorch --extra onnx --extra dev
} else {
    Write-Host "Syncing PyTorch 2.14 + torchvision 0.29 + CUDA 13.2..."
    uv sync --extra pytorch --extra dev
}

Write-Host "Environment check:"
uv run rfcrowd doctor

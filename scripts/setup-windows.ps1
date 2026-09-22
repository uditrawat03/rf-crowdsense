param(
    [switch]$Reset
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

Write-Host "Syncing PyTorch 2.14 + torchvision 0.29 + CUDA 13.2..."
uv sync --extra pytorch --extra dev

Write-Host "Environment check:"
uv run rfcrowd doctor

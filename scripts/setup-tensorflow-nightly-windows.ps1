$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "Installing TensorFlow nightly for Python 3.14."
Write-Host "Native Windows TensorFlow will run on CPU; use WSL2 for NVIDIA GPU support."
uv sync --extra pytorch --extra tensorflow --extra dev
uv run rfcrowd doctor

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

uv python install 3.14
uv python pin 3.14
uv sync --extra pytorch --extra tensorflow-gpu --extra dev
uv run rfcrowd doctor

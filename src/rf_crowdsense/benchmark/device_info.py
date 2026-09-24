from __future__ import annotations

import platform
import subprocess
import sys


def _nvidia_smi() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        rows = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            name, driver, memory = [part.strip() for part in line.split(",", maxsplit=2)]
            rows.append({"name": name, "driver": driver, "memory_mib": int(memory)})
        return {"available": True, "gpus": rows}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def collect() -> dict:
    result: dict = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "nvidia_smi": _nvidia_smi(),
        "pytorch": {},
        "torchvision": {},
        "tensorflow": {},
        "onnxruntime": {},
    }

    try:
        import torch

        result["pytorch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        result["pytorch"] = {"available": False, "error": str(exc)}

    try:
        import torchvision

        result["torchvision"] = {"version": torchvision.__version__}
    except Exception as exc:
        result["torchvision"] = {"available": False, "error": str(exc)}

    try:
        import onnxruntime as ort

        result["onnxruntime"] = {
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
            "cuda_provider_available": "CUDAExecutionProvider" in ort.get_available_providers(),
        }
    except Exception as exc:
        result["onnxruntime"] = {"available": False, "error": str(exc)}

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        result["tensorflow"] = {
            "version": tf.__version__,
            "gpu_count": len(gpus),
            "gpus": [gpu.name for gpu in gpus],
        }
    except Exception as exc:
        result["tensorflow"] = {"available": False, "error": str(exc)}

    return result

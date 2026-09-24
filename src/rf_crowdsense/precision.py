from __future__ import annotations

SUPPORTED_PRECISIONS = ("fp32", "fp16", "bf16")
AMP_DTYPES = ("fp16", "bf16")


def normalize_precision(value: str) -> str:
    precision = value.strip().lower()
    aliases = {
        "float32": "fp32",
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
    }
    precision = aliases.get(precision, precision)
    if precision not in SUPPORTED_PRECISIONS:
        expected = ", ".join(SUPPORTED_PRECISIONS)
        raise ValueError(f"Unsupported precision {value!r}; expected one of: {expected}")
    return precision


def normalize_precision_list(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        raw = [part for part in value.split(",") if part.strip()]
    else:
        raw = list(value)
    if not raw:
        raise ValueError("At least one precision must be requested")

    normalized: list[str] = []
    for item in raw:
        precision = normalize_precision(item)
        if precision not in normalized:
            normalized.append(precision)
    return normalized


def normalize_amp_dtype(value: str) -> str:
    precision = normalize_precision(value)
    if precision not in AMP_DTYPES:
        expected = ", ".join(AMP_DTYPES)
        raise ValueError(f"AMP dtype must be one of: {expected}")
    return precision


def torch_dtype(torch, precision: str):
    precision = normalize_precision(precision)
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def precision_support(torch, precision: str, device) -> tuple[bool, str | None]:
    precision = normalize_precision(precision)
    if precision == "fp32":
        return True, None
    if getattr(device, "type", str(device)) != "cuda":
        return False, f"{precision} benchmark requires CUDA in this milestone"
    if not torch.cuda.is_available():
        return False, "PyTorch cannot access a CUDA device"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        return False, "CUDA device does not report native BF16 support"
    return True, None


def grad_scaler_enabled(*, amp_enabled: bool, amp_dtype: str) -> bool:
    return bool(amp_enabled and normalize_amp_dtype(amp_dtype) == "fp16")

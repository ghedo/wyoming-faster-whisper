"""Translation of the --device option into each backend's own vocabulary.

There is one user-facing device string, but the backends disagree on how to name
hardware: CTranslate2 and torch expose ROCm devices through their "cuda" APIs,
sherpa-onnx has no ROCm provider, and onnxruntime uses named execution providers.
Resolving that in one place keeps the string-sniffing out of the handlers, and
keeps the rules testable without any backend installed.

Accepted values are "cpu", "cuda", "cuda:N", "rocm", and "rocm:N". Anything
else is passed through unchanged to the torch- and CTranslate2-based backends (so
"auto", "mps", or "xpu" still reach the library that understands them) and
treated as CPU for the onnxruntime-based ones, which need an explicit provider
list.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_LOGGER = logging.getLogger(__name__)

# Execution provider entries are either a name or (name, options).
OnnxProvider = Union[str, Tuple[str, Dict[str, Any]]]

_CPU_PROVIDER = "CPUExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_MIGRAPHX_PROVIDER = "MIGraphXExecutionProvider"


def is_rocm(device: str) -> bool:
    """Report whether a device string names a ROCm GPU."""
    return device.strip().lower().startswith("rocm")


def is_gpu(device: str) -> bool:
    """Report whether a device string names a supported GPU family."""
    normalized = device.strip().lower()
    return normalized.startswith("cuda") or normalized.startswith("rocm")


def device_index(device: str) -> Optional[int]:
    """Extract the GPU ordinal from "cuda:N" or "rocm:N"."""
    _, _, index = device.strip().lower().partition(":")
    if not index:
        return None

    try:
        return int(index)
    except ValueError:
        _LOGGER.warning("Ignoring unparsable device index in '%s'", device)
        return None


def torch_device(device: str) -> str:
    """Return the device string for a torch-based backend (transformers, FunASR).

    PyTorch deliberately reuses its "cuda" API for ROCm devices.
    """
    normalized = device.strip()
    if is_rocm(normalized):
        _, separator, index = normalized.lower().partition(":")
        return f"cuda{separator}{index}"

    return normalized


def ctranslate2_device(device: str) -> Tuple[str, Optional[int]]:
    """Return (device, device_index) for faster-whisper.

    CTranslate2 calls both CUDA and ROCm devices "cuda". It also takes the ordinal
    as a separate ``device_index`` argument rather than as part of the device
    string.
    """
    translated = torch_device(device) if is_rocm(device) else device.strip()
    index = device_index(device)
    if index is None:
        return translated, None

    base, _, _ = translated.partition(":")
    return base, index


def sherpa_provider(device: str) -> str:
    """Return the sherpa-onnx provider name.

    Requires a CUDA-enabled sherpa-onnx build. The CPU wheel published on PyPI
    accepts provider="cuda" and silently runs on the CPU anyway, so a GPU image
    must install the "+cuda" wheel from the k2-fsa index. sherpa-onnx does not
    currently expose a ROCm provider.
    """
    return "cuda" if device.strip().lower().startswith("cuda") else "cpu"


def onnx_providers(device: str) -> List[OnnxProvider]:
    """Return the onnxruntime execution providers to try, in order.

    The CPU provider is always kept as the last entry so a container without a
    usable driver degrades to CPU inference instead of failing to load a model.
    """
    if not is_gpu(device):
        return [_CPU_PROVIDER]

    gpu_provider = _MIGRAPHX_PROVIDER if is_rocm(device) else _CUDA_PROVIDER
    index = device_index(device)
    if index is None:
        return [gpu_provider, _CPU_PROVIDER]

    return [(gpu_provider, {"device_id": index}), _CPU_PROVIDER]


def warn_if_no_onnx_gpu(device: str, providers: Sequence[str]) -> bool:
    """Warn when a requested GPU provider is not in use by onnxruntime.

    Takes the provider list as an argument rather than importing onnxruntime, so
    this module stays importable (and testable) without it. Returns whether the
    GPU is in use.

    Prefer passing a live session's ``get_providers()`` over
    ``ort.get_available_providers()``: the latter can list a GPU provider even
    when its shared library cannot be loaded, in which case sessions silently
    fall back to the CPU. Either way the failure is quiet, so it is worth a
    warning rather than leaving it to be noticed as "the GPU image is no faster".
    """
    if not is_gpu(device):
        return False

    gpu_provider = _MIGRAPHX_PROVIDER if is_rocm(device) else _CUDA_PROVIDER
    if gpu_provider in providers:
        return True

    runtime_package = "onnxruntime-migraphx" if is_rocm(device) else "onnxruntime-gpu"
    runtime_requirements = "ROCm/MIGraphX" if is_rocm(device) else "CUDA/cuDNN"
    _LOGGER.warning(
        "Device '%s' was requested but onnxruntime is not using %s "
        "(in effect: %s), so inference will run on the CPU. Either "
        "%s is not installed, or its %s requirements are "
        "not met here - check the onnxruntime errors logged above.",
        device,
        gpu_provider,
        ", ".join(providers) or "none",
        runtime_package,
        runtime_requirements,
    )
    return False


def resolve_compute_type(compute_type: str, device: str) -> str:
    """Pick the CTranslate2 compute type, filling in a GPU-appropriate default.

    CTranslate2's own "default" means "whatever the model was converted to",
    which for the int8 models this project downloads by default would mean int8
    on hardware that is much faster at float16.
    """
    if (compute_type == "default") and is_gpu(device):
        return "float16"

    return compute_type

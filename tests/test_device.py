"""Tests for --device translation.

Dependency-free: the point of wyoming_faster_whisper.device is that the rules can
be checked without torch, onnxruntime, or sherpa-onnx installed.
"""

import sys
from typing import List, Tuple

import pytest

from wyoming_faster_whisper.device import (
    ctranslate2_device,
    device_index,
    disable_onnx_spinning,
    is_gpu,
    is_rocm,
    onnx_providers,
    register_migraphx_ep,
    resolve_compute_type,
    sherpa_provider,
    torch_device,
    warn_if_no_onnx_gpu,
)

# --- is_gpu ----------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:3", "CUDA", " cuda "])
def test_cuda_is_gpu(device) -> None:
    assert is_gpu(device)


@pytest.mark.parametrize("device", ["rocm", "rocm:0", "rocm:3", "ROCM", " rocm "])
def test_rocm_is_gpu(device) -> None:
    assert is_gpu(device)
    assert is_rocm(device)


@pytest.mark.parametrize("device", ["cpu", "CPU", "auto", "mps", "xpu", ""])
def test_everything_else_is_not_gpu(device) -> None:
    # "auto" deliberately reads as non-GPU so the onnxruntime backends get a
    # provider list that works rather than one that might not.
    assert not is_gpu(device)
    assert not is_rocm(device)


# --- device index ----------------------------------------------------------


def test_index_is_none_when_unspecified() -> None:
    assert device_index("cuda") is None
    assert device_index("cpu") is None


def test_index_is_parsed() -> None:
    assert device_index("cuda:0") == 0
    assert device_index("cuda:2") == 2
    assert device_index("rocm:1") == 1


def test_unparsable_index_is_ignored() -> None:
    assert device_index("cuda:first") is None


# --- per-backend translation ----------------------------------------------


def test_torch_device_translation() -> None:
    assert torch_device("cuda") == "cuda"
    assert torch_device("cuda:1") == "cuda:1"
    # PyTorch exposes ROCm devices through the CUDA API.
    assert torch_device("rocm") == "cuda"
    assert torch_device("rocm:1") == "cuda:1"
    assert torch_device(" cpu ") == "cpu"


def test_ctranslate2_splits_the_ordinal_out() -> None:
    # CTranslate2 takes device_index as a separate argument.
    assert ctranslate2_device("cuda:1") == ("cuda", 1)
    assert ctranslate2_device("rocm:1") == ("cuda", 1)


def test_ctranslate2_leaves_bare_devices_alone() -> None:
    assert ctranslate2_device("cuda") == ("cuda", None)
    assert ctranslate2_device("rocm") == ("cuda", None)
    assert ctranslate2_device("cpu") == ("cpu", None)
    # "auto" is CTranslate2's own value and must survive untouched.
    assert ctranslate2_device("auto") == ("auto", None)


def test_sherpa_provider() -> None:
    assert sherpa_provider("cuda") == "cuda"
    assert sherpa_provider("cuda:1") == "cuda"
    assert sherpa_provider("rocm") == "cpu"
    assert sherpa_provider("cpu") == "cpu"


def test_onnx_providers_on_cpu() -> None:
    assert onnx_providers("cpu") == ["CPUExecutionProvider"]


def test_onnx_providers_keep_cpu_fallback() -> None:
    # A container without a usable driver must degrade to CPU rather than fail
    # to load the model at all.
    providers = onnx_providers("cuda")
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_onnx_providers_carry_the_device_id() -> None:
    providers = onnx_providers("cuda:2")
    assert providers == [
        ("CUDAExecutionProvider", {"device_id": 2}),
        "CPUExecutionProvider",
    ]


def test_onnx_providers_use_migraphx_for_rocm() -> None:
    assert onnx_providers("rocm") == [
        "MIGraphXExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert onnx_providers("rocm:2") == [
        ("MIGraphXExecutionProvider", {"device_id": 2}),
        "CPUExecutionProvider",
    ]


# --- compute type ---------------------------------------------------------


def test_compute_type_default_becomes_float16_on_gpu() -> None:
    assert resolve_compute_type("default", "cuda") == "float16"
    assert resolve_compute_type("default", "cuda:1") == "float16"
    assert resolve_compute_type("default", "rocm") == "float16"


def test_compute_type_default_is_left_alone_on_cpu() -> None:
    assert resolve_compute_type("default", "cpu") == "default"


def test_explicit_compute_type_is_never_overridden() -> None:
    # Someone asking for int8_float16 on a GPU means it.
    assert resolve_compute_type("int8_float16", "cuda") == "int8_float16"
    assert resolve_compute_type("int8", "cuda") == "int8"


# --- silent-CPU-fallback warning ------------------------------------------


def test_no_warning_when_cpu_was_asked_for(caplog) -> None:
    assert not warn_if_no_onnx_gpu("cpu", ["CPUExecutionProvider"])
    assert not caplog.records


def test_gpu_available_is_not_warned_about(caplog) -> None:
    assert warn_if_no_onnx_gpu(
        "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert not caplog.records


def test_missing_cuda_provider_warns(caplog) -> None:
    # The whole point: onnxruntime loads the model on the CPU provider without
    # complaint, so the only symptom would be no speedup. This fires both when
    # the CPU package is installed and when a session falls back because the
    # CUDA provider library could not be loaded.
    assert not warn_if_no_onnx_gpu("cuda", ["CPUExecutionProvider"])
    assert len(caplog.records) == 1
    assert "onnxruntime-gpu" in caplog.records[0].getMessage()


# --- disable_onnx_spinning --------------------------------------------------


class _RecordingSessionOptions:
    """Stand-in for onnxruntime.SessionOptions that records config entries."""

    def __init__(self) -> None:
        self.entries: dict = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.entries[key] = value


def test_disable_onnx_spinning_turns_off_both_pools() -> None:
    options = _RecordingSessionOptions()

    assert disable_onnx_spinning(options) is options
    assert options.entries == {
        "session.intra_op.allow_spinning": "0",
        "session.inter_op.allow_spinning": "0",
    }


def test_migraphx_provider_is_recognized(caplog) -> None:
    assert warn_if_no_onnx_gpu(
        "rocm", ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
    )
    assert not caplog.records


def test_missing_migraphx_provider_warns(caplog) -> None:
    assert not warn_if_no_onnx_gpu("rocm", ["CPUExecutionProvider"])
    assert len(caplog.records) == 1
    assert "onnxruntime-ep-migraphx" in caplog.records[0].getMessage()


# --- MIGraphX EP plugin registration ----------------------------------------


class _FakeEp:
    """Enough of onnxruntime_ep_migraphx to register."""

    @staticmethod
    def get_library_path():
        return "/fake/libmigraphx-ep.so"


def _fake_ort(providers, registrations):
    class _Ort:
        @staticmethod
        def get_available_providers():
            return providers

        @staticmethod
        def register_execution_provider_library(name, path):
            registrations.append((name, path))
            providers.append(name)

    return _Ort()


def test_register_migraphx_ep_without_the_plugin_is_a_noop(monkeypatch) -> None:
    # Neither onnxruntime nor the plugin is importable in the test environment;
    # registration must be a silent no-op so device.py stays dependency-free.
    for name in ("onnxruntime", "onnxruntime_ep_migraphx"):
        monkeypatch.setitem(sys.modules, name, None)
    assert register_migraphx_ep() is False


def test_register_migraphx_ep_registers_the_plugin(monkeypatch) -> None:
    providers: List[str] = ["CPUExecutionProvider"]
    registrations: List[Tuple[str, str]] = []
    ort = _fake_ort(providers, registrations)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_migraphx", _FakeEp)

    assert register_migraphx_ep() is True
    assert registrations == [("MIGraphXExecutionProvider", "/fake/libmigraphx-ep.so")]

    # A second call sees the provider is already available and does nothing.
    assert register_migraphx_ep() is True
    assert len(registrations) == 1

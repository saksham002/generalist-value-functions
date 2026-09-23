import os
import shutil

import pytest


def set_jax_cpu_backend_if_no_gpu() -> None:
    # A CPU backend keeps the tests deterministic and cheap on machines without an NVIDIA
    # driver; hosts that want another backend export JAX_PLATFORMS themselves.
    if "JAX_PLATFORMS" not in os.environ and shutil.which("nvidia-smi") is None:
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()

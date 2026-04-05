import os
import sys
import pytest


# GPU tests must be explicitly opted in via environment variable.
# This follows the pattern from source/pip/tests/test_gpu_simulator.py.
if not os.environ.get("QDK_GPU_TESTS"):
    pytest.skip(
        "Skipping GPU simulator Python tests (QDK_GPU_TESTS not set)",
        allow_module_level=True,
    )


GPU_AVAILABLE = False
SKIP_REASON = "GPU is not available"

try:
    from qdk_gpu_sim import GpuQuantumSim

    # Probe GPU availability by attempting to create a simulator.
    _probe = GpuQuantumSim(seed=0)
    del _probe
    GPU_AVAILABLE = True
    print("*** GPU simulator available", file=sys.stderr)
except RuntimeError as e:
    SKIP_REASON = str(e)
except ImportError as e:
    SKIP_REASON = f"qdk_gpu_sim not installed: {e}"


@pytest.fixture(autouse=True)
def _skip_without_gpu():
    """Skip every test in this directory if no GPU is available."""
    if not GPU_AVAILABLE:
        pytest.skip(SKIP_REASON)


@pytest.fixture
def sim():
    """Provide a deterministic GpuQuantumSim instance for each test."""
    return GpuQuantumSim(seed=42)

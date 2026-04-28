"""Round trip for count_ground_states_gpu."""

import numpy as np
import pytest

try:
    import tropical_gemm
    _fn = getattr(
        tropical_gemm, "count_ground_states_gpu",
        getattr(tropical_gemm, "count_ground_states_gpu_py", None),
    )
    HAVE_EXT = _fn is not None
except ImportError:
    HAVE_EXT = False
    _fn = None

pytestmark = pytest.mark.skipif(not HAVE_EXT, reason="tropical_gemm[cuda] extension not built")


def test_trivial_gpu_1x1():
    a = np.array([[3.0]], dtype=np.float32)
    b = np.array([[4.0]], dtype=np.float32)
    values, counts = _fn(a, b, "max")
    assert values[0, 0] == 7.0
    assert int(counts[0, 0]) == 1


def test_ties_merge_gpu_max():
    a = np.array([[2.0, 3.0]], dtype=np.float32)
    b = np.array([[3.0], [2.0]], dtype=np.float32)
    values, counts = _fn(a, b, "max")
    assert values[0, 0] == 5.0
    assert int(counts[0, 0]) == 2


def test_gpu_matches_cpu():
    a = np.random.RandomState(42).randint(0, 5, size=(8, 12)).astype(np.float32)
    b = np.random.RandomState(43).randint(0, 5, size=(12, 6)).astype(np.float32)
    gpu_v, gpu_c = _fn(a, b, "max")
    cpu_v, cpu_c = tropical_gemm.count_ground_states(a, b, "max")
    np.testing.assert_array_equal(gpu_v, cpu_v)
    assert gpu_c.shape == cpu_c.shape
    for (i, j), x in np.ndenumerate(gpu_c):
        assert int(x) == int(cpu_c[i, j])

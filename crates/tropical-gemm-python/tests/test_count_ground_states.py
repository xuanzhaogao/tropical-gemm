"""End-to-end round trip for count_ground_states."""

import numpy as np
import pytest

try:
    import tropical_gemm
    _fn = getattr(
        tropical_gemm, "count_ground_states",
        getattr(tropical_gemm, "count_ground_states_py", None),
    )
    HAVE_EXT = _fn is not None
except ImportError:
    HAVE_EXT = False
    _fn = None

pytestmark = pytest.mark.skipif(not HAVE_EXT, reason="tropical_gemm extension not built")


def test_trivial_1x1():
    a = np.array([[3.0]], dtype=np.float32)
    b = np.array([[4.0]], dtype=np.float32)
    values, counts = _fn(a, b, "max")
    assert values.shape == (1, 1)
    assert counts.shape == (1, 1)
    assert values[0, 0] == 7.0
    assert int(counts[0, 0]) == 1


def test_ties_merge_max():
    a = np.array([[2.0, 3.0]], dtype=np.float32)
    b = np.array([[3.0], [2.0]], dtype=np.float32)
    values, counts = _fn(a, b, "max")
    assert values[0, 0] == 5.0
    assert int(counts[0, 0]) == 2


def test_ties_merge_min():
    a = np.array([[2.0, 3.0]], dtype=np.float32)
    b = np.array([[3.0], [2.0]], dtype=np.float32)
    values, counts = _fn(a, b, "min")
    assert values[0, 0] == 5.0
    assert int(counts[0, 0]) == 2


def test_returns_python_int_not_numpy():
    a = np.zeros((1, 5), dtype=np.float32)
    b = np.zeros((5, 1), dtype=np.float32)
    _, counts = _fn(a, b, "max")
    assert counts.dtype == object
    assert isinstance(counts[0, 0], int)
    assert counts[0, 0] == 5


def test_bad_direction_raises():
    a = np.array([[1.0]], dtype=np.float32)
    b = np.array([[1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        _fn(a, b, "sideways")

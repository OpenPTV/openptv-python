"""Tests for the vec_utils module."""

import numpy as np

from openptv_python.vec_utils import (
    unit_vector,
    vec_add,
    vec_cmp,
    vec_copy,
    vec_cross,
    vec_diff_norm,
    vec_dot,
    vec_norm,
    vec_scalar_mul,
    vec_set,
    vec_subt,
)


def test_vec_set():
    """Test vec_set()."""
    dest = vec_set(1.2, 3.4, 5.6)
    assert np.allclose(dest, np.array([1.2, 3.4, 5.6]))


def test_vec_copy():
    """Test vec_copy()."""
    src = np.array([1.2, 3.4, 5.6])
    dest = vec_copy(src)
    assert np.all(dest == src)


def test_vec_subt():
    """Test vec_subt()."""
    from_ = np.array([1.0, 2.0, 3.0])
    sub = np.array([4.0, 5.0, 6.0])
    output = vec_subt(from_, sub)
    assert np.allclose(output, np.array([-3.0, -3.0, -3.0]))


def test_vec_add():
    """Test vec_add()."""
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([4.0, 5.0, 6.0])
    output = vec_add(vec1, vec2)
    assert np.allclose(output, np.array([5.0, 7.0, 9.0]))


def test_vec_scalar_mul():
    """Test vec_scalar_mul()."""
    vec = np.array([1.0, 2.0, 3.0])
    scalar = 2.0
    output = vec_scalar_mul(vec, scalar)
    assert np.all(output == np.array([2.0, 4.0, 6.0]))


def test_vec_diff_norm():
    """Test vec_diff_norm()."""
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([4.0, 5.0, 6.0])
    assert vec_diff_norm(vec1, vec2) == 5.196152422706632


def test_vec_norm():
    """Test vec_norm()."""
    vec = np.array([1.0, 2.0, 3.0])
    assert np.allclose(vec_norm(vec), 3.7416, rtol=1e-4)


def test_vec_dot():
    """Test vec_dot()."""
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([4.0, 5.0, 6.0])
    assert np.isclose(vec_dot(vec1, vec2), 32.0, rtol=1e-4)


def test_vec_cross():
    """Test vec_cross()."""
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([4.0, 5.0, 6.0])
    output = vec_cross(vec1, vec2)
    assert np.all(output == [-3.0, 6.0, -3.0])


def test_vec_cmp():
    """Test vec_cmp()."""
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([1.0, 2.0, 3.0])
    assert vec_cmp(vec1, vec2, 1e-4)
    vec3 = np.array([4.0, 5.0, 6.0])
    assert not vec_cmp(vec1, vec3, 1e-4)


def test_unit_vector():
    """Tests unit vector as a list of floats with norm of 1.0."""
    vec = np.array([1.0, 2.0, 3.0])
    out = unit_vector(vec)
    assert np.isclose(vec_norm(out), 1.0, rtol=1e-4)

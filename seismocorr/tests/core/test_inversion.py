# tests/test_inversion.py

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from seismocorr.core.inversion import get_inversion


def _obj(A, d, x, x0, alpha, beta):
    """和 inversion.py 里的目标函数保持一致。"""
    r = A @ x - d
    loss = float(r @ r)
    if alpha > 0:
        loss += float(alpha * np.sum((x - x0) ** 2))
    if beta > 0:
        loss += float(beta * np.sum(np.abs(x - x0)))
    return loss


def _ridge_closed_form(A, d, x0, alpha):
    """min ||Ax-d||^2 + alpha||x-x0||^2 的解析解（dense）"""
    A = np.asarray(A, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    n = x0.size
    lhs = A.T @ A + alpha * np.eye(n, dtype=np.float64)
    rhs = A.T @ d + alpha * x0
    return np.linalg.solve(lhs, rhs)


def test_get_inversion_supported_and_callable():
    for reg in ["none", "l2", "l1", "l1_l2"]:
        inv = get_inversion(reg)
        assert callable(inv)


def test_none_dense_matches_lstsq():
    rng = np.random.default_rng(0)

    m, n = 30, 8
    A = rng.normal(size=(m, n))
    x_true = rng.normal(size=n)
    d = A @ x_true + 0.01 * rng.normal(size=m)

    # x0 只是初值，不应该影响最小二乘最优解
    x0 = rng.normal(size=n)

    inv = get_inversion("none")
    res = inv(A=A, d=d, x0=x0, alpha=0.0, beta=0.0)

    assert isinstance(res, dict)
    assert "x" in res and "success" in res and "fun" in res and "niter" in res and "message" in res

    x_hat = np.asarray(res["x"], dtype=np.float64).reshape(-1)

    x_lstsq, *_ = np.linalg.lstsq(A, d, rcond=None)

    # L-BFGS-B 在这种小问题上应能非常接近
    np.testing.assert_allclose(x_hat, x_lstsq, rtol=1e-4, atol=1e-5)

    # fun 与目标函数一致
    fun_check = _obj(A, d, x_hat, x0, alpha=0.0, beta=0.0)
    assert np.isfinite(res["fun"])
    np.testing.assert_allclose(res["fun"], fun_check, rtol=1e-10, atol=1e-10)


def test_l2_dense_matches_closed_form():
    rng = np.random.default_rng(1)

    m, n = 40, 10
    A = rng.normal(size=(m, n))
    x_true = rng.normal(size=n)
    d = A @ x_true + 0.02 * rng.normal(size=m)

    x0 = rng.normal(size=n)
    alpha = 0.3

    inv = get_inversion("l2")
    res = inv(A=A, d=d, x0=x0, alpha=alpha, beta=0.0)
    x_hat = np.asarray(res["x"], dtype=np.float64).reshape(-1)

    x_cf = _ridge_closed_form(A, d, x0, alpha=alpha)

    np.testing.assert_allclose(x_hat, x_cf, rtol=1e-4, atol=1e-5)

    fun_check = _obj(A, d, x_hat, x0, alpha=alpha, beta=0.0)
    np.testing.assert_allclose(res["fun"], fun_check, rtol=1e-10, atol=1e-10)


def test_l2_sparse_runs():
    rng = np.random.default_rng(2)

    m, n = 60, 12
    A_dense = rng.normal(size=(m, n))

    # 构造一个稀疏版本：保留部分元素
    mask = rng.random(size=A_dense.shape) < 0.25
    A_sparse = csr_matrix(A_dense * mask)

    x_true = rng.normal(size=n)
    d = A_sparse @ x_true + 0.01 * rng.normal(size=m)

    x0 = np.zeros(n, dtype=np.float64)
    alpha = 0.2

    inv = get_inversion("l2")
    res = inv(A=A_sparse, d=d, x0=x0, alpha=alpha, beta=0.0)

    x_hat = np.asarray(res["x"], dtype=np.float64).reshape(-1)
    assert x_hat.shape == (n,)
    assert np.isfinite(res["fun"])


def test_l1_and_l1_l2_run_and_decrease_objective():
    rng = np.random.default_rng(3)

    m, n = 50, 10
    A = rng.normal(size=(m, n))
    x_true = rng.normal(size=n)
    d = A @ x_true + 0.01 * rng.normal(size=m)

    x0 = np.zeros(n, dtype=np.float64)

    # L1
    inv1 = get_inversion("l1")
    beta = 0.5
    res1 = inv1(A=A, d=d, x0=x0, alpha=0.0, beta=beta)
    x1 = np.asarray(res1["x"], dtype=np.float64).reshape(-1)

    f0 = _obj(A, d, x0, x0, alpha=0.0, beta=beta)  # 用 x0 作为初始点的目标函数
    f1 = _obj(A, d, x1, x0, alpha=0.0, beta=beta)
    assert f1 <= f0 + 1e-8

    # L1 + L2
    inv12 = get_inversion("l1_l2")
    alpha = 0.2
    beta = 0.4
    res12 = inv12(A=A, d=d, x0=x0, alpha=alpha, beta=beta)
    x12 = np.asarray(res12["x"], dtype=np.float64).reshape(-1)

    f0 = _obj(A, d, x0, x0, alpha=alpha, beta=beta)
    f12 = _obj(A, d, x12, x0, alpha=alpha, beta=beta)
    assert f12 <= f0 + 1e-8


def test_validate_shapes_errors():
    inv = get_inversion("l2")

    A = np.ones((5, 3), dtype=np.float64)
    d = np.ones(4, dtype=np.float64)   # 行数不匹配
    x0 = np.ones(3, dtype=np.float64)

    with pytest.raises(ValueError):
        inv(A=A, d=d, x0=x0, alpha=0.1, beta=0.0)

    d_ok = np.ones(5, dtype=np.float64)
    x0_bad = np.ones(4, dtype=np.float64)  # 列数不匹配
    with pytest.raises(ValueError):
        inv(A=A, d=d_ok, x0=x0_bad, alpha=0.1, beta=0.0)

    with pytest.raises(ValueError):
        inv(A=A, d=np.array([]), x0=x0, alpha=0.1, beta=0.0)

    with pytest.raises(ValueError):
        inv(A=A, d=d_ok, x0=np.array([]), alpha=0.1, beta=0.0)

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from seismocorr.plugins.processing.svd_wf import SVDLowRankDenoiser, WienerFilterDenoiser


def make_synthetic_ncf_windows(
    n_windows: int = 300,
    n_lag: int = 2048,
    *,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成示例数据（窗口 × lag）用于 demo。

    Args:
        n_windows: 窗口数量。
        n_lag: 每条窗口的采样点数（lag 轴长度）。
        seed: 随机种子，用于可复现。

    Returns:
        (X_noisy, X_clean)
            - X_noisy: shape (n_windows, n_lag)，包含干扰/噪声的输入矩阵
            - X_clean: shape (n_windows, n_lag)，仅包含相干模板（用于对照）
    """
    rng = np.random.default_rng(seed)
    lag = np.linspace(-1.0, 1.0, int(n_lag))

    def packet(mu: float, f: float, w: float) -> np.ndarray:
        return np.exp(-((lag - mu) / w) ** 2) * np.cos(2.0 * np.pi * f * (lag - mu))

    template = 1.2 * packet(0.25, f=10.0, w=0.08) + 1.0 * packet(-0.25, f=10.0, w=0.08)
    template += 0.6 * packet(0.45, f=6.0, w=0.12) + 0.6 * packet(-0.45, f=6.0, w=0.12)

    amps = 1.0 + 0.15 * rng.standard_normal(int(n_windows))
    X_clean = amps[:, None] * template[None, :]

    spike = np.exp(-(lag / 0.03) ** 2)
    inter_amp = np.zeros(int(n_windows), dtype=float)
    idx = rng.choice(int(n_windows), size=int(n_windows) // 3, replace=False)
    inter_amp[idx] = 2.5 + 0.5 * rng.standard_normal(len(idx))
    interference = inter_amp[:, None] * spike[None, :]

    noise = 0.8 * rng.standard_normal((int(n_windows), int(n_lag)))

    X_noisy = X_clean + interference + noise
    return X_noisy, X_clean


def main() -> None:
    """运行 SVD + Wiener 去噪 demo（与原 SVDWF 流程等价）。

    输入:
        X: shape (n_windows, n_samples)

    处理:
        1) SVD 低秩：在中心化域得到 X_lr0
        2) 残差估计：residual = X0 - X_lr0
        3) Wiener：用 X_lr0 与 residual 估计增益并滤波得到 X_filt0
        4) 回到原始基线：X_out = X_filt0 + mean_

    输出:
        X_out: shape (n_windows, n_samples)，去噪后的矩阵
    """
    X, _ = make_synthetic_ncf_windows()

    svd_den = SVDLowRankDenoiser(
        rank=1,  # 若需自动选 rank：设 rank=None，并配置 method/energy/thresh
        method="energy",
        energy=0.90,
        center=True,
        random_sign_fix=True,
    ).fit(X)

    X0 = svd_den.center_data(X)
    X_lr0 = svd_den.transform(X, add_mean=False)

    residual = X0 - X_lr0
    wien_den = WienerFilterDenoiser(
        psd_smooth=21,
        wiener_beta=1.0,
        gain_floor=0.03,
    ).fit(X_lr0, residual)

    X_filt0 = wien_den.transform(X_lr0)
    X_out = X_filt0 + svd_den.mean_

    # -------------------------
    # 可视化：奇异值谱
    # -------------------------
    _, ax1 = plt.subplots()
    svd_den.plot_spectrum(ax=ax1, log=True)

    # -------------------------
    # 可视化：Wiener 增益
    # -------------------------
    _, ax2 = plt.subplots()
    wien_den.plot_wiener_gain(ax=ax2)

    # -------------------------
    # 可视化：输入/输出/去除部分
    # -------------------------
    w_slice = slice(0, 120)
    lag_slice = slice(600, 1450)

    fig3, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(X[w_slice, lag_slice], aspect="auto")
    axes[0].set_title("Input X")
    axes[0].set_xlabel("Lag sample index")
    axes[0].set_ylabel("Window index")
    fig3.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(X_out[w_slice, lag_slice], aspect="auto")
    axes[1].set_title(f"Output (rank={svd_den.rank_})")
    axes[1].set_xlabel("Lag sample index")
    axes[1].set_ylabel("Window index")
    fig3.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow((X - X_out)[w_slice, lag_slice], aspect="auto")
    axes[2].set_title("Removed (X - Output)")
    axes[2].set_xlabel("Lag sample index")
    axes[2].set_ylabel("Window index")
    fig3.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.show()

    # -------------------------
    # 叠加曲线对比（stack）
    # -------------------------
    stack_in = X.mean(axis=0)
    stack_out = X_out.mean(axis=0)

    _, ax4 = plt.subplots()
    ax4.plot(stack_in, label="Stacked input")
    ax4.plot(stack_out, label="Stacked output")
    ax4.set_title("Stack comparison")
    ax4.grid(True)
    ax4.legend()
    plt.show()


if __name__ == "__main__":
    main()

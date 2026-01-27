"""
svdwf.py

功能：
- SVDLowRankDenoiser：对窗口堆叠矩阵做截断 SVD 低秩重构（子空间投影）
- WienerFilterDenoiser：基于“信号谱/噪声谱”估计 Wiener 增益，对信号做频域滤波

数据约定：
- 输入矩阵 X 的形状固定为 (n_windows, n_samples)
  * n_windows：窗口/片段数量（例如每个时间窗一条 NCF）
  * n_samples：每条片段的采样点数（例如 lag 轴采样点）
- FFT / rFFT 一律沿最后一维（n_samples）进行

依赖：
- numpy, matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =========================
# rank 选择辅助
# =========================

def _safe_rank_from_energy(s: np.ndarray, energy: float) -> int:
    """
    根据奇异值能量累计占比选择 rank。

    输入
    ----
    s : np.ndarray, shape (r,)
        奇异值（降序），r = min(n_windows, n_samples)
    energy : float
        目标能量占比 (0, 1]，基于 σ^2 计算累计能量

    输出
    ----
    k : int
        最小的 k，使得 sum_{i<=k} σ_i^2 / sum σ_i^2 >= energy，且 k >= 1
    """
    if not (0 < energy <= 1.0):
        raise ValueError("energy 必须在 (0, 1]。")
    power = s**2
    cum = np.cumsum(power) / np.sum(power)
    k = int(np.searchsorted(cum, energy) + 1)
    return max(k, 1)


def _safe_rank_from_thresh(s: np.ndarray, thresh: float) -> int:
    """
    根据相对阈值选择 rank。

    输入
    ----
    s : np.ndarray, shape (r,)
        奇异值（降序）
    thresh : float
        相对阈值 (0, 1]，保留满足 σ_i >= thresh * σ_0 的分量

    输出
    ----
    k : int
        保留分量数，且 k >= 1
    """
    if not (0 < thresh <= 1.0):
        raise ValueError("thresh 必须在 (0, 1]。")
    k = int(np.sum(s >= (thresh * s[0])))
    return max(k, 1)


# =========================
# PSD / 平滑辅助
# =========================

def _moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    """
    一维滑动平均。

    输入
    ----
    x : np.ndarray, shape (n,)
        待平滑序列（PSD 或增益曲线）
    win : int
        窗口长度。win<=1 表示不平滑

    输出
    ----
    y : np.ndarray, shape (n,)
        平滑后序列（长度不变）
    """
    if win <= 1:
        return x
    win = int(win)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(xpad, kernel, mode="valid")


def _rfft_psd(X: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    计算跨窗口平均功率谱（PSD）。

    输入
    ----
    X : np.ndarray, shape (n_windows, n_samples)
        时域矩阵
    axis : int
        FFT 轴。默认 -1（沿 n_samples）
    eps : float
        PSD 下限，避免除零

    输出
    ----
    psd : np.ndarray, shape (n_freq,)
        平均 PSD。n_freq = n_samples//2 + 1（rFFT 频点数）
    """
    F = np.fft.rfft(X, axis=axis)
    P = np.mean(np.abs(F) ** 2, axis=0)
    return np.maximum(P, eps)


# =========================
# 模块 1：SVD 低秩重构
# =========================

@dataclass
class SVDLowRankDenoiser:
    """
    截断 SVD 低秩重构（子空间投影）。

    典型用途
    --------
    - 输入为窗口堆叠矩阵 X (n_windows, n_samples)
    - 在 fit() 中对中心化后的训练矩阵做 SVD，得到右奇异向量子空间 V_k
    - 在 transform() 中对任意同列数矩阵 X，用 V_k 做投影重构：
        X0 = X - mean_
        X_lr0 = (X0 @ V_k^T) @ V_k
      该形式用于“对新数据复用同一子空间”更稳健（不依赖训练时的 U、σ）。

    成员变量（fit 后可用）
    ----------------------
    mean_ : (1, n_samples) 列均值
    s_    : (r,) 奇异值
    Vt_   : (r, n_samples) 右奇异向量转置
    rank_ : int 实际保留阶数
    """

    rank: Optional[int] = None
    method: Literal["fixed", "energy", "thresh"] = "energy"
    energy: float = 0.95
    thresh: float = 0.05

    center: bool = True
    random_sign_fix: bool = True

    mean_: Optional[np.ndarray] = None
    U_: Optional[np.ndarray] = None
    s_: Optional[np.ndarray] = None
    Vt_: Optional[np.ndarray] = None
    rank_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "SVDLowRankDenoiser":
        """
        拟合低秩子空间。

        输入
        ----
        X : np.ndarray, shape (n_windows, n_samples)
            训练矩阵

        输出
        ----
        self : SVDLowRankDenoiser
            拟合后的对象（包含 mean_ / Vt_ / s_ / rank_）
        """
        X = self._validate_2d(X)
        X0 = X.copy()

        # 中心化：按列减均值。输出 mean_ shape (1, n_samples)
        if self.center:
            self.mean_ = np.mean(X0, axis=0, keepdims=True)
            X0 = X0 - self.mean_
        else:
            self.mean_ = np.zeros((1, X0.shape[1]), dtype=X0.dtype)

        # SVD：X0 = U Σ V^T
        U, s, Vt = np.linalg.svd(X0, full_matrices=False)

        # SVD 符号不唯一：统一符号便于稳定对比/可视化
        if self.random_sign_fix:
            signs = np.sign(U[np.argmax(np.abs(U), axis=0), range(U.shape[1])])
            signs[signs == 0] = 1.0
            U = U * signs
            Vt = Vt * signs[:, None]

        self.U_, self.s_, self.Vt_ = U, s, Vt

        # rank 选择
        if self.rank is not None:
            if self.rank < 1 or self.rank > len(s):
                raise ValueError(f"rank 必须在 [1, {len(s)}]。")
            self.rank_ = int(self.rank)
        else:
            if self.method == "energy":
                self.rank_ = _safe_rank_from_energy(s, self.energy)
            elif self.method == "thresh":
                self.rank_ = _safe_rank_from_thresh(s, self.thresh)
            elif self.method == "fixed":
                self.rank_ = 1
            else:
                raise ValueError(f"未知 method: {self.method}")

        return self

    def center_data(self, X: np.ndarray) -> np.ndarray:
        """
        对输入做与 fit 同样的中心化。

        输入
        ----
        X : np.ndarray, shape (n_windows, n_samples)
            原始矩阵

        输出
        ----
        X0 : np.ndarray, shape (n_windows, n_samples)
            中心化矩阵 X - mean_
        """
        self._check_is_fitted()
        X = self._validate_2d(X)
        if X.shape[1] != self.mean_.shape[1]:
            raise ValueError("X 列数与 fit() 时不一致（n_samples 不同）。")
        return X - self.mean_

    def transform(self, X: np.ndarray, *, add_mean: bool = True) -> np.ndarray:
        """
        使用拟合的子空间做低秩重构。

        输入
        ----
        X : np.ndarray, shape (n_windows, n_samples)
            待处理矩阵（要求 n_samples 与 fit 一致）
        add_mean : bool
            True  -> 输出加回 mean_（回到原始基线）
            False -> 输出为中心化域结果（常用于后续噪声估计）

        输出
        ----
        X_lr : np.ndarray, shape (n_windows, n_samples)
            低秩重构结果
        """
        self._check_is_fitted()
        X = self._validate_2d(X)
        if X.shape[1] != self.mean_.shape[1]:
            raise ValueError("transform() 的 X 列数与 fit() 时不一致（n_samples 不同）。")

        X0 = X - self.mean_
        X_lr0 = self._low_rank_reconstruct_centered(X0)
        return (X_lr0 + self.mean_) if add_mean else X_lr0

    def fit_transform(self, X: np.ndarray, *, add_mean: bool = True) -> np.ndarray:
        """等价于 fit(X) 后 transform(X)。"""
        return self.fit(X).transform(X, add_mean=add_mean)

    def _low_rank_reconstruct_centered(self, X0: np.ndarray) -> np.ndarray:
        """
        对中心化矩阵做子空间投影重构。

        输入
        ----
        X0 : np.ndarray, shape (n_windows, n_samples)
            中心化矩阵

        输出
        ----
        X_lr0 : np.ndarray, shape (n_windows, n_samples)
            中心化域的低秩重构
        """
        k = int(self.rank_)
        Vtk = self.Vt_[:k, :]  # (k, n_samples)
        return (X0 @ Vtk.T) @ Vtk

    def singular_values(self) -> np.ndarray:
        """输出奇异值数组 s_（copy）。"""
        self._check_is_fitted()
        return self.s_.copy()

    def explained_energy(self) -> np.ndarray:
        """输出累计能量占比曲线（基于 σ^2）。"""
        self._check_is_fitted()
        power = self.s_**2
        return np.cumsum(power) / np.sum(power)

    def plot_spectrum(self, ax: Optional[plt.Axes] = None, *, log: bool = True) -> plt.Axes:
        """
        绘制奇异值谱（用于检查 rank 选择）。

        输入
        ----
        ax : matplotlib.axes.Axes or None
            None 则内部创建
        log : bool
            True 使用对数纵轴

        输出
        ----
        ax : matplotlib.axes.Axes
        """
        self._check_is_fitted()
        if ax is None:
            _, ax = plt.subplots()
        y = self.s_
        x = np.arange(1, len(y) + 1)
        if log:
            ax.semilogy(x, y, marker="o")
        else:
            ax.plot(x, y, marker="o")
        ax.set_xlabel("Component index (starting from 1)")
        ax.set_ylabel("Singular value" + (" (log scale)" if log else ""))
        ax.set_title(f"Singular-value spectrum (chosen rank = {self.rank_})")
        ax.grid(True)
        return ax

    @staticmethod
    def _validate_2d(X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"期望输入 2D 矩阵，但得到形状 {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("输入矩阵包含 NaN 或 Inf。")
        return X

    def _check_is_fitted(self) -> None:
        if self.mean_ is None or self.Vt_ is None or self.s_ is None or self.rank_ is None:
            raise RuntimeError("SVDLowRankDenoiser 未拟合，请先调用 fit()。")


# =========================
# 模块 2：Wiener 滤波
# =========================

@dataclass
class WienerFilterDenoiser:
    """
    Wiener 频域滤波器（逐行 rFFT）。

    使用方式
    --------
    1) 已有信号估计 X_signal（例如低秩重构结果，中心化域）
    2) 已有噪声估计 X_noise（例如残差，中心化域）
    3) fit(X_signal, X_noise) -> 计算平均 PSD 并生成 Wiener 增益 wiener_gain_
    4) transform(X_signal) -> 对输入按增益滤波并回到时域

    注意
    ----
    - fit 与 transform 的 n_samples 必须一致，否则频点数不匹配
    """

    wiener_beta: float = 1.0
    psd_smooth: int = 9
    gain_floor: float = 0.02

    signal_psd_: Optional[np.ndarray] = None
    noise_psd_: Optional[np.ndarray] = None
    wiener_gain_: Optional[np.ndarray] = None

    def fit(self, X_signal: np.ndarray, X_noise: np.ndarray) -> "WienerFilterDenoiser":
        """
        估计 Wiener 增益。

        输入
        ----
        X_signal : np.ndarray, shape (n_windows, n_samples)
            信号估计（将被滤波的部分）
        X_noise : np.ndarray, shape (n_windows, n_samples)
            噪声估计（用于 PSD 估计；通常为残差）

        输出
        ----
        self : WienerFilterDenoiser
            拟合后的对象（包含 signal_psd_ / noise_psd_ / wiener_gain_）
        """
        X_signal = self._validate_2d(X_signal)
        X_noise = self._validate_2d(X_noise)
        if X_signal.shape != X_noise.shape:
            raise ValueError("X_signal 与 X_noise 形状必须一致。")

        sig_psd = _rfft_psd(X_signal, axis=-1)
        noi_psd = _rfft_psd(X_noise, axis=-1)

        sig_psd = _moving_average_1d(sig_psd, self.psd_smooth)
        noi_psd = _moving_average_1d(noi_psd, self.psd_smooth)

        self.signal_psd_ = sig_psd
        self.noise_psd_ = noi_psd

        # Wiener 增益：G = S / (S + N)
        gain = sig_psd / (sig_psd + noi_psd)
        gain = np.clip(gain, self.gain_floor, 1.0)

        # beta：用于调节抑制强度（>1 更强）
        if self.wiener_beta != 1.0:
            gain = gain ** float(self.wiener_beta)

        gain = _moving_average_1d(gain, self.psd_smooth)
        self.wiener_gain_ = np.clip(gain, self.gain_floor, 1.0)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        对输入做 Wiener 滤波（rFFT * gain -> irFFT）。

        输入
        ----
        X : np.ndarray, shape (n_windows, n_samples)
            待滤波矩阵（通常为 X_signal）。要求 n_samples 与 fit 一致。

        输出
        ----
        Y : np.ndarray, shape (n_windows, n_samples)
            滤波后矩阵（时域）
        """
        self._check_is_fitted()
        X = self._validate_2d(X)

        n_freq_expected = X.shape[1] // 2 + 1
        if len(self.wiener_gain_) != n_freq_expected:
            raise ValueError(
                f"Wiener 增益长度({len(self.wiener_gain_)})与 rFFT 频点数({n_freq_expected})不匹配。"
                "请确保 fit() 与 transform() 的 n_samples 一致。"
            )

        F = np.fft.rfft(X, axis=-1)
        F_filt = F * self.wiener_gain_[None, :]
        return np.fft.irfft(F_filt, n=X.shape[1], axis=-1)

    def fit_transform(self, X_signal: np.ndarray, X_noise: np.ndarray) -> np.ndarray:
        """等价于 fit(X_signal, X_noise) 后 transform(X_signal)。"""
        return self.fit(X_signal, X_noise).transform(X_signal)

    def plot_wiener_gain(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        绘制增益曲线（频点索引 -> 增益）。

        输入
        ----
        ax : matplotlib.axes.Axes or None
            None 则内部创建

        输出
        ----
        ax : matplotlib.axes.Axes
        """
        self._check_is_fitted()
        if ax is None:
            _, ax = plt.subplots()
        g = self.wiener_gain_
        ax.plot(np.arange(len(g)), g)
        ax.set_xlabel("Frequency bin index (rFFT)")
        ax.set_ylabel("Wiener gain")
        ax.set_title("Estimated Wiener gain curve")
        ax.grid(True)
        return ax

    @staticmethod
    def _validate_2d(X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"期望输入 2D 矩阵，但得到形状 {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("输入矩阵包含 NaN 或 Inf。")
        return X

    def _check_is_fitted(self) -> None:
        if self.wiener_gain_ is None or self.signal_psd_ is None or self.noise_psd_ is None:
            raise RuntimeError("WienerFilterDenoiser 未拟合，请先调用 fit()。")


# =========================
# Demo 数据生成
# =========================

def make_synthetic_ncf_windows(
    n_windows: int = 300,
    n_lag: int = 2048,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成示例数据（窗口×lag）用于 demo。

    输出
    ----
    X_noisy : np.ndarray, shape (n_windows, n_lag)
        含干扰/噪声的输入矩阵
    X_clean : np.ndarray, shape (n_windows, n_lag)
        仅包含相干模板（用于对照）
    """
    rng = np.random.default_rng(seed)
    lag = np.linspace(-1.0, 1.0, n_lag)

    def packet(mu, f, w):
        return np.exp(-((lag - mu) / w) ** 2) * np.cos(2 * np.pi * f * (lag - mu))

    template = 1.2 * packet(0.25, f=10, w=0.08) + 1.0 * packet(-0.25, f=10, w=0.08)
    template += 0.6 * packet(0.45, f=6, w=0.12) + 0.6 * packet(-0.45, f=6, w=0.12)

    amps = 1.0 + 0.15 * rng.standard_normal(n_windows)
    X_clean = amps[:, None] * template[None, :]

    spike = np.exp(-(lag / 0.03) ** 2)
    inter_amp = np.zeros(n_windows)
    idx = rng.choice(n_windows, size=n_windows // 3, replace=False)
    inter_amp[idx] = 2.5 + 0.5 * rng.standard_normal(len(idx))
    interference = inter_amp[:, None] * spike[None, :]

    noise = 0.8 * rng.standard_normal((n_windows, n_lag))

    X_noisy = X_clean + interference + noise
    return X_noisy, X_clean


# =========================
# Demo
# =========================

def demo():
    """
    Demo 流程（与原 SVDWF 等价）：

    输入
    ----
    X : (n_windows, n_samples)

    处理
    ----
    1) SVD 低秩：在中心化域得到 X_lr0
    2) 残差估计：residual = X0 - X_lr0
    3) Wiener：用 X_lr0 与 residual 估计增益，并滤波得到 X_filt0
    4) 回到原始基线：X_out = X_filt0 + mean_

    输出
    ----
    X_out : (n_windows, n_samples)
        去噪后的矩阵
    """
    X, _ = make_synthetic_ncf_windows()

    # 1) SVD 低秩
    svd_den = SVDLowRankDenoiser(
        rank=1,            # 若需自动选 rank：设 rank=None，并配置 method/energy/thresh
        method="energy",
        energy=0.90,
        center=True,
        random_sign_fix=True,
    )
    svd_den.fit(X)

    X0 = svd_den.center_data(X)                 # (n_windows, n_samples)
    X_lr0 = svd_den.transform(X, add_mean=False)  # (n_windows, n_samples)，中心化域

    # 2) Wiener（基于低秩部分与残差估计谱）
    residual = X0 - X_lr0
    wien_den = WienerFilterDenoiser(
        psd_smooth=21,
        wiener_beta=1.0,
        gain_floor=0.03,
    )
    wien_den.fit(X_lr0, residual)
    X_filt0 = wien_den.transform(X_lr0)

    # 3) 输出：加回均值
    X_out = X_filt0 + svd_den.mean_

    # 可视化：奇异值谱
    fig1, ax1 = plt.subplots()
    svd_den.plot_spectrum(ax=ax1, log=True)

    # 可视化：Wiener 增益
    fig2, ax2 = plt.subplots()
    wien_den.plot_wiener_gain(ax=ax2)

    # 对比图：输入 / 输出 / 去除部分
    w_slice = slice(0, 120)
    lag_slice = slice(600, 1450)

    fig3, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(X[w_slice, lag_slice], aspect="auto")
    axes[0].set_title("Input X")
    axes[0].set_xlabel("Lag sample index")
    axes[0].set_ylabel("Window index")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(X_out[w_slice, lag_slice], aspect="auto")
    axes[1].set_title(f"Output (rank={svd_den.rank_})")
    axes[1].set_xlabel("Lag sample index")
    axes[1].set_ylabel("Window index")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow((X - X_out)[w_slice, lag_slice], aspect="auto")
    axes[2].set_title("Removed (X - Output)")
    axes[2].set_xlabel("Lag sample index")
    axes[2].set_ylabel("Window index")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.show()

    # 叠加曲线对比（很多流程只看 stack）
    stack_in = X.mean(axis=0)
    stack_out = X_out.mean(axis=0)

    fig4, ax4 = plt.subplots()
    ax4.plot(stack_in, label="Stacked input")
    ax4.plot(stack_out, label="Stacked output")
    ax4.legend()
    ax4.set_title("Stack comparison")
    ax4.grid(True)
    plt.show()


if __name__ == "__main__":
    demo()

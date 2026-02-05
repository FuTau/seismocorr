from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from seismocorr.core.correlation.correlation import CorrelationConfig, CorrelationEngine
from seismocorr.core.correlation.stacking import stack_ccfs
from seismocorr.plugins.processing.three_stations_interferometry import (
    ThreeStationConfig,
    ThreeStationInterferometry,
)


def snr_peak_over_rms(
    lags: np.ndarray,
    ccf: np.ndarray,
    *,
    signal_win: Tuple[float, float],
    noise_win: Tuple[float, float],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """计算 peak/RMS 定义的 SNR。

    定义：
        SNR = max(|ccf| in signal window) / RMS(ccf in noise window)
    """
    t1, t2 = signal_win
    n1, n2 = noise_win
    if t1 > t2 or n1 > n2:
        raise ValueError("window order must be (start, end) with start <= end")

    lags = np.asarray(lags, dtype=float).reshape(-1)
    ccf = np.asarray(ccf, dtype=float).reshape(-1)
    if lags.shape != ccf.shape:
        raise ValueError("lags and ccf must have the same shape")

    sig_mask = (lags >= t1) & (lags <= t2)
    noi_mask = (lags >= n1) & (lags <= n2)

    if not np.any(sig_mask):
        raise ValueError("signal window has no samples; adjust signal_win")
    if not np.any(noi_mask):
        raise ValueError("noise window has no samples; adjust noise_win")

    sig = ccf[sig_mask]
    noi = ccf[noi_mask]

    peak_idx = int(np.argmax(np.abs(sig)))
    peak_val = float(np.abs(sig[peak_idx]))
    peak_time = float(lags[sig_mask][peak_idx])

    noise_rms = float(np.sqrt(np.mean(noi**2)))
    snr = float(peak_val / (noise_rms + eps))

    return {
        "snr": snr,
        "peak": peak_val,
        "noise_rms": noise_rms,
        "peak_time": peak_time,
    }


def align_by_interpolate(
    lags_ref: np.ndarray,
    y_ref: np.ndarray,
    lags_other: np.ndarray,
    y_other: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将 other 插值到 ref 的 lag 网格上，并裁剪到共同 lag 范围。

    目的：
        当两条曲线的 lag 采样网格不一致时，直接裁剪/截断可能造成时间轴错位。
        通过插值到同一网格可保证逐点对齐，便于对比和绘图。
    """
    lags_ref = np.asarray(lags_ref, dtype=float).reshape(-1)
    y_ref = np.asarray(y_ref, dtype=float).reshape(-1)
    lags_other = np.asarray(lags_other, dtype=float).reshape(-1)
    y_other = np.asarray(y_other, dtype=float).reshape(-1)

    if lags_ref.shape != y_ref.shape or lags_other.shape != y_other.shape:
        raise ValueError("lags and y must have matching shapes for both series")

    tmin = float(max(lags_ref.min(), lags_other.min()))
    tmax = float(min(lags_ref.max(), lags_other.max()))
    if tmin >= tmax:
        raise ValueError("no overlapping lag range to align")

    mask = (lags_ref >= tmin) & (lags_ref <= tmax)
    if not np.any(mask):
        raise ValueError("no samples in overlapping lag range after masking")

    lags = lags_ref[mask]
    y1 = y_ref[mask]
    y2 = np.interp(lags, lags_other, y_other)
    return lags, y1, y2


def make_demo_traces(
    n_stations: int,
    n_samples: int,
    *,
    sr: float,
    seed: int = 0,
) -> np.ndarray:
    """生成用于演示的阵列 traces。

    构造：
        - 公共窄带信号 s(t)（带高斯包络）
        - 每个台站引入不同传播时延（线性变化）
        - 叠加局部随机项与噪声

    目标：
        使 direct CCF 与 three-station NCF 在演示中呈现较直观的峰值特征。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sr

    f0 = 6.0
    env = np.exp(-0.5 * ((t - t.mean()) / 4.0) ** 2)
    s = env * np.sin(2 * np.pi * f0 * t)

    traces = np.zeros((n_stations, n_samples), dtype=float)

    # 传播时延趋势（秒），用于模拟线性阵列上的到时差
    delays = np.linspace(-0.6, 0.6, n_stations)

    for i in range(n_stations):
        shift = int(round(delays[i] * sr))
        si = np.roll(s, shift)

        local = 0.15 * rng.standard_normal(n_samples)
        noise = 0.8 * rng.standard_normal(n_samples)
        traces[i] = 0.9 * si + local + noise

    return traces


def _print_lag_info(name: str, lags: np.ndarray) -> None:
    """打印 lag 轴的采样信息，用于快速检查网格与范围。"""
    lags = np.asarray(lags, float).ravel()
    dt = float(np.median(np.diff(lags))) if lags.size > 1 else float("nan")
    print(
        f"{name}: n={lags.size}, range=[{lags.min():.6f}, {lags.max():.6f}], dt~{dt:.6g}, "
        f"closest_to_0={lags[np.argmin(np.abs(lags))]:.6e}"
    )


def main() -> None:
    """演示：Direct CCF 与 Three-Station Interferometry 的对比。

    内容：
        - 对指定台站对 (i, j) 计算 direct cross-correlation
        - 计算 three-station interferometry，得到多条 ncf_ijk
        - 对 ncf_ijk 进行叠加（linear / PWS）
        - 统计 auto 模式下 convolution / correlation 的分段数量
        - 对齐 lag 网格后进行 SNR 比较与可视化
    """
    sr = 200.0
    n_stations = 60
    n_samples = int(sr * 60.0)

    # 1) 构造演示数据
    traces = make_demo_traces(n_stations=n_stations, n_samples=n_samples, sr=sr, seed=0)

    # 2) 一阶互相关设置
    engine = CorrelationEngine()
    max_lag_1 = 2.0
    cc_cfg = CorrelationConfig(method="freq-domain", max_lag=max_lag_1, nfft=None)

    def xcorr_func(x: np.ndarray, y: np.ndarray, *, config: CorrelationConfig):
        return engine.compute_cross_correlation(x, y, sampling_rate=sr, config=config)

    # 3) 三站干涉设置
    tsi = ThreeStationInterferometry(
        sampling_rate=sr,
        xcorr_func=xcorr_func,
        cfg=ThreeStationConfig(mode="auto", max_lag2=2.0),
    )

    i, j = 10, 40
    k_list: Optional[np.ndarray] = None  # None 表示使用全部 k（排除 i/j）

    # A) 直接互相关
    lags_ij, ccf_ij = xcorr_func(traces[i], traces[j], config=cc_cfg)

    # B) 三站干涉：输出多条 ncf_ijk（对应不同 k）
    res = tsi.compute_pair(traces, i=i, j=j, k_list=k_list, config=cc_cfg)
    lags2 = res["lags2"]
    ccfs_ijk = res["ccfs"]
    ks_used = res["ks"]

    if len(ccfs_ijk) == 0:
        raise RuntimeError("No valid ncf_ijk produced; check xcorr or input traces.")

    # 4) auto 分段统计：按索引顺序假定为线性阵列顺序
    lo, hi = (i, j) if i < j else (j, i)
    mode_per_k: List[str] = [
        ("convolution" if (lo < k < hi) else "correlation") for k in ks_used
    ]
    n_conv = sum(m == "convolution" for m in mode_per_k)
    n_corr = sum(m == "correlation" for m in mode_per_k)

    # 分组（用于可选的分组叠加对比）
    ccfs_conv = [c for c, m in zip(ccfs_ijk, mode_per_k) if m == "convolution"]
    ccfs_corr = [c for c, m in zip(ccfs_ijk, mode_per_k) if m == "correlation"]

    # 5) 叠加
    stacked_3s_linear = stack_ccfs(ccfs_ijk, method="linear")
    stacked_3s_pws = stack_ccfs(ccfs_ijk, method="pws", power=2)

    stacked_conv = stack_ccfs(ccfs_conv, method="linear") if len(ccfs_conv) else None
    stacked_corr = stack_ccfs(ccfs_corr, method="linear") if len(ccfs_corr) else None

    # 6) lag 网格检查
    print("\n=== Lag diagnostics ===")
    _print_lag_info("Direct lags", lags_ij)
    _print_lag_info("3S lags2  ", lags2)
    print(f"\n=== 3S details for pair ({i}, {j}) ===")
    print(f"Produced ncf_ijk count = {len(ccfs_ijk)}")
    print(f"auto split: convolution={n_conv}, correlation={n_corr}")
    print(f"example ks (first 10): {ks_used[:10]}")

    # 7) 将 three-station 结果插值到 direct 的 lag 网格，保证逐点对齐
    lags, direct_aligned, lin_aligned = align_by_interpolate(
        lags_ij, ccf_ij, lags2, stacked_3s_linear
    )
    _, _, pws_aligned = align_by_interpolate(lags_ij, ccf_ij, lags2, stacked_3s_pws)

    conv_aligned = None
    corr_aligned = None
    if stacked_conv is not None:
        _, _, conv_aligned = align_by_interpolate(lags_ij, ccf_ij, lags2, stacked_conv)
    if stacked_corr is not None:
        _, _, corr_aligned = align_by_interpolate(lags_ij, ccf_ij, lags2, stacked_corr)

    # 8) SNR 计算
    signal_win = (-0.3, 0.3)
    noise_win = (1.0, 2.0)

    snr_direct = snr_peak_over_rms(lags, direct_aligned, signal_win=signal_win, noise_win=noise_win)
    snr_3s_lin = snr_peak_over_rms(lags, lin_aligned, signal_win=signal_win, noise_win=noise_win)
    snr_3s_pws = snr_peak_over_rms(lags, pws_aligned, signal_win=signal_win, noise_win=noise_win)

    print("\n=== SNR comparison (peak/RMS) ===")
    print(
        "Direct CCF:       "
        f"SNR={snr_direct['snr']:.3f}, peak={snr_direct['peak']:.4g} "
        f"at {snr_direct['peak_time']:.3f}s, noise_rms={snr_direct['noise_rms']:.4g}"
    )
    print(
        "3-station linear: "
        f"SNR={snr_3s_lin['snr']:.3f}, peak={snr_3s_lin['peak']:.4g} "
        f"at {snr_3s_lin['peak_time']:.3f}s, noise_rms={snr_3s_lin['noise_rms']:.4g}"
    )
    print(
        "3-station PWS:    "
        f"SNR={snr_3s_pws['snr']:.3f}, peak={snr_3s_pws['peak']:.4g} "
        f"at {snr_3s_pws['peak_time']:.3f}s, noise_rms={snr_3s_pws['noise_rms']:.4g}"
    )

    # 9) 可视化：Direct vs Three-station stacks（同一 lag 轴）
    plt.figure(figsize=(11, 4))
    plt.plot(lags, direct_aligned, label=f"Direct CCF (SNR={snr_direct['snr']:.2f})")
    plt.plot(lags, lin_aligned, label=f"3S linear stack (SNR={snr_3s_lin['snr']:.2f})")
    plt.plot(lags, pws_aligned, label=f"3S PWS stack (SNR={snr_3s_pws['snr']:.2f})")

    if conv_aligned is not None:
        plt.plot(lags, conv_aligned, linestyle="--", linewidth=1.2, label=f"3S conv-only (n={n_conv})")
    if corr_aligned is not None:
        plt.plot(lags, corr_aligned, linestyle="--", linewidth=1.2, label=f"3S corr-only (n={n_corr})")

    plt.axvspan(signal_win[0], signal_win[1], alpha=0.15, label="signal window")
    plt.axvspan(noise_win[0], noise_win[1], alpha=0.10, label="noise window")
    plt.title(f"Direct CCF vs Three-station Interferometry (pair {i}-{j})")
    plt.xlabel("Lag (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

    # 10) 可视化：展示全部 ncf_ijk
    plt.figure(figsize=(11, 4))
    for idx, (k, m) in enumerate(zip(ks_used, mode_per_k)):
        ls = "-" if m == "convolution" else "--"
        plt.plot(lags2, ccfs_ijk[idx], linewidth=0.8, alpha=0.35, linestyle=ls)

    plt.title(f"All ncf_ijk on lags2 grid (total={len(ccfs_ijk)}; conv={n_conv}, corr={n_corr})")
    plt.xlabel("Lag2 (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    #热力图
    M = np.stack(ccfs_ijk, axis=0)  # shape: (n_k, n_lag2)

    plt.figure(figsize=(11, 5))
    extent = [float(lags2.min()), float(lags2.max()), 0, M.shape[0]]

    plt.imshow(
        M,
        aspect="auto",
        extent=extent,
        origin="lower",
    )
    plt.colorbar(label="Amplitude")
    plt.title(f"All ncf_ijk as image (total={len(ccfs_ijk)}; conv={n_conv}, corr={n_corr})")
    plt.xlabel("Lag2 (s)")
    plt.ylabel("k index (order in ks_used)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

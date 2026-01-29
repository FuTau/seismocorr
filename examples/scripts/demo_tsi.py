from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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

    Args:
        lags: lag 轴数组，shape (n,)。
        ccf: 互相关（或叠加后的 NCF）数组，shape (n,)。
        signal_win: 信号窗 (t1, t2) 秒，例如 (-0.3, 0.3)。
        noise_win: 噪声窗 (t3, t4) 秒，例如 (1.0, 2.0) 或 (-2.0, -1.0)。
        eps: 防止除零的小量。

    Returns:
        dict，包含：
            - snr: SNR 数值
            - peak: signal window 内的峰值（取绝对值）
            - noise_rms: noise window 内 RMS
            - peak_time: 峰值对应的 lag 时间（秒）

    Raises:
        ValueError: 窗口顺序错误或窗口内无样本点。
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


def align_by_common_lag(
    lags_a: np.ndarray,
    ccf_a: np.ndarray,
    lags_b: np.ndarray,
    ccf_b: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """裁剪到共同 lag 范围并按 lag 对齐（假设均匀采样且 dt 一致）。

    适用前提：
        - 两条曲线 lag 轴均匀采样
        - 采样间隔一致（允许小误差）

    Args:
        lags_a: 第一条 lag 轴。
        ccf_a: 第一条曲线。
        lags_b: 第二条 lag 轴。
        ccf_b: 第二条曲线。
        rtol: dt 比较相对容差。
        atol: dt 比较绝对容差。

    Returns:
        (lags_a2, ccf_a2, lags_b2, ccf_b2)，长度一致。

    Raises:
        ValueError: dt 不一致或对齐后无有效样本。
    """
    lags_a = np.asarray(lags_a, dtype=float).reshape(-1)
    ccf_a = np.asarray(ccf_a, dtype=float).reshape(-1)
    lags_b = np.asarray(lags_b, dtype=float).reshape(-1)
    ccf_b = np.asarray(ccf_b, dtype=float).reshape(-1)

    if lags_a.shape != ccf_a.shape or lags_b.shape != ccf_b.shape:
        raise ValueError("lags and ccf must have matching shapes for both series")

    dt_a = float(np.median(np.diff(lags_a)))
    dt_b = float(np.median(np.diff(lags_b)))
    if not np.isclose(dt_a, dt_b, rtol=rtol, atol=atol):
        raise ValueError("lag sampling intervals differ; cannot align by simple slicing")

    tmin = float(max(lags_a.min(), lags_b.min()))
    tmax = float(min(lags_a.max(), lags_b.max()))

    mask_a = (lags_a >= tmin) & (lags_a <= tmax)
    mask_b = (lags_b >= tmin) & (lags_b <= tmax)

    if not np.any(mask_a) or not np.any(mask_b):
        raise ValueError("no overlapping lag range to align")

    lags_a2, ccf_a2 = lags_a[mask_a], ccf_a[mask_a]
    lags_b2, ccf_b2 = lags_b[mask_b], ccf_b[mask_b]

    m = int(min(len(lags_a2), len(lags_b2)))
    return lags_a2[:m], ccf_a2[:m], lags_b2[:m], ccf_b2[:m]


def make_demo_traces(
    n_stations: int,
    n_samples: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """生成演示用 traces（带公共成分）。

    Args:
        n_stations: 台站数（通道数）。
        n_samples: 每条 trace 的样本点数。
        seed: 随机种子。

    Returns:
        traces: shape (n_stations, n_samples)
    """
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(int(n_samples))
    traces = np.stack(
        [0.5 * common + rng.standard_normal(int(n_samples)) for _ in range(int(n_stations))],
        axis=0,
    )
    return traces


def main() -> None:
    """运行：直接互相关 vs 三站干涉（叠加）并比较 SNR。"""
    sr = 200.0
    n_stations = 60
    n_samples = int(sr * 60.0)

    traces = make_demo_traces(n_stations=n_stations, n_samples=n_samples, seed=0)

    engine = CorrelationEngine()
    max_lag_1 = 2.0
    cc_cfg = CorrelationConfig(method="freq-domain", max_lag=max_lag_1, nfft=None)

    def xcorr_func(x: np.ndarray, y: np.ndarray, *, config: CorrelationConfig):
        return engine.compute_cross_correlation(x, y, sampling_rate=sr, config=config)

    tsi = ThreeStationInterferometry(
        sampling_rate=sr,
        xcorr_func=xcorr_func,
        cfg=ThreeStationConfig(mode="auto", max_lag2=2.0),
    )

    i, j = 10, 40
    k_list: Optional[np.ndarray] = None

    # A) 直接互相关
    lags_ij, ccf_ij = xcorr_func(traces[i], traces[j], config=cc_cfg)

    # B) 三站干涉：得到多条 ncf_ijk
    res = tsi.compute_pair(traces, i=i, j=j, k_list=k_list, config=cc_cfg)
    lags2 = res["lags2"]
    ccfs_ijk = res["ccfs"]

    stacked_3s_linear = stack_ccfs(ccfs_ijk, method="linear")
    stacked_3s_pws = stack_ccfs(ccfs_ijk, method="pws", power=2)

    # 对齐到共同 lag
    l1, a1, l2, a2 = align_by_common_lag(lags_ij, ccf_ij, lags2, stacked_3s_linear)
    _, b1, _, b2 = align_by_common_lag(lags_ij, ccf_ij, lags2, stacked_3s_pws)

    signal_win = (-0.3, 0.3)
    noise_win = (1.0, 2.0)

    snr_direct = snr_peak_over_rms(l1, a1, signal_win=signal_win, noise_win=noise_win)
    snr_3s_lin = snr_peak_over_rms(l2, a2, signal_win=signal_win, noise_win=noise_win)
    snr_3s_pws = snr_peak_over_rms(l2, b2, signal_win=signal_win, noise_win=noise_win)

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

    # 可视化对比
    plt.figure(figsize=(10, 4))
    plt.plot(l1, a1, label=f"Direct CCF (SNR={snr_direct['snr']:.2f})")
    plt.plot(l2, a2, label=f"3-station linear stack (SNR={snr_3s_lin['snr']:.2f})")
    plt.plot(l2, b2, label=f"3-station PWS stack (SNR={snr_3s_pws['snr']:.2f})")
    plt.axvspan(signal_win[0], signal_win[1], alpha=0.15, label="signal window")
    plt.axvspan(noise_win[0], noise_win[1], alpha=0.10, label="noise window")
    plt.title(f"Direct CCF vs Three-station Interferometry (pair {i}-{j})")
    plt.xlabel("Lag (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

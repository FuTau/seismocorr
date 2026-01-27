# seismocorr/plugins/three_stations_interferometry.py
"""
Three-Station Interferometry (三台/三站干涉) - Minimal Orchestrator

1) 输入：直接输入 traces (N,T) + (i,j) / pairs + k_list（None 默认全部k）
2) 输出：多对 NCF（每对输出多条 ncf_ijk），输出结构可直接接入 stacking.py
3) 本模块只调用外部互相关函数，不调用叠加函数，不做预处理

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np

Array = np.ndarray
LagsAndCCF = Tuple[np.ndarray, np.ndarray]
XCorrFunc = Callable[..., LagsAndCCF]  # 外部互相关函数：xcorr(x,y,**kw)->(lags,ccf)


class PairNCFResult(TypedDict):
    lags2: np.ndarray
    ccfs: List[np.ndarray]
    ks: List[int]


@dataclass
class ThreeStationConfig:
    """
    mode:
      - "correlation": 二次干涉固定用互相关
      - "convolution": 二次干涉固定用卷积
      - "auto": 线性阵列自动分段：
          k 在 i/j 中间 -> convolution
          否则 -> correlation
    """
    mode: str = "auto"                      # "correlation" | "convolution" | "auto"
    second_stage_nfft: Optional[int] = None
    max_lag2: Optional[float] = None


class ThreeStationInterferometry:
    def __init__(
        self,
        sampling_rate: float,
        xcorr_func: XCorrFunc,
        cfg: Optional[ThreeStationConfig] = None,
    ):
        self.sr = float(sampling_rate)
        self.xcorr = xcorr_func
        self.cfg = cfg or ThreeStationConfig()

        if self.cfg.mode not in ("correlation", "convolution", "auto"):
            raise ValueError('ThreeStationConfig.mode must be "correlation", "convolution", or "auto"')

    def compute_pair(
        self,
        traces: np.ndarray,
        i: int,
        j: int,
        k_list: Optional[Sequence[int]] = None,
        **xcorr_kwargs,
    ) -> PairNCFResult:
        if traces.ndim != 2:
            raise ValueError("traces must be a 2D array with shape (N, T)")
        n_stations = traces.shape[0]
        if not (0 <= i < n_stations and 0 <= j < n_stations):
            raise IndexError("i/j out of range")
        if i == j:
            raise ValueError("i and j must be different")

        # 默认 k：全部（排除 i/j）
        if k_list is None:
            ks = [k for k in range(n_stations) if k not in (i, j)]
        else:
            ks = [int(k) for k in k_list if 0 <= int(k) < n_stations and int(k) not in (i, j)]

        if not ks:
            return {"lags2": np.array([]), "ccfs": [], "ks": []}

        xi = traces[i]
        xj = traces[j]

        # 用第一个 k 定标：固定二次输出长度（保证可直接 stacking）
        k0 = ks[0]
        _, ccf_ik0 = self.xcorr(xi, traces[k0], **xcorr_kwargs)
        _, ccf_jk0 = self.xcorr(xj, traces[k0], **xcorr_kwargs)

        if ccf_ik0.size == 0 or ccf_jk0.size == 0:
            return {"lags2": np.array([]), "ccfs": [], "ks": []}

        base_len = min(ccf_ik0.shape[-1], ccf_jk0.shape[-1])
        nfft2 = self._choose_nfft2(base_len)

        lags2_full = self._lags_for_len(nfft2)
        if self.cfg.max_lag2 is not None:
            crop_start, crop_end = self._crop_indices(nfft2, float(self.cfg.max_lag2))
            lags2 = lags2_full[crop_start:crop_end]
        else:
            crop_start, crop_end = 0, nfft2
            lags2 = lags2_full

        ccfs: List[np.ndarray] = []
        ks_used: List[int] = []

        # 线性阵列分段依据（索引顺序即空间顺序）
        lo, hi = (i, j) if i < j else (j, i)

        for k in ks:
            xk = traces[k]

            _, ccf_ik = self.xcorr(xi, xk, **xcorr_kwargs)
            _, ccf_jk = self.xcorr(xj, xk, **xcorr_kwargs)
            if ccf_ik.size == 0 or ccf_jk.size == 0:
                continue

            m = min(ccf_ik.shape[-1], ccf_jk.shape[-1], base_len)
            ccf_ik = ccf_ik[:m]
            ccf_jk = ccf_jk[:m]

            # ========= 自动分段：为每个 k 选择二次干涉模式 =========
            mode_k = self._mode_for_k(i=i, j=j, k=k, lo=lo, hi=hi)

            ncf_full = self._second_stage(ccf_ik, ccf_jk, nfft2, mode_k=mode_k)
            ncf = ncf_full[crop_start:crop_end]

            ccfs.append(ncf)
            ks_used.append(int(k))

        return {"lags2": lags2, "ccfs": ccfs, "ks": ks_used}

    def compute_many(
        self,
        traces: np.ndarray,
        pairs: Sequence[Tuple[int, int]],
        k_list: Optional[Sequence[int]] = None,
        **xcorr_kwargs,
    ) -> Dict[str, PairNCFResult]:
        results: Dict[str, PairNCFResult] = {}
        for (i, j) in pairs:
            results[f"{i}--{j}"] = self.compute_pair(
                traces=traces, i=i, j=j, k_list=k_list, **xcorr_kwargs
            )
        return results

    def _mode_for_k(self, *, i: int, j: int, k: int, lo: int, hi: int) -> str:
        """
        线性DAS阵列自动分段：
          - k 在 (lo, hi) 开区间内 => convolution
          - 否则 => correlation

        若 cfg.mode 不是 auto，则直接返回 cfg.mode。
        """
        if self.cfg.mode != "auto":
            return self.cfg.mode

        # k 是否在 i/j 中间（按索引顺序）
        between = (lo < k < hi)
        return "convolution" if between else "correlation"

    def _choose_nfft2(self, base_len: int) -> int:
        if self.cfg.second_stage_nfft is None:
            return int(2 ** np.ceil(np.log2(max(2, base_len))))
        nfft = int(self.cfg.second_stage_nfft)
        if nfft < base_len:
            raise ValueError(f"second_stage_nfft ({nfft}) must be >= base_len ({base_len}).")
        return nfft

    def _second_stage(self, ccf_ik: Array, ccf_jk: Array, nfft2: int, *, mode_k: str) -> Array:
        Fik = np.fft.fft(ccf_ik, n=nfft2)
        Fjk = np.fft.fft(ccf_jk, n=nfft2)

        if mode_k == "correlation":
            F = Fik * np.conj(Fjk)
        elif mode_k == "convolution":
            F = Fik * Fjk
        else:
            raise ValueError(f"Unknown mode_k={mode_k}")

        ncf = np.real(np.fft.ifft(F))
        return np.fft.fftshift(ncf)

    def _lags_for_len(self, n: int) -> Array:
        return (np.arange(n) - (n // 2)) / self.sr

    def _crop_indices(self, n: int, max_lag2: float) -> Tuple[int, int]:
        half = int(round(max_lag2 * self.sr))
        center = n // 2
        start = max(0, center - half)
        end = min(n, center + half + 1)
        return start, end


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from seismocorr.core.correlation.correlation import CorrelationEngine, CorrelationConfig
    from seismocorr.core.correlation.stacking import stack_ccfs

    # -----------------------------
    # 1) 一个通用的 SNR 计算函数
    # -----------------------------
    def snr_peak_over_rms(
        lags: np.ndarray,
        ccf: np.ndarray,
        *,
        signal_win: tuple[float, float],
        noise_win: tuple[float, float],
    ) -> dict:
        """
        SNR 定义：SNR = max(|ccf| in signal window) / RMS(ccf in noise window)

        Args:
            signal_win: (t1, t2) 秒，比如 (-0.3, 0.3)
            noise_win : (t3, t4) 秒，比如 (1.0, 2.0) 或 (-2.0,-1.0)
                      你也可以传正窗或负窗，这里按绝对值 RMS 计算

        Returns:
            dict: snr, peak, noise_rms, peak_time
        """
        t1, t2 = signal_win
        n1, n2 = noise_win
        if t1 > t2 or n1 > n2:
            raise ValueError("window order must be (start, end) with start<=end")

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

        noise_rms = float(np.sqrt(np.mean(noi.astype(float) ** 2)))
        snr = float(peak_val / (noise_rms + 1e-12))

        return {
            "snr": snr,
            "peak": peak_val,
            "noise_rms": noise_rms,
            "peak_time": peak_time,
        }

    def align_by_common_lag(lags_a, ccf_a, lags_b, ccf_b):
        """
        把两条曲线裁剪到共同的 lag 范围，并按 lag 对齐（假设采样间隔相同）。
        适用于：两者都是均匀采样且 dt=1/sr。
        """
        dt_a = np.median(np.diff(lags_a))
        dt_b = np.median(np.diff(lags_b))
        if not np.isclose(dt_a, dt_b, rtol=1e-6, atol=1e-12):
            raise ValueError("lag sampling intervals differ; cannot align by simple slicing")

        tmin = max(lags_a.min(), lags_b.min())
        tmax = min(lags_a.max(), lags_b.max())

        mask_a = (lags_a >= tmin) & (lags_a <= tmax)
        mask_b = (lags_b >= tmin) & (lags_b <= tmax)

        lags_a2, ccf_a2 = lags_a[mask_a], ccf_a[mask_a]
        lags_b2, ccf_b2 = lags_b[mask_b], ccf_b[mask_b]

        # 再次确保长度一致（可能因端点取整差1个点）
        m = min(len(lags_a2), len(lags_b2))
        return lags_a2[:m], ccf_a2[:m], lags_b2[:m], ccf_b2[:m]

    # -----------------------------
    # 2) 生成/读取数据（你换成真实 traces 即可）
    # -----------------------------
    sr = 200.0
    N = 60
    T = int(sr * 60)
    rng = np.random.default_rng(0)

    # 演示用：带公共成分的随机信号（真实数据请直接用你的 traces）
    common = rng.standard_normal(T)
    traces = np.stack([0.5 * common + rng.standard_normal(T) for _ in range(N)], axis=0)

    # -----------------------------
    # 3) 外部互相关函数（调用你包里的 CorrelationEngine）
    # -----------------------------
    engine = CorrelationEngine()

    # 一次互相关窗口（论文/常规 NCF 都会先截到 ±max_lag）
    max_lag_1 = 2.0
    cc_cfg = CorrelationConfig(method="freq-domain", max_lag=max_lag_1, nfft=None)

    def xcorr_func(x: np.ndarray, y: np.ndarray, *, config: CorrelationConfig):
        return engine.compute_cross_correlation(x, y, sampling_rate=sr, config=config)

    # -----------------------------
    # 4) 三站干涉（AUTO 分段）对象
    # -----------------------------
    # 你用的是我给你的 auto 分段版（k 在中间卷积，否则相关）
    tsi = ThreeStationInterferometry(
        sampling_rate=sr,
        xcorr_func=xcorr_func,
        cfg=ThreeStationConfig(mode="auto", max_lag2=2.0),  # 二次输出也裁到 ±2s，便于对齐比较
    )

    # -----------------------------
    # 5) 选一对台站做对比
    # -----------------------------
    i, j = 10, 40
    k_list = None  # None = 默认全部 k（排除 i/j）

    # A) 直接互相关
    lags_ij, ccf_ij = xcorr_func(traces[i], traces[j], config=cc_cfg)

    # B) 三站干涉：得到很多条 ncf_ijk
    res = tsi.compute_pair(traces, i=i, j=j, k_list=k_list, config=cc_cfg)
    lags2 = res["lags2"]
    ccfs_ijk = res["ccfs"]

    # B-1) 叠加（只在 demo 做）：你可换 linear / pws / robust 等
    stacked_3s_linear = stack_ccfs(ccfs_ijk, method="linear")
    stacked_3s_pws = stack_ccfs(ccfs_ijk, method="pws", power=2)

    # -----------------------------
    # 6) 对齐到共同 lag 再算 SNR
    # -----------------------------
    l1, a1, l2, a2 = align_by_common_lag(lags_ij, ccf_ij, lags2, stacked_3s_linear)
    _,  b1, _,  b2 = align_by_common_lag(lags_ij, ccf_ij, lags2, stacked_3s_pws)

    # 你自己设窗：这里给一个常见例子
    # signal: 中心附近（比如面波主能量在小延迟）
    # noise : 远离中心的一段
    signal_win = (-0.3, 0.3)
    noise_win = (1.0, 2.0)

    snr_direct = snr_peak_over_rms(l1, a1, signal_win=signal_win, noise_win=noise_win)
    snr_3s_lin = snr_peak_over_rms(l2, a2, signal_win=signal_win, noise_win=noise_win)
    snr_3s_pws = snr_peak_over_rms(l2, b2, signal_win=signal_win, noise_win=noise_win)

    print("\n=== SNR comparison (peak/RMS) ===")
    print(f"Direct CCF:      SNR={snr_direct['snr']:.3f}, peak={snr_direct['peak']:.4g} at {snr_direct['peak_time']:.3f}s, noise_rms={snr_direct['noise_rms']:.4g}")
    print(f"3-station linear: SNR={snr_3s_lin['snr']:.3f}, peak={snr_3s_lin['peak']:.4g} at {snr_3s_lin['peak_time']:.3f}s, noise_rms={snr_3s_lin['noise_rms']:.4g}")
    print(f"3-station PWS:    SNR={snr_3s_pws['snr']:.3f}, peak={snr_3s_pws['peak']:.4g} at {snr_3s_pws['peak_time']:.3f}s, noise_rms={snr_3s_pws['noise_rms']:.4g}")

    # -----------------------------
    # 7) 可视化对比
    # -----------------------------
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


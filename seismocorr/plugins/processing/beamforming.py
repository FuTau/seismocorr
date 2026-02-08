# seismocorr/plugins/processing/beamforming.py
"""
聚束成形计算模块（频域延时叠加 / Bartlett）。

数据约定：
    - data: (n_chan, n_samples)
    - xy_m: (n_chan, 2)，单位 m
    - 输出功率图 power: (Ns, Naz)

功能：
    - Beamformer：对多通道数据做频域延时叠加（Bartlett）扫描，
      输出 BeamformingResult。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window
from seismocorr.config.builder import BeamformingConfig


@dataclass(frozen=True)
class DelayAndSumResult:
    """
    聚束成形输出结果容器。

    Attributes:
        azimuth_deg: 方位角数组 (Naz,)，单位：度
        slowness_s_per_m: 慢度数组 (Ns,)，单位：s/m
        power: 聚束功率谱 (Ns, Naz)
        freqs_hz: 参与计算的频率数组 (Nf_band,)
        meta: 结果相关的元数据字典
    """

    azimuth_deg: np.ndarray
    slowness_s_per_m: np.ndarray
    power: np.ndarray
    freqs_hz: np.ndarray
    meta: Dict[str, Any]


@dataclass(frozen=True)
class BackprojectionResult:
    """
    反投影输出结果容器。

    Attributes:
        power: 反投影能量图 (n_grid,)
        freqs_hz: 参与计算的频率数组 (Nf_band,)
        frame_power: 可选的逐帧能量 (n_frames, n_grid)；若未请求则为 None
        meta: 元数据
    """
    power: np.ndarray
    freqs_hz: np.ndarray
    frame_power: np.ndarray | None
    meta: Dict[str, Any]


class Beamformer:
    """Frequency-domain delay-and-sum (Bartlett) beamformer.

    该类实现经典 Bartlett/延时叠加聚束成形的频域版本：对多通道数据按帧做 rFFT，
    在指定频带内对每个扫描方向(azimuth)与慢度(slowness)构造相位延迟项，
    将所有通道频谱相干叠加得到波束输出，并以平均功率作为扫描能量图。

    处理流程（核心步骤）：
        1) 分帧：每通道以 frame_len_s / hop_s 做无补零分帧，并施加窗函数；
        2) 频域：对每帧做 rFFT，选取 [fmin, fmax] 内频点；
        3) （可选）白化：按通道/频点对幅度做归一化，降低谱形差异的影响；
        4) 扫描：对每个 (slow, azimuth) 计算相位因子
               exp(j * (r·u) * slow * ω)
           并将通道频谱叠加形成 beam(t, n, f)；
        5) 输出功率：对 time-frame 和频点取均值得到 power(slow, azimuth)。

    坐标与方位角定义：
        - 阵列坐标 xy_m: (x=East, y=North)，单位 m。
        - azimuth_deg: 0°=北，顺时针为正，输出表示“传播方向”(propagation direction)，
          与代码中 u = (sin(azi), cos(azi)) 的定义一致。

    Args:
        fs: 采样率 (Hz)。
        fmin: 频带下限 (Hz)，需满足 0 <= fmin < fmax。
        fmax: 频带上限 (Hz)。
        frame_len_s: 分帧长度（秒）。
        hop_s: 帧移长度（秒）。
        window: 窗函数名称（scipy.signal.get_window）。
        whiten: 是否进行频谱白化（用跨帧 RMS 幅度对频谱归一）。
        eps: 白化时避免除零的小量。

    Returns:
        beamform(): 返回 BeamformingResult，其中包含：
            - azimuth_deg: 扫描方位角数组 (Naz,)
            - slowness_s_per_m: 扫描慢度数组 (Ns,)
            - power: 聚束功率图 (Ns, Naz)
            - freqs_hz: 实际参与叠加的频点 (Nf,)
            - meta: 参数与约定信息字典，便于复现与记录

    Notes:
        - 该实现为 Bartlett（非自适应）方法：只做相位对齐与求和，不含 MVDR/Capon 等权重估计。
        - 分帧为“无补零”方式；若数据长度不足以形成一帧会抛出异常。
        - 白化有利于抑制强窄带/谱形差异，但也可能降低真实幅度信息；可按任务需求关闭。
    """
    def __init__(self, config=None):
        """初始化 Beamformer。

        Args:
            fs: 采样率 (Hz)
            fmin: 频带下限 (Hz)
            fmax: 频带上限 (Hz)
            frame_len_s: 分帧长度 (秒)
            hop_s: 帧移长度 (秒)
            window: 窗函数名称（scipy.signal.get_window）
            whiten: 是否进行频谱白化
            eps: 防止除零的小量

        Raises:
            ValueError: 参数取值不合法
        """
        self.config = config or BeamformingConfig()

    def delay_and_sum(
        self,
        data: np.ndarray,
        xy_m: np.ndarray,
        azimuth_deg: np.ndarray,
        slowness_s_per_m: np.ndarray,
    ) -> DelayAndSumResult:
        """执行频域延时叠加（Bartlett）聚束成形扫描。

        Args:
            data: 多通道时间序列数据 (n_chan, n_samples)
            xy_m: 台阵坐标 (n_chan, 2)，单位 m，x=East, y=North
            azimuth_deg: 扫描方位角数组 (Naz,)，0°=北，顺时针
            slowness_s_per_m: 扫描慢度数组 (Ns,)，单位 s/m

        Returns:
            BeamformingResult: 聚束成形结果对象

        Raises:
            ValueError: 输入数据维度或参数不合法
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("data 必须是二维数组 (n_chan, n_samples)")

        n_chan, n_samp = data.shape
        if n_chan < 1 or n_samp < 2:
            raise ValueError("data 尺寸不合法")

        xy_m = np.asarray(xy_m, dtype=float)
        if xy_m.shape != (n_chan, 2):
            raise ValueError("xy_m 必须为 (n_chan, 2)，且与 data 通道数一致")

        azimuth_deg = np.asarray(azimuth_deg, dtype=float).reshape(-1)
        slowness_s_per_m = np.asarray(slowness_s_per_m, dtype=float).reshape(-1)

        if azimuth_deg.size == 0:
            raise ValueError("azimuth_deg 不能为空")
        if slowness_s_per_m.size == 0:
            raise ValueError("slowness_s_per_m 不能为空")

        frame_len = int(round(self.frame_len_s * self.fs))
        hop = int(round(self.hop_s * self.fs))

        if frame_len <= 1 or hop <= 0:
            raise ValueError("由 frame_len_s/hop_s 计算得到的帧长/步长不合法")
        if n_samp < frame_len:
            raise ValueError("n_samples < frame_len")

        win = get_window(self.window, frame_len, fftbins=True).astype(float)

        freqs = rfftfreq(frame_len, d=1.0 / self.fs)
        band = (freqs >= self.fmin) & (freqs <= self.fmax)
        freqs_band = freqs[band]

        if freqs_band.size == 0:
            raise ValueError("频带内无可用频点")

        n_frames = 1 + (n_samp - frame_len) // hop
        if n_frames < 1:
            raise ValueError("无法形成有效分帧")

        spectra = np.empty(
            (n_chan, n_frames, freqs_band.size),
            dtype=np.complex128,
        )

        for chan_idx in range(n_chan):
            frames = self._frame_1d(data[chan_idx], frame_len, hop)
            frames *= win[None, :]
            spectra[chan_idx] = rfft(frames, axis=-1)[:, band]

        if self.whiten:
            amp = (
                np.sqrt(np.mean(np.abs(spectra) ** 2, axis=1, keepdims=True))
                + self.eps
            )
            spectra /= amp

        azimuth_rad = np.deg2rad(azimuth_deg)
        ux = np.sin(azimuth_rad)[None, :]
        uy = np.cos(azimuth_rad)[None, :]
        r_dot_u = xy_m[:, 0:1] * ux + xy_m[:, 1:2] * uy

        omega = 2.0 * np.pi * freqs_band
        spectra_tcf = np.transpose(spectra, (1, 0, 2))

        power = np.zeros(
            (slowness_s_per_m.size, azimuth_deg.size),
            dtype=float,
        )

        for i_s, slow in enumerate(slowness_s_per_m):
            phase = np.exp(
                1j * r_dot_u[:, :, None] * slow * omega[None, None, :]
            )
            beam = np.einsum(
                "tcf,cnf->tnf",
                spectra_tcf,
                phase,
                optimize=True,
            )
            power[i_s] = np.mean(np.mean(np.abs(beam) ** 2, axis=0), axis=-1)

        meta = {
            "method": "delay_and_sum_frequency_domain",
            "fs": self.fs,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "frame_len_s": self.frame_len_s,
            "hop_s": self.hop_s,
            "window": self.window,
            "whiten": self.whiten,
            "n_frames": int(n_frames),
            "n_chan": int(n_chan),
            "azimuth_definition": "from_north_clockwise_deg",
            "azimuth_output": "propagation_direction",
            "coords": "x_east_m,y_north_m",
        }

        return DelayAndSumResult(
            azimuth_deg=azimuth_deg,
            slowness_s_per_m=slowness_s_per_m,
            power=power,
            freqs_hz=freqs_band,
            meta=meta,
        )

    def backproject(
        self,
        data: np.ndarray,
        travel_times_s: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        return_frame_power: bool = False,
        chunk_size: int = 256,
    ) -> BackprojectionResult:
        """
        频域反投影（Backprojection）：按给定走时表对齐后做相干叠加，输出空间能量图。

        Args:
            data: (n_chan, n_samples)
            travel_times_s: 走时表，形状 (n_grid, n_chan)，单位秒
            weights: (n_chan,) 可选通道权重（如 SNR 权重/距离权重），默认全 1
            return_frame_power: 是否返回逐帧能量 (n_frames, n_grid)
            chunk_size: 网格分块大小，避免一次性 phase 占用过大内存

        Returns:
            BackprojectionResult:
                power: (n_grid,)
                freqs_hz: (Nf_band,)
                frame_power: (n_frames, n_grid) 或 None
                meta: 参数字典
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("data 必须是二维数组 (n_chan, n_samples)")
        n_chan, n_samp = data.shape
        if n_chan < 1 or n_samp < 2:
            raise ValueError("data 尺寸不合法")

        travel_times_s = np.asarray(travel_times_s, dtype=float)
        if travel_times_s.ndim != 2:
            raise ValueError("travel_times_s 必须是二维数组 (n_grid, n_chan)")
        n_grid, n_chan_tt = travel_times_s.shape
        if n_chan_tt != n_chan:
            raise ValueError("travel_times_s 的第二维必须等于 n_chan")

        if weights is None:
            w = np.ones((n_chan,), dtype=float)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.size != n_chan:
                raise ValueError("weights 必须是 (n_chan,)")

        frame_len = int(round(self.frame_len_s * self.fs))
        hop = int(round(self.hop_s * self.fs))
        if frame_len <= 1 or hop <= 0:
            raise ValueError("由 frame_len_s/hop_s 计算得到的步长不合法")
        if n_samp < frame_len:
            raise ValueError("n_samples < frame_len")

        win = get_window(self.window, frame_len, fftbins=True).astype(float)

        freqs = rfftfreq(frame_len, d=1.0 / self.fs)
        band = (freqs >= self.fmin) & (freqs <= self.fmax)
        freqs_band = freqs[band]
        if freqs_band.size == 0:
            raise ValueError("频带内无可用频点")

        n_frames = 1 + (n_samp - frame_len) // hop
        if n_frames < 1:
            raise ValueError("无法形成有效分帧")
        
        spectra = np.empty((n_chan, n_frames, freqs_band.size), dtype=np.complex128)
        for chan_idx in range(n_chan):
            frames = self._frame_1d(data[chan_idx], frame_len, hop)
            frames *= win[None, :]
            spectra[chan_idx] = rfft(frames, axis=-1)[:, band]

        if self.whiten:
            amp = np.sqrt(np.mean(np.abs(spectra) ** 2, axis=1, keepdims=True)) + self.eps
            spectra /= amp

        spectra *= w[:, None, None]

        spectra_tcf = np.transpose(spectra, (1, 0, 2))

        omega = 2.0 * np.pi * freqs_band  # (f,)

        power = np.zeros((n_grid,), dtype=float)
        frame_power = np.zeros((n_frames, n_grid), dtype=float) if return_frame_power else None

        for g0 in range(0, n_grid, int(chunk_size)):
            g1 = min(n_grid, g0 + int(chunk_size))
            tau = travel_times_s[g0:g1, :] 

            mask = np.isfinite(tau)  
            tau0 = np.where(mask, tau, 0.0)
            phase = np.exp(1j * tau0[:, :, None] * omega[None, None, :])
            phase *= mask[:, :, None]  # 无效通道不贡献
            phase = np.exp(1j * tau[:, :, None] * omega[None, None, :])

            beam = np.einsum("tcf,gcf->tgf", spectra_tcf, phase, optimize=True)

            if return_frame_power:
                fp = np.mean(np.abs(beam) ** 2, axis=-1)
                frame_power[:, g0:g1] = fp

            p = np.mean(np.mean(np.abs(beam) ** 2, axis=0), axis=-1)
            power[g0:g1] = p

        meta = {
            "method": "backprojection_frequency_domain",
            "fs": self.fs,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "frame_len_s": self.frame_len_s,
            "hop_s": self.hop_s,
            "window": self.window,
            "whiten": self.whiten,
            "n_frames": int(n_frames),
            "n_chan": int(n_chan),
            "n_grid": int(n_grid),
            "travel_time_convention": "phase = exp(+j*omega*tau) for alignment",
            "recommendation": "use relative travel times to reduce model bias",
        }

        return BackprojectionResult(
            power=power,
            freqs_hz=freqs_band,
            frame_power=frame_power,
            meta=meta,
        )

    @staticmethod
    def _frame_1d(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
        """对一维信号进行无补零分帧。

        Args:
            x: 一维时间序列
            frame_len: 帧长度（采样点）
            hop: 帧移（采样点）

        Returns:
            frames: 分帧结果数组 (n_frames, frame_len)

        Raises:
            ValueError: 输入维度或长度不合法
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("x 必须是一维数组")

        n_samples = x.size
        if n_samples < frame_len:
            raise ValueError("信号长度小于一帧")

        n_frames = 1 + (n_samples - frame_len) // hop
        idx = (
            np.arange(frame_len)[None, :]
            + hop * np.arange(n_frames)[:, None]
        )
        return x[idx]

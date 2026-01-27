# beamforming.py
# -*- coding: utf-8 -*-
"""
聚束成形计算模块（频域延时叠加 / Bartlett）。

数据约定：
- data: (n_chan, n_samples)
- xy_m: (n_chan, 2)，单位 m
- 输出功率图 power: (Ns, Naz)

功能：
- Beamformer：对多通道数据做频域延时叠加（Bartlett）扫描，输出 BeamformingResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq


@dataclass(frozen=True)
class BeamformingResult:
    """聚束成形输出容器。"""
    azimuth_deg: np.ndarray          # (Naz,)
    slowness_s_per_m: np.ndarray     # (Ns,)
    power: np.ndarray                # (Ns, Naz)
    freqs_hz: np.ndarray             # (Nf_band,)
    meta: Dict


class Beamformer:
    """
    聚束成形计算类。

    初始化参数用于复用（采样率、频带、分帧、窗函数、白化设置）。
    """

    def __init__(
        self,
        fs: float,
        fmin: float = 1.0,
        fmax: float = 20.0,
        frame_len_s: float = 4.0,
        hop_s: float = 2.0,
        window: str = "hann",
        whiten: bool = True,
        eps: float = 1e-12,
    ) -> None:
        self.fs = float(fs)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.frame_len_s = float(frame_len_s)
        self.hop_s = float(hop_s)
        self.window = str(window)
        self.whiten = bool(whiten)
        self.eps = float(eps)

        if self.fs <= 0:
            raise ValueError("fs 必须为正数")
        if self.fmin < 0 or self.fmax <= 0 or self.fmax <= self.fmin:
            raise ValueError("频带需要满足 0 <= fmin < fmax")
        if self.frame_len_s <= 0 or self.hop_s <= 0:
            raise ValueError("frame_len_s 与 hop_s 必须为正数")

    def beamform(
        self,
        data: np.ndarray,
        xy_m: np.ndarray,
        azimuth_deg: np.ndarray,
        slowness_s_per_m: np.ndarray,
    ) -> BeamformingResult:
        """
        频域延时叠加（Bartlett）扫描。

        参数：
        - data: (n_chan, n_samples)
        - xy_m: (n_chan, 2)
        - azimuth_deg: (Naz,)
        - slowness_s_per_m: (Ns,)

        返回：
        - BeamformingResult，power: (Ns, Naz)
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
        if azimuth_deg.size < 1:
            raise ValueError("azimuth_deg 不能为空")
        if slowness_s_per_m.size < 1:
            raise ValueError("slowness_s_per_m 不能为空")

        frame_len = int(round(self.frame_len_s * self.fs))
        hop = int(round(self.hop_s * self.fs))
        if frame_len <= 1 or hop <= 0:
            raise ValueError("由 frame_len_s/hop_s 计算得到的帧长/步长不合法")
        if n_samp < frame_len:
            raise ValueError("n_samples < frame_len：请增大数据长度或减小 frame_len_s")

        # 构造窗函数
        win = get_window(self.window, frame_len, fftbins=True).astype(float)

        # 频率轴与频带选择
        freqs = rfftfreq(frame_len, d=1.0 / self.fs)
        band = (freqs >= self.fmin) & (freqs <= self.fmax)
        fb = freqs[band]
        if fb.size == 0:
            raise ValueError("频带内无可用频点：请调整 frame_len_s 或 fmin/fmax")

        # 分帧数量（不补零）
        n_frames = 1 + (n_samp - frame_len) // hop
        if n_frames < 1:
            raise ValueError("无法形成有效分帧")

        # 计算每通道、每帧的频谱（仅保留频带内频点）
        X = np.empty((n_chan, n_frames, fb.size), dtype=np.complex128)  # (c, t, f)
        for c in range(n_chan):
            frames = self._frame_1d(data[c], frame_len, hop)  # (t, L)
            frames = frames * win[None, :]
            F = rfft(frames, axis=-1)  # (t, n_freq)
            X[c] = F[:, band]

        # 频谱白化（按通道、频点对帧均方根归一）
        if self.whiten:
            amp = np.sqrt(np.mean(np.abs(X) ** 2, axis=1, keepdims=True)) + self.eps
            X = X / amp

        # 角度约定：azimuth 从北顺时针（0=North, 90=East）
        # 坐标约定：x=East, y=North
        az = np.deg2rad(azimuth_deg)
        ux = np.sin(az)[None, :]  # (1, Naz)
        uy = np.cos(az)[None, :]
        rdotu = xy_m[:, 0:1] * ux + xy_m[:, 1:2] * uy  # (n_chan, Naz)


        omega = 2.0 * np.pi * fb  # (n_fb,)
        Xt = np.transpose(X, (1, 0, 2))  # (t, c, f)

        Ns = slowness_s_per_m.size
        Naz = azimuth_deg.size
        power = np.zeros((Ns, Naz), dtype=float)

        # 扫描慢度：对每个慢度构造相位项并对通道求和
        for i_s, s in enumerate(slowness_s_per_m):
            # 传播方向扫描：x_i(t)≈A e^{jω(t-τ_i)}, τ_i = s (r_i·u)
            # 对齐补偿相位：e^{+jωτ_i}
            phase = np.exp(+1j * (rdotu[:, :, None] * s) * omega[None, None, :])
            # y: (t, Naz, f)
            y = np.einsum("tcf,cnf->tnf", Xt, phase, optimize=True)
            # p: (Naz, f)
            p = np.mean(np.abs(y) ** 2, axis=0)
            # power: (Naz,)
            power[i_s] = np.mean(p, axis=-1).real

        meta = dict(
            method="delay_and_sum_frequency_domain",
            fs=self.fs,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_len_s=self.frame_len_s,
            hop_s=self.hop_s,
            window=self.window,
            whiten=self.whiten,
            n_frames=int(n_frames),
            n_chan=int(n_chan),
            azimuth_definition="from_north_clockwise_deg",
            azimuth_output="propagation_direction",
            coords="x_east_m,y_north_m",

        )

        return BeamformingResult(
            azimuth_deg=azimuth_deg,
            slowness_s_per_m=slowness_s_per_m,
            power=power,
            freqs_hz=fb,
            meta=meta,
        )

    @staticmethod
    def _frame_1d(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
        """
        对一维信号进行无补零分帧。

        返回：
        - frames: (n_frames, frame_len)
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("x 必须是一维数组")
        n = x.shape[0]
        if n < frame_len:
            raise ValueError("信号长度小于一帧")
        n_frames = 1 + (n - frame_len) // hop
        idx0 = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
        return x[idx0]

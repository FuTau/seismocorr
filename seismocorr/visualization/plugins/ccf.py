# seismocorr/visualization/plugins/ccf.py
from __future__ import annotations

from typing import Any, Optional
import numpy as np

from ..types import Plugin, PlotSpec, Layer, Param


def _build_ccf_wiggle(
    data: Any,
    *,
    title: Optional[str] = None,
    normalize: bool = True,
    clip: Optional[float] = None,
    scale: float = 1.0,
    x_label: str = "Lag (s)",
    y_label: str = "Trace index",

    highlights=None,
    sort=None,

    show_zero_line: bool = False,
) -> PlotSpec:
    """
    CCF wiggle 插件：生成 PlotSpec（后端无关）

    data 约定：
      dict:
        - "cc": (n_tr, n_lags)
        - "lags": (n_lags,)
        - "labels": optional list[str], length n_tr
    """
    if not isinstance(data, dict):
        raise TypeError("ccf.wiggle 当前仅支持 dict 输入：{'cc','lags','labels?'}")

    cc = np.asarray(data["cc"], dtype=float)
    lags = np.asarray(data["lags"], dtype=float)
    labels = data.get("labels")

    if cc.ndim != 2:
        raise ValueError("cc 必须是二维矩阵 (n_traces, n_lags)")
    if lags.ndim != 1 or lags.shape[0] != cc.shape[1]:
        raise ValueError("lags 必须是一维，且长度等于 cc.shape[1]")

    cc_disp = cc.copy()

    # 每道归一化
    if normalize:
        denom = np.max(np.abs(cc_disp), axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cc_disp = cc_disp / denom

    # 裁剪
    if clip is not None:
        cc_disp = np.clip(cc_disp, -abs(clip), abs(clip))

    wiggle = Layer(
        type="wiggle",
        data={
            "x": lags,
            "traces": cc_disp,
            "labels": labels,
            "highlights": highlights,
            "sort": sort,
        },
        style={
            "scale": float(scale),
            "linewidth": 0.8,
            "alpha": 1.0,
        },
        name="CCF",
    )

    layers = [wiggle]

    # lag=0 竖线（可选）
    if show_zero_line:
        v0 = Layer(
            type="vlines",
            data={"xs": [0.0]},
            style={"linewidth": 1.0, "alpha": 0.6},
            name="lag=0",
        )
        layers.append(v0)

    layout = {
        "title": title or "CCF Wiggle",
        "x_label": x_label,
        "y_label": y_label,
    }

    return PlotSpec(plot_id="ccf.wiggle", layers=layers, layout=layout)


PLUGINS = [
    Plugin(
        id="ccf.wiggle",
        title="互相关多道 Wiggle",
        build=_build_ccf_wiggle,
        default_layout={"figsize": (10, 6)},

        data_spec={
            "type": "dict",
            "required_keys": {
                "cc": "2D array, shape (n_tr, n_lags) 互相关矩阵",
                "lags": "1D array, shape (n_lags,) lag/时间轴",
            },
            "optional_keys": {
                "labels": "list[str], length n_tr 道名/台站名（<=60 时显示在 y 轴）",
            },
            "notes": [
                "normalize=True 时会对每道按最大绝对值归一化",
                "clip 可在归一化后做幅值裁剪",
                "highlights 的 trace 索引默认按原始道号（若启用 sort，会自动映射到新顺序）",
            ],
        },

        params={
            "title": Param("str", None, "图标题"),
            "normalize": Param("bool", True, "是否对每道归一化（推荐 True）"),
            "clip": Param("float", None, "幅值裁剪（归一化后），如 0.8"),
            "scale": Param("float", 1.0, "wiggle 幅值缩放系数"),
            "x_label": Param("str", "Lag (s)", "x轴标签"),
            "y_label": Param("str", "Trace index", "y轴标签"),

            # ✅默认不显示 lag=0 中间线
            "show_zero_line": Param("bool", False, "是否显示 lag=0 的竖线（中间线）"),

            "highlights": Param(
                "list[dict]",
                None,
                "高亮配置：对指定道、指定时间段叠加彩色线段（默认红色、默认全时段）。",
                item_schema={
                    "trace": Param("int", None, "道号（0-based，必填）", required=True),
                    "t0": Param("float", None, "起始时间（可选，缺省=全时段）"),
                    "t1": Param("float", None, "终止时间（可选，缺省=全时段）"),
                    "color": Param("str", None, "高亮颜色（默认 red）"),
                    "linewidth": Param("float", None, "高亮线宽（缺省=底线线宽）"),
                    "alpha": Param("float", 1.0, "高亮透明度（默认 1.0）"),
                },
            ),

            "sort": Param(
                "dict",
                None,
                "排序/纵轴控制：{'by': 距离数组(n_tr,), 'ascending': True/False, "
                "'y_mode': 'index'|'distance', 'label': 'Distance (km)'}",
            ),
        },
    )
]

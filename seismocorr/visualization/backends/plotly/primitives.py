# visualization/backends/plotly/primitives.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


def mpl_cmap_to_plotly(cmap: str) -> str:
    """将常见 Matplotlib colormap 名称映射为 Plotly colorscale 名称。

    Plotly 的 colorscale 名称一般是首字母大写格式，因此这里做最小映射，
    以便用户在 style 中沿用常见 mpl cmap 名称（如 "viridis"）。

    Args:
        cmap: Matplotlib colormap 名称。

    Returns:
        Plotly colorscale 名称字符串。
    """
    mapping = {
        "viridis": "Viridis",
        "plasma": "Plasma",
        "inferno": "Inferno",
        "magma": "Magma",
        "cividis": "Cividis",
        "turbo": "Turbo",
        "gray": "Greys",
        "grey": "Greys",
        "Greys": "Greys",
    }
    return mapping.get(str(cmap), "Viridis")


def add_heatmap(
    fig: go.Figure,
    z: Any,
    x: Optional[Any] = None,
    y: Optional[Any] = None,
    *,
    cmap: str = "viridis",
    colorbar_label: str = "",
    name: Optional[str] = None,
) -> None:
    """向 Plotly Figure 添加 heatmap 图层。

    Args:
        fig: Plotly Figure。
        z: 2D 数组，shape (ny, nx)。
        x: 可选横轴坐标，默认使用 0..nx-1。
        y: 可选纵轴坐标，默认使用 0..ny-1。
        cmap: colormap 名称（mpl 风格，会映射到 Plotly colorscale）。
        colorbar_label: 色标标题。
        name: trace 名称（可选）。
    """
    z_arr = np.asarray(z)
    if x is None:
        x = np.arange(z_arr.shape[1])
    if y is None:
        y = np.arange(z_arr.shape[0])

    fig.add_trace(
        go.Heatmap(
            z=z_arr,
            x=x,
            y=y,
            colorscale=mpl_cmap_to_plotly(cmap),
            colorbar=dict(title=str(colorbar_label)) if colorbar_label else None,
            name=name,
            showscale=True,
        )
    )


def add_lines(
    fig: go.Figure,
    x: Any,
    y: Any,
    *,
    linewidth: float = 1.0,
    alpha: float = 1.0,
    name: Optional[str] = None,
) -> None:
    """向 Plotly Figure 添加折线图层。

    Args:
        fig: Plotly Figure。
        x: x 坐标。
        y: y 坐标。
        linewidth: 线宽。
        alpha: 透明度。
        name: trace 名称（可选）。
    """
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(width=float(linewidth)),
            opacity=float(alpha),
            name=name,
        )
    )


def add_wiggle(
    fig: go.Figure,
    x: Any,
    traces: Any,
    *,
    scale: float = 1.0,
    linewidth: float = 0.8,
    alpha: float = 1.0,
    labels: Optional[List[Any]] = None,
    name: Optional[str] = None,
    highlights: Optional[List[Dict[str, Any]]] = None,
    sort: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """向 Plotly Figure 添加 wiggle（多道）图层。

    说明：
        - Plotly 实现为“每道一个 Scatter(lines) trace”，道数过多时会较重。
        - 支持排序 sort（按指标重排 traces / labels，并确保 highlights 仍按原始索引生效）。
        - 支持 highlights：在指定时间窗 [t0, t1] 内对某道进行着色/加粗。

    Args:
        fig: Plotly Figure。
        x: 一维横轴坐标，shape (n_samples,)。
        traces: 二维数据，shape (n_traces, n_samples)。
        scale: 振幅缩放系数。
        linewidth: 线宽。
        alpha: 透明度。
        labels: 可选标签列表（用于 y 轴 ticktext）。
        name: 图例名称（仅在第 1 道显示 legend）。
        highlights: 可选高亮配置 list[dict]。
        sort: 可选排序配置 dict：
            - by: 1D 数组（长度 n_traces），排序依据
            - ascending: bool（默认 True）
            - y_mode: "index" 或 "distance"
            - label: 可选 y 轴标签含义说明（当前未用于 layout）

    Returns:
        dict：用于 fig.update_yaxes(**dict) 的参数片段（如 tickvals/ticktext）。
    """
    x_arr = np.asarray(x)
    tr = np.asarray(traces)
    if tr.ndim != 2:
        raise ValueError("traces 必须是二维数组 (n_traces, n_samples)。")
    n_tr, _ = tr.shape

    # 排序相关
    order = np.arange(n_tr)
    inv_order: Optional[np.ndarray] = None
    y_mode = "index"

    sort_by_sorted: Optional[np.ndarray] = None
    if sort is not None:
        if not isinstance(sort, dict) or "by" not in sort:
            raise ValueError(
                "sort 必须是 dict 且包含 'by'，例如 {'by': dist, 'ascending': True, 'y_mode': 'distance'}。"
            )

        sort_by = np.asarray(sort["by"], dtype=float)
        if sort_by.ndim != 1 or sort_by.shape[0] != n_tr:
            raise ValueError("sort['by'] 必须是一维且长度等于 traces.shape[0]（道数）。")

        ascending = bool(sort.get("ascending", True))
        y_mode = str(sort.get("y_mode", "index"))

        order = np.argsort(sort_by)
        if not ascending:
            order = order[::-1]

        tr = tr[order, :]

        if labels is not None:
            if len(labels) != n_tr:
                raise ValueError("labels 长度必须等于道数 n_tr。")
            labels = [labels[i] for i in order]

        inv_order = np.empty(n_tr, dtype=int)
        inv_order[order] = np.arange(n_tr)
        sort_by_sorted = sort_by[order]

    # 基线 y0：index / distance
    if y_mode == "distance":
        if sort_by_sorted is None:
            raise ValueError("y_mode='distance' 时必须提供 sort={'by': distance, ...}。")
        y0 = sort_by_sorted.astype(float)
    elif y_mode == "index":
        y0 = np.arange(n_tr, dtype=float)
    else:
        raise ValueError("sort['y_mode'] 只能是 'index' 或 'distance'。")

    # highlights：按“原始索引”输入，若发生排序则映射到“新索引”
    hl_by_trace: Dict[int, List[Dict[str, Any]]] = {}
    if highlights:
        if not isinstance(highlights, (list, tuple)):
            raise ValueError("highlights 必须是 list[dict] 或 tuple[dict]。")

        for item in highlights:
            if not isinstance(item, dict) or "trace" not in item:
                raise ValueError("highlights 每项必须是 dict 且包含 'trace' 字段。")

            old_i = int(item["trace"])
            if old_i < 0 or old_i >= n_tr:
                raise ValueError(f"highlights.trace 越界: {old_i} (0~{n_tr - 1})。")

            new_i = int(inv_order[old_i]) if inv_order is not None else old_i
            hl_by_trace.setdefault(new_i, []).append(item)

    # 添加 wiggle 曲线
    for i in range(n_tr):
        y_curve = y0[i] + tr[i] * float(scale)

        fig.add_trace(
            go.Scatter(
                x=x_arr,
                y=y_curve,
                mode="lines",
                line=dict(width=float(linewidth), color="black"),
                opacity=float(alpha),
                name=(name if (name and i == 0) else None),
                showlegend=bool(name and i == 0),
                hoverinfo="skip",
            )
        )

        if i not in hl_by_trace:
            continue

        for cfg in hl_by_trace[i]:
            t0 = float(cfg.get("t0", float(np.min(x_arr))))
            t1 = float(cfg.get("t1", float(np.max(x_arr))))
            if t1 < t0:
                t0, t1 = t1, t0

            mask = (x_arr >= t0) & (x_arr <= t1)
            if not np.any(mask):
                continue

            hl_color = str(cfg.get("color", "red"))
            hl_lw = float(cfg.get("linewidth", linewidth))
            hl_alpha = float(cfg.get("alpha", 1.0))

            fig.add_trace(
                go.Scatter(
                    x=x_arr[mask],
                    y=y_curve[mask],
                    mode="lines",
                    line=dict(width=hl_lw, color=hl_color),
                    opacity=hl_alpha,
                    hoverinfo="skip",
                )
            )

    # y 轴刻度/标签：道数太多时不建议显示
    yaxis: Dict[str, Any] = {}
    if labels is not None and len(labels) == n_tr and n_tr <= 60:
        yaxis.update(
            tickmode="array",
            tickvals=list(y0.astype(float)),  # 注意：distance 模式必须用 y0
            ticktext=[str(s) for s in labels],
        )

    return yaxis

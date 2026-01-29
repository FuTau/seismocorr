# visualization/backends/plotly/render.py
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go

from ...types import FigureHandle, PlotSpec
from . import primitives


def is_available() -> bool:
    """判断 plotly 是否可用（是否安装依赖）。

    Returns:
        True 表示可导入 plotly；False 表示不可用。
    """
    try:
        import plotly  # noqa: F401

        return True
    except Exception:
        return False


def render(spec: PlotSpec) -> FigureHandle:
    """将 PlotSpec 渲染为 Plotly Figure。

    支持的 layer.type：
        - "heatmap"
        - "lines"
        - "wiggle"
        - "vlines"
        - "annotations"
        - "polar_heatmap"

    Args:
        spec: 后端无关的绘图说明书（PlotSpec）。

    Returns:
        FigureHandle：backend="plotly"，handle 为 plotly Figure。

    Raises:
        ValueError: 遇到不支持的 layer.type 或 polar_heatmap 参数不合法。
        KeyError: layer.data 缺少必要字段（如 heatmap 需要 "z" 等）。
    """
    layout = spec.layout or {}
    fig = go.Figure()

    shapes = []
    yaxis_overrides: Dict[str, Any] = {}

    for layer in spec.layers:
        if layer.type == "heatmap":
            primitives.add_heatmap(
                fig,
                layer.data["z"],
                x=layer.data.get("x"),
                y=layer.data.get("y"),
                cmap=layer.style.get("cmap", "viridis"),
                colorbar_label=layer.style.get("colorbar_label", ""),
                name=layer.name,
            )
            continue

        if layer.type == "lines":
            primitives.add_lines(
                fig,
                layer.data["x"],
                layer.data["y"],
                linewidth=layer.style.get("linewidth", 1.0),
                alpha=layer.style.get("alpha", 1.0),
                name=layer.name,
            )
            continue

        if layer.type == "vlines":
            xs = layer.data["xs"]
            shapes.extend(
                primitives.add_vlines(
                    fig,
                    xs,
                    linewidth=layer.style.get("linewidth", 1.0),
                    alpha=layer.style.get("alpha", 1.0),
                    name=layer.name,
                )
            )
            continue

        if layer.type == "wiggle":
            yaxis_part = primitives.add_wiggle(
                fig,
                x=layer.data["x"],
                traces=layer.data["traces"],
                scale=layer.style.get("scale", 1.0),
                linewidth=layer.style.get("linewidth", 0.8),
                alpha=layer.style.get("alpha", 1.0),
                labels=layer.data.get("labels"),
                name=layer.name,
                highlights=layer.data.get("highlights"),
                sort=layer.data.get("sort"),
            )
            yaxis_overrides.update(yaxis_part)
            continue

        if layer.type == "annotations":
            # layer.data 约定：{"texts":[{"x":..,"y":..,"text":".."}, ...]}
            annotations = []
            for item in layer.data.get("texts", []):
                annotations.append(
                    dict(
                        x=float(item["x"]),
                        y=float(item["y"]),
                        text=str(item["text"]),
                        showarrow=False,
                    )
                )
            fig.update_layout(annotations=annotations)
            continue

        if layer.type == "polar_heatmap":
            theta = np.asarray(layer.data["theta"], dtype=float)
            r = np.asarray(layer.data["r"], dtype=float)
            z = np.asarray(layer.data["z"], dtype=float)
            theta_unit = layer.data.get("theta_unit", "deg")

            if theta_unit == "deg":
                theta = np.deg2rad(theta)
            elif theta_unit != "rad":
                raise ValueError("theta_unit 必须是 'deg' 或 'rad'。")

            if z.shape != (r.shape[0], theta.shape[0]):
                raise ValueError("polar_heatmap: z 形状必须是 (n_r, n_theta)。")

            # 生成规则网格（分辨率可调，越大越细但更重）
            nxy = int(layer.style.get("nxy", 400))

            r_max = float(np.max(r))
            x = np.linspace(-r_max, r_max, nxy)
            y = np.linspace(-r_max, r_max, nxy)
            X, Y = np.meshgrid(x, y)
            RR = np.sqrt(X**2 + Y**2)
            TT = (np.arctan2(X, Y) + 2.0 * np.pi) % (2.0 * np.pi)  # 0在N，顺时针

            # RR, TT 映射到 z（简化实现：最近邻）
            r_idx = np.searchsorted(r, RR, side="left")
            r_idx = np.clip(r_idx, 0, len(r) - 1)

            t_idx = np.searchsorted(theta, TT, side="left")
            t_idx = np.clip(t_idx, 0, len(theta) - 1)

            Z = z[r_idx, t_idx]
            Z = np.where(RR <= r_max, Z, np.nan)  # 圆外透明

            colorscale = primitives.mpl_cmap_to_plotly(layer.style.get("cmap", "viridis"))
            cbar_label = layer.style.get("colorbar_label", "")

            fig.add_trace(
                go.Heatmap(
                    z=Z,
                    x=x,
                    y=y,
                    colorscale=colorscale,
                    colorbar=dict(title=cbar_label) if cbar_label else None,
                    zsmooth="best",
                )
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            continue

        raise ValueError(f"plotly 后端不支持的 layer.type: {layer.type!r}。")

    fig.update_layout(
        title=layout.get("title", "") or "",
        xaxis_title=layout.get("x_label", "") or "",
        yaxis_title=layout.get("y_label", "") or "",
        shapes=shapes if shapes else None,
        margin=dict(l=60, r=20, t=60, b=50),
    )

    figsize = layout.get("figsize")
    if figsize:
        width_px = int(float(figsize[0]) * 110.0)
        height_px = int(float(figsize[1]) * 110.0)
        fig.update_layout(width=width_px, height=height_px)

    if yaxis_overrides:
        fig.update_yaxes(**yaxis_overrides)

    return FigureHandle(backend="plotly", handle=fig, spec=spec)

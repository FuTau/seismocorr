# seismocorr/visualization/public.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from .core import PluginRegistry, Visualizer
from .plugins import make_plugins
from .types import BackendName, FigureHandle


_REGISTRY = PluginRegistry(plugins=make_plugins())
_VIS = Visualizer(_REGISTRY)

_DEFAULT_BACKEND: BackendName = "mpl"


def set_default_backend(name: BackendName) -> None:
    """设置全局默认绘图后端。

    Args:
        name: 后端名称（例如 "mpl" 或 "plotly"）。
    """
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = name


def get_default_backend() -> BackendName:
    """获取全局默认绘图后端。"""
    return _DEFAULT_BACKEND


def list_plots() -> List[str]:
    """列出所有可用 plot_id。

    Returns:
        已注册插件 id 列表（通常已排序）。
    """
    return _REGISTRY.list_ids()


def plot(
    plot_id: str,
    data: Any,
    *,
    backend: Optional[BackendName] = None,
    fallback: bool = True,
    **kwargs: Any,
) -> FigureHandle:
    """统一绘图入口。

    内部流程：
        1) 使用 plot_id 找到插件 plugin；
        2) plugin.build(data, **kwargs) -> PlotSpec；
        3) 交给后端渲染 -> FigureHandle。

    Args:
        plot_id: 插件 id，例如 "ccf.wiggle"。
        data: 业务数据（结构由插件 data_spec 或 build() 约定）。
        backend: 指定后端；为 None 则使用全局默认后端（get_default_backend）。
        fallback: 指定后端不可用时，是否自动回退到另一个后端。
        **kwargs: 传递给插件 build 的参数（会在 core 中按 ParamSpec 校验/填默认）。

    Returns:
        FigureHandle：后端图形句柄。

    Raises:
        KeyError: plot_id 未注册。
        RuntimeError: 后端不可用且不允许回退，或回退后仍不可用。
        TypeError/ValueError: kwargs 参数校验失败。
    """
    backend_name = backend or _DEFAULT_BACKEND
    return _VIS.plot(plot_id, data, backend=backend_name, fallback=fallback, **kwargs)


def get_plot_schema(plot_id: str) -> Dict[str, Any]:
    """返回机器可读的插件 schema。

    内容包括：
        - id/title
        - data 输入规范（data_spec）
        - kwargs 参数规范（params）

    Args:
        plot_id: 插件 id。

    Returns:
        schema 字典。

    Raises:
        KeyError: plot_id 未注册。
    """
    plugin = _REGISTRY.get(plot_id)

    schema: Dict[str, Any] = {
        "id": plugin.id,
        "title": plugin.title,
        "data_spec": plugin.data_spec or {},
        "params": {},
    }

    for key, param in (plugin.params or {}).items():
        schema["params"][key] = {
            "type": param.ptype,
            "default": param.default,
            "required": param.required,
            "doc": param.doc,
            "choices": param.choices,
        }

        if param.item_schema:
            schema["params"][key]["item_schema"] = {
                item_key: {
                    "type": item_param.ptype,
                    "default": item_param.default,
                    "required": item_param.required,
                    "doc": item_param.doc,
                    "choices": item_param.choices,
                }
                for item_key, item_param in param.item_schema.items()
            }

    return schema


def help_plot(plot_id: str) -> str:
    """返回人类可读的插件帮助信息。

    包含：
        - plot_id / title
        - data 输入结构说明（data_spec）
        - 可用 kwargs（类型/默认值/说明/choices）
        - 若为 list[dict]，会展开 item_schema 的子字段说明

    Args:
        plot_id: 插件 id。

    Returns:
        帮助说明字符串。

    Raises:
        KeyError: plot_id 未注册。
    """
    plugin = _REGISTRY.get(plot_id)

    lines: List[str] = []
    lines.append(f"Plot ID: {plugin.id}")
    lines.append(f"Title: {plugin.title}")

    if plugin.data_spec:
        ds = plugin.data_spec
        lines.append("")
        lines.append("data 输入格式：")

        if "type" in ds:
            lines.append(f"  type: {ds.get('type')}")

        required_keys = ds.get("required_keys", {}) or {}
        if required_keys:
            lines.append("  必填字段：")
            for key, desc in required_keys.items():
                lines.append(f"    - {key}: {desc}")

        optional_keys = ds.get("optional_keys", {}) or {}
        if optional_keys:
            lines.append("  可选字段：")
            for key, desc in optional_keys.items():
                lines.append(f"    - {key}: {desc}")

        notes = ds.get("notes", []) or []
        if notes:
            lines.append("  备注：")
            for s in notes:
                lines.append(f"    - {s}")

    lines.append("")
    lines.append("可用参数（kwargs）：")
    if not plugin.params:
        lines.append("  （该插件未提供 ParamSpec；请查看 build() 定义/文档。）")
        return "\n".join(lines)

    for key in sorted(plugin.params.keys()):
        param = plugin.params[key]
        default_str = f"default={param.default!r}" if not param.required else "required=True"
        choice_str = f" choices={param.choices!r}" if param.choices else ""
        lines.append(f"  - {key} ({param.ptype}) {default_str}{choice_str}")

        if param.doc:
            lines.append(f"      {param.doc}")

        if param.item_schema:
            lines.append("      子字段（list[dict] 的 dict keys）：")
            for item_key, item_param in param.item_schema.items():
                d2 = f"default={item_param.default!r}" if not item_param.required else "required=True"
                doc = item_param.doc or ""
                lines.append(f"        * {item_key} ({item_param.ptype}) {d2} - {doc}")

    return "\n".join(lines)


def show(fig_handle: FigureHandle) -> None:
    """显示绘制的图形。

    Args:
        fig_handle: FigureHandle，包含 backend 与具体 handle。

    Raises:
        ValueError: 不支持的 backend。
    """
    if fig_handle.backend == "mpl":
        plt.show()
        return

    if fig_handle.backend == "plotly":
        fig_handle.handle.show()
        return

    raise ValueError(f"不支持的 backend: {fig_handle.backend!r}。")

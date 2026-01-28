# seismocorr/visualization/core.py
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .types import BackendName, FigureHandle, Param, ParamSpec, PlotSpec, Plugin


Kwargs = Dict[str, Any]


def _type_check(name: str, value: Any, param: Param) -> None:
    """对插件参数做轻量类型检查与 choices 校验。

    说明：
        - 该检查只覆盖常见基础类型（bool/int/float/str/dict/list/list[dict]）。
        - 其它类型保持宽松，以免过度限制插件自由度。

    Args:
        name: 参数名。
        value: 用户传入的参数值。
        param: 参数定义（包含 ptype / required / default / choices 等字段）。

    Raises:
        TypeError: value 的类型不符合 param.ptype。
        ValueError: value 不在 param.choices 中。
    """
    if value is None:
        return

    ptype = param.ptype
    ok = True

    if ptype == "bool":
        ok = isinstance(value, bool)
    elif ptype == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif ptype == "float":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif ptype == "str":
        ok = isinstance(value, str)
    elif ptype == "dict":
        ok = isinstance(value, dict)
    elif ptype == "list":
        ok = isinstance(value, list)
    elif ptype == "list[dict]":
        ok = isinstance(value, list) and all(isinstance(x, dict) for x in value)
    else:
        ok = True

    if not ok:
        raise TypeError(
            f"参数 {name!r} 类型不符合要求：期望 {ptype}，实际 {type(value).__name__}。"
        )

    if param.choices is not None and value not in param.choices:
        raise ValueError(f"参数 {name!r} 必须是 {param.choices} 之一，实际为 {value!r}。")


def _prepare_kwargs(params: ParamSpec, user_kwargs: Kwargs) -> Kwargs:
    """根据 ParamSpec 合并参数并校验。

    规则：
        1) 检查未知参数；
        2) 合并默认值；
        3) 检查 required；
        4) 进行基础类型检查与 choices 检查。

    Args:
        params: 插件参数规范定义。
        user_kwargs: 用户传入参数。

    Returns:
        合并后的参数字典（包含默认值）。

    Raises:
        TypeError: 发现未知参数或缺少必填参数。
        ValueError: 参数 choices 校验不通过。
    """
    for key in user_kwargs.keys():
        if key not in params:
            raise TypeError(f"未知参数: {key!r}。可用参数: {sorted(params.keys())}")

    merged: Kwargs = {}
    for key, spec in params.items():
        if key in user_kwargs:
            merged[key] = user_kwargs[key]
        else:
            if spec.required:
                raise TypeError(f"缺少必填参数: {key!r}")
            merged[key] = spec.default

        _type_check(key, merged[key], spec)

    return merged


class PluginRegistry:
    """插件注册表（PlotSpec 生成器管理）。

    插件以 plugin.id 作为唯一键。注册表提供：
        - add_plugin(): 注册插件（检测重复 id）
        - get(): 按 id 获取插件
        - list_ids(): 列出所有已注册 id
    """

    def __init__(self, plugins: Optional[List[Plugin]] = None) -> None:
        """初始化注册表。

        Args:
            plugins: 可选的初始插件列表；为 None 时表示空注册表。
        """
        self._plugins: Dict[str, Plugin] = {}
        if plugins:
            for plugin in plugins:
                self.add_plugin(plugin)

    def add_plugin(self, plugin: Plugin) -> None:
        """注册一个插件。

        Args:
            plugin: 待注册插件对象。

        Raises:
            ValueError: plugin.id 已存在。
        """
        if plugin.id in self._plugins:
            raise ValueError(f"重复的 plugin id: {plugin.id!r}。请确保插件 id 唯一。")
        self._plugins[plugin.id] = plugin

    def get(self, plot_id: str) -> Plugin:
        """根据 plot_id 获取插件。

        Args:
            plot_id: 插件 id（如 "ccf.wiggle"）。

        Returns:
            对应的插件对象。

        Raises:
            KeyError: 未找到对应 id。
        """
        if plot_id not in self._plugins:
            raise KeyError(f"未找到插件: {plot_id!r}。已注册: {sorted(self._plugins.keys())}")
        return self._plugins[plot_id]

    def list_ids(self) -> List[str]:
        """列出所有插件 id（排序后）。"""
        return sorted(self._plugins.keys())


@dataclass
class Backend:
    """绘图后端对象：用于把 PlotSpec 渲染为 FigureHandle。

    Attributes:
        name: 后端名称（如 "mpl" / "plotly"）。
        available: 后端是否可用（依赖是否安装）。
        render: 渲染函数 render(spec) -> FigureHandle。
    """

    name: BackendName
    available: bool
    render: Any  # Callable[[PlotSpec], FigureHandle]


def _load_backend(name: BackendName) -> Backend:
    """动态加载后端模块（相对本包，不写死顶层名字）。

    约定的导入路径：
        - seismocorr.visualization.backends.mpl.render
        - seismocorr.visualization.backends.plotly.render

    Args:
        name: 后端名称。

    Returns:
        Backend 对象。

    Raises:
        ValueError: name 不在支持列表中。
    """
    base = __package__  # "seismocorr.visualization"

    if name == "mpl":
        mod = importlib.import_module(f"{base}.backends.mpl.render")
        return Backend(name="mpl", available=mod.is_available(), render=mod.render)

    if name == "plotly":
        mod = importlib.import_module(f"{base}.backends.plotly.render")
        return Backend(name="plotly", available=mod.is_available(), render=mod.render)

    raise ValueError(f"未知后端: {name!r}。支持: 'mpl' / 'plotly'。")


class Visualizer:
    """统一绘图入口：插件生成 PlotSpec，后端负责渲染。

    设计：
        1) plugin.build(data, **kwargs) -> PlotSpec（后端无关）
        2) backend.render(spec) -> FigureHandle（后端相关）
    """

    def __init__(self, registry: PluginRegistry, default_backend: BackendName = "mpl") -> None:
        """初始化 Visualizer。

        Args:
            registry: 插件注册表。
            default_backend: 默认后端名称（未显式指定 backend 时使用）。
        """
        self.registry = registry
        self._default_backend: BackendName = default_backend

    def set_default_backend(self, backend: BackendName) -> None:
        """设置默认后端。"""
        self._default_backend = backend

    def get_default_backend(self) -> BackendName:
        """获取默认后端。"""
        return self._default_backend

    def plot(
        self,
        plugin_id: str,
        data: Any,
        backend: Optional[BackendName] = None,
        *,
        fallback: bool = True,
        **kwargs: Any,
    ) -> FigureHandle:
        """统一绘图函数。

        Args:
            plugin_id: 插件 id，如 "ccf.wiggle"。
            data: 业务数据对象（可以是 ndarray / dict / 自定义类）。
            backend: 指定后端（mpl/plotly）；不传则使用默认后端。
            fallback: 指定后端不可用时，是否自动回退到另一个后端。
            **kwargs: 传给插件 build() 的参数（如 title/cmap/normalize...）。

        Returns:
            FigureHandle：后端返回的图对象句柄。

        Raises:
            KeyError: 找不到 plugin_id。
            RuntimeError: 后端不可用且 fallback=False，或回退后仍不可用。
            ValueError/TypeError: 插件参数校验失败或后端名称非法。
        """
        plugin = self.registry.get(plugin_id)

        prepared = _prepare_kwargs(plugin.params, kwargs) if plugin.params else kwargs
        spec: PlotSpec = plugin.build(data, **prepared)

        merged_layout = dict(plugin.default_layout)
        merged_layout.update(spec.layout or {})
        spec.layout = merged_layout

        backend_name = backend or self._default_backend
        backend_obj = _load_backend(backend_name)

        if not backend_obj.available:
            if not fallback:
                raise RuntimeError(
                    f"后端 {backend_name!r} 不可用（可能未安装依赖），且 fallback=False。"
                )

            alt_name: BackendName = "plotly" if backend_name == "mpl" else "mpl"
            alt_obj = _load_backend(alt_name)
            if not alt_obj.available:
                raise RuntimeError(
                    f"后端 {backend_name!r} 不可用，且回退后端 {alt_name!r} 也不可用。"
                    "请安装对应依赖。"
                )

            fig = alt_obj.render(spec)
            setattr(fig, "spec", spec)
            return fig

        fig = backend_obj.render(spec)
        setattr(fig, "spec", spec)
        return fig

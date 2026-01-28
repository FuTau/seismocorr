# seismocorr/visualization/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, TypeAlias


LayerType: TypeAlias = Literal[
    "heatmap",
    "lines",
    "wiggle",
    "vlines",
    "annotations",
    "polar_heatmap",
]

BackendName: TypeAlias = Literal["mpl", "plotly"]

JSONDict: TypeAlias = Dict[str, Any]


@dataclass
class Layer:
    """后端无关的“图层”描述。

    一张图由多个 Layer 叠加组成。各后端只需要理解 Layer.type / data / style，
    即可把它渲染成具体图形对象。

    Attributes:
        type: 图层类型，如 "wiggle" / "heatmap" / "vlines" 等。
        data: 绘图数据（由插件写入，后端读取）。
        style: 样式参数（由插件写入，后端读取）。
        name: 可选名称，用于图例/识别。
    """

    type: LayerType
    data: JSONDict = field(default_factory=dict)
    style: JSONDict = field(default_factory=dict)
    name: Optional[str] = None


@dataclass
class PlotSpec:
    """后端无关的“绘图说明书”。

    插件负责把业务数据 + kwargs 转换为 PlotSpec；后端负责把 PlotSpec 渲染为图形对象。

    Attributes:
        plot_id: 插件 id（便于调试/追踪来源）。
        layers: 图层列表（叠加渲染）。
        layout: 标题、坐标轴标签、画布大小等通用布局参数（后端尽量理解）。
    """

    plot_id: str
    layers: List[Layer]
    layout: JSONDict = field(default_factory=dict)


@dataclass
class FigureHandle:
    """统一返回的图形句柄。

    Attributes:
        backend: 使用的后端名称（"mpl" 或 "plotly"）。
        handle: 后端原生对象（matplotlib Figure / plotly Figure 等）。
        spec: 可选，保留 PlotSpec 便于调试/再渲染。
        extra: 可选扩展字段，用于存放后端额外信息或渲染副产物。
    """

    backend: BackendName
    handle: Any
    spec: Optional[PlotSpec] = None
    extra: JSONDict = field(default_factory=dict)


@dataclass
class Param:
    """参数元信息（机器可读）。

    字段说明：
        - ptype: 参数类型（字符串形式，便于打印和轻量检查）
          常用：'bool'/'int'/'float'/'str'/'dict'/'list'/'list[dict]'
        - default: 默认值（required=False 且用户未提供则使用）
        - doc: 中文说明
        - required: 是否必填
        - choices: 枚举值（可选）
        - item_schema: 当 ptype 为 "list[dict]" 时，用于描述 dict 的子字段（例如 highlights）

    Attributes:
        ptype: 参数类型字符串。
        default: 默认值。
        doc: 参数说明文本。
        required: 是否必填。
        choices: 可选枚举列表。
        item_schema: list[dict] 的 dict 子字段 schema。
    """

    ptype: str
    default: Any = None
    doc: str = ""
    required: bool = False
    choices: Optional[List[Any]] = None
    item_schema: Optional[Dict[str, "Param"]] = None


ParamSpec: TypeAlias = Dict[str, Param]


@dataclass
class Plugin:
    """插件定义：业务数据 + kwargs -> PlotSpec（后端无关）。

    重要字段：
        - id: 图的唯一标识（用户 plot() 用）
        - title: 展示名称（help/list 用）
        - build: Callable(data, **kwargs) -> PlotSpec
        - params: ParamSpec（用于 help/默认值/校验）
        - data_spec: data 输入规范（只用于文档/帮助，不参与 kwargs 校验）

    Attributes:
        id: 插件唯一 id。
        title: 插件显示名称。
        build: 插件构建函数，返回 PlotSpec。
        default_layout: 插件默认布局（会与 PlotSpec.layout 合并）。
        params: 插件 kwargs 参数规范（ParamSpec）。
        data_spec: data 输入规范描述（机器/人类可读）。
    """

    id: str
    title: str
    build: Callable[..., PlotSpec]

    default_layout: JSONDict = field(default_factory=dict)
    params: ParamSpec = field(default_factory=dict)
    data_spec: JSONDict = field(default_factory=dict)

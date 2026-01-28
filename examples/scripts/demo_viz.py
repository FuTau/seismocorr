from __future__ import annotations

from typing import Any, Dict

import numpy as np

from seismocorr.visualization import help_plot, plot, set_default_backend, show


def generate_ccf_data(n_tr: int = 40, n_lags: int = 401, *, seed: int = 0) -> Dict[str, Any]:
    """生成模拟 CCF 数据。

    Args:
        n_tr: 道数（traces 数量）。
        n_lags: lag 采样点数。
        seed: 随机种子，用于可复现。

    Returns:
        dict，包含：
            - cc: (n_tr, n_lags) 的 CCF 矩阵
            - lags: (n_lags,) 的 lag 轴
            - labels: (n_tr,) 的标签列表
            - dist_km: (n_tr,) 的距离数组（用于排序示例）
    """
    rng = np.random.default_rng(seed)

    lags = np.linspace(-20.0, 20.0, int(n_lags))
    cc = 0.08 * rng.standard_normal((int(n_tr), int(n_lags)))

    center = int(n_lags) // 2
    for i in range(int(n_tr)):
        shift = int(round((i - n_tr / 2.0) * 0.5))
        idx = int(np.clip(center + shift, 0, int(n_lags) - 1))
        cc[i, idx] += 1.0

    labels = [f"STA{i:02d}" for i in range(int(n_tr))]
    dist_km = rng.uniform(10.0, 300.0, size=int(n_tr))

    return {"cc": cc, "lags": lags, "labels": labels, "dist_km": dist_km}


def generate_beamforming_data(
    n_azimuth: int = 100,
    n_radius: int = 50,
    *,
    seed: int = 1,
) -> Dict[str, Any]:
    """生成模拟波束形成功率矩阵数据。

    Args:
        n_azimuth: 方位角采样数。
        n_radius: 慢度采样数（极坐标半径方向）。
        seed: 随机种子，用于可复现。

    Returns:
        dict，包含：
            - power: (n_radius, n_azimuth) 波束功率矩阵
            - azimuth_deg: (n_azimuth,) 方位角数组（0..360）
            - slowness_s_per_m: (n_radius,) 慢度数组
    """
    rng = np.random.default_rng(seed)

    azimuth_deg = np.linspace(0.0, 360.0, int(n_azimuth))
    slowness_s_per_m = np.linspace(0.1, 1.0, int(n_radius))

    # 原始生成是 (n_azimuth, n_radius)，插件约定这里用 (n_radius, n_azimuth)
    power = rng.random((int(n_azimuth), int(n_radius))).T

    return {
        "power": power,
        "azimuth_deg": azimuth_deg,
        "slowness_s_per_m": slowness_s_per_m,
    }


def test_ccf_wiggle_plot() -> None:
    """测试并绘制 CCF wiggle 图。"""
    set_default_backend("mpl")

    ccf = generate_ccf_data()

    fig = plot(
        "ccf.wiggle",
        data={"cc": ccf["cc"], "lags": ccf["lags"], "labels": ccf["labels"]},
        normalize=True,
        clip=0.9,
        sort={
            "by": ccf["dist_km"],
            "ascending": True,
            "y_mode": "index",
            "label": "Distance (km)",
        },
        highlights=[
            {"trace": 5, "t0": -2.0, "t1": 2.0},
            {"trace": 10, "color": "blue"},
            {"trace": 12, "t0": 1.0, "t1": 3.5, "color": "#ff00ff", "linewidth": 3},
        ],
        scale=5,
    )

    show(fig)
    print(help_plot("ccf.wiggle"))


def test_beamforming_polar_heatmap() -> None:
    """测试并绘制波束形成极坐标热力图。"""
    data = generate_beamforming_data()

    fig = plot("beamforming.polar_heatmap", data)
    # 如需指定 plotly 后端：
    # fig = plot("beamforming.polar_heatmap", data, backend="plotly")

    show(fig)
    print(help_plot("beamforming.polar_heatmap"))


def main() -> None:
    """运行示例测试。"""
    test_ccf_wiggle_plot()
    test_beamforming_polar_heatmap()


if __name__ == "__main__":
    main()

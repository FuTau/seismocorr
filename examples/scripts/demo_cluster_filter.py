import numpy as np
import matplotlib.pyplot as plt
from seismocorr.plugins.processing.cluster_filter import ClusterFilter

def generate_data(fs, duration_s, n_samples, n_lags):
    """
    生成一个简单的模拟数据：带噪声的 Cross-Correlation Functions (CCFs)。
    """
    n = int(round(duration_s * fs))  # 计算样本总数
    t = np.arange(n) / fs  # 时间向量
    
    # 为每个样本生成带噪声的正弦波 CCF 数据
    data = np.empty((n_samples, n_lags))  # 初始化数据数组
    for i in range(n_samples):
        data[i, :] = np.sin(2 * np.pi * 0.5 * t[:n_lags]) + 0.1 * np.random.randn(n_lags)  # 每个样本是一个正弦波加噪声

    return t[:n_lags], data  # 返回时间和 CCF 数据

def main():

    fs = 50.0  # 采样频率 (Hz)
    duration_s = 20.0  # 数据时长 20 秒
    n_samples = 10  # 10 个样本
    n_lags = 100  # 每个样本 100 个延时点

    t, data = generate_data(fs, duration_s, n_samples, n_lags)

    cluster_filter = ClusterFilter(
        lag_window=(-1.5, 1.5),  # 设置延迟窗口范围
        emd_weight=0.7,          # EMD 的权重
        energy_weight=0.3,       # Energy distance 的权重
        select_percentile=(0.3, 0.7)  # 筛选的百分比范围
    )
    #用法
    out_lags, out_ccfs= cluster_filter.filter(lags=t, ccfs=data)

    ccf_list = cluster_filter.filter_to_list(lags=t, ccfs=data)

    cluster_filter.fit(lags=t, ccfs=data)
    out_lags, out_ccfs = cluster_filter.transform()
    #可视化
    plt.figure(figsize=(10, 6))
    plt.imshow(
        out_ccfs,
        aspect="auto",
        origin="lower",
        extent=[out_lags[0], out_lags[-1], 0, len(out_ccfs)],
    )
    plt.xlabel("Lag (s)")
    plt.ylabel("Sample Index")
    plt.title("Filtered Cross-Correlation Functions (CCFs)")
    plt.colorbar(label="Amplitude")
    plt.show()

if __name__ == "__main__":
    main()

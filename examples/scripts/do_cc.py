import h5py
from obspy import UTCDateTime
import numpy as np
from seismocorr.core.correlation.correlation import batch_cross_correlation,compute_cross_correlation
import matplotlib.pyplot as plt

def read_zdh5(h5_path):
    meta = {}
    with h5py.File(h5_path, "r") as fp:
        data = fp["Acquisition/Raw[0]/RawData"][...].T
        dt = 1 / fp['Acquisition/Raw[0]'].attrs['OutputDataRate']
        dx = fp['Acquisition'].attrs['SpatialSamplingInterval']
        begin_time = str(fp['Acquisition'].attrs['MeasurementStartTime'])[2:-15]
        meta['starttime'] = begin_time+'0'
        meta['lamb'] = fp['Acquisition'].attrs['PulseWidth']
        begin_time = UTCDateTime(begin_time)
        gl = fp['Acquisition'].attrs['GaugeLength']  # m
        n = 1.45
        lamd = 3e8 / n / fp['Acquisition'].attrs['PulseWidth'] * 1e-9
        eta = 0.78
        factor = 4.0 * np.pi * eta * n * gl / lamd
        radconv = 10430.378850470453
        DAS_data = data / factor / radconv * 1e6
        nx, nt = data.shape
        x = np.arange(nx) * dx
        t = np.arange(nt) * dt
    meta['dx'] = dx
    meta['fs'] = 1/dt
    meta['GaugeLength'] = gl
    meta['nch'] = data.shape[0]
    meta['time_length'] = data.shape[1]*dt
    return DAS_data, x, t, begin_time, meta

def cross_correlation_with_reference_channel(DAS_data, reference_index, sampling_rate, 
                                          channel_range=None, **correlation_kwargs):
    """
    计算指定参考道与其他道的互相关
    
    Parameters:
    -----------
    DAS_data : numpy.ndarray
        2D数组，形状为(nch, nt)，表示多道地震数据
    reference_index : int
        参考道的索引（从0开始）
    sampling_rate : float
        采样率 (Hz)
    channel_range : tuple or None
        要计算的其他道的范围，如(0, 100)表示前100道，None表示所有道
    **correlation_kwargs : dict
        传递给batch_cross_correlation的参数
        
    Returns:
    --------
    results : dict
        互相关结果，键为"ref_ch{ref_idx}--ch{ch_idx}"，值为(lags, ccf)
    """
    # 检查输入数据
    nch, nt = DAS_data.shape
    if reference_index < 0 or reference_index >= nch:
        raise ValueError(f"参考道索引{reference_index}超出范围[0, {nch-1}]")
    
    # 确定要计算的道范围
    if channel_range is None:
        channel_indices = range(nch)
    else:
        start, end = channel_range
        channel_indices = range(max(0, start), min(nch, end))
    
    # 创建道字典
    traces = {}
    for i in channel_indices:
        traces[f'ch{i:04d}'] = DAS_data[i, :]
    
    # 创建道对列表：参考道与每个其他道的组合
    ref_channel = f'ch{reference_index:04d}'
    pairs = [(ref_channel, f'ch{i:04d}') for i in channel_indices]
    print(channel_indices)
    # 计算互相关
    results = batch_cross_correlation(
        traces=traces,
        pairs=pairs,
        sampling_rate=sampling_rate,
        **correlation_kwargs
    )
    
    return results

def plot_cross_correlation_results(results, reference_index, dx, max_lag=None):
    """
    绘制互相关结果
    
    Parameters:
    -----------
    results : dict
        互相关结果字典
    reference_index : int
        参考道索引
    dx : float
        道间距
    max_lag : float or None
        要显示的滞后时间范围
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 提取距离和互相关数据
    distances = []
    ccfs = []
    lags = None
    
    for key, (lags_arr, ccf_arr) in results.items():
        # 解析道索引
        ch_idx = int(key.split('--')[1][2:])
        distance = abs(ch_idx - reference_index) * dx
        distances.append(distance)
        ccfs.append(ccf_arr)
        
        if lags is None:
            lags = lags_arr
    
    # 按距离排序
    sorted_indices = np.argsort(distances)
    distances = np.array(distances)[sorted_indices]
    ccfs = np.array(ccfs)[sorted_indices]
    
    # 确定滞后时间范围
    if max_lag is not None:
        lag_mask = np.abs(lags) <= max_lag
        lags_plot = lags[lag_mask]
        ccfs_plot = ccfs[:, lag_mask]
    else:
        lags_plot = lags
        ccfs_plot = ccfs
    
    # 绘制互相关矩阵
    normalize = lambda x: (x) / np.max(np.abs(x), axis=-1, keepdims=True)
    im = ax1.imshow(normalize(ccfs_plot), aspect='auto', origin='lower',
                   extent=[lags_plot[0], lags_plot[-1], distances[-1], distances[0]],
                   cmap='RdBu_r', vmin=-0.8, vmax=0.8)
    ax1.set_xlabel('Lag Time (s)')
    ax1.set_ylabel('Distance from Reference (m)')
    ax1.set_title(f'Cross-correlation with Reference Channel {reference_index}')
    plt.colorbar(im, ax=ax1, label='Cross-correlation')
    
    # 绘制几个典型距离的互相关曲线
    
    for i in range(0,len(distances),10):
        ax2.plot(lags_plot, ccfs_plot[i]/np.max(np.abs(ccfs_plot[i])) + distances[i], 
                label=f'{distances[i]:.1f}m', linewidth=1,color='black')
    
    ax2.set_xlabel('Lag Time (s)')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Cross-correlation')
    ax2.set_xlim(lags_plot[0], lags_plot[-1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from seismocorr.utils.io import scan_h5_files
    
    # 扫描H5文件
    files = scan_h5_files('../2024091912', pattern="*.h5")
    
    if len(files) == 0:
        print("未找到H5文件")
    else:
        # 读取第一个文件
        DAS_data, x, t, begin_time, meta = read_zdh5(files[0])
        print(f"数据形状: {DAS_data.shape}")
        print(f"采样率: {meta['fs']} Hz")
        print(f"道间距: {meta['dx']} m")
        
        # 设置互相关参数
        correlation_params = {
            'method': 'freq-domain',           # 使用频域方法
            'time_normalize': 'one-bit',       # 一位归一化
            'freq_normalize': 'no',            # 频域不归一化
            'freq_band': (1, 20),              # 带通滤波 1-20 Hz
            'max_lag': 2.0,                   # 最大滞后10秒
            'nfft': None,                       # 自动选择FFT长度
            'sampling_rate': meta['fs'],
            'time_norm_kwargs': {'fmin':1,'Fs':meta['fs'],'norm_win':0.5},
            'freq_norm_kwargs': {'smooth_win':20}
        }
        
        # 选择参考道（例如中间道）
        reference_channel = 100
        print(f"使用第{reference_channel}道作为参考道")
        
        # 为了演示，只计算前100道（避免计算量过大）
        channel_range = (0, DAS_data.shape[0])
        
        # 计算互相关
        results = cross_correlation_with_reference_channel(
            DAS_data=DAS_data,
            reference_index=reference_channel,
            channel_range=channel_range,
            **correlation_params
        )
        print(results)
        print(f"成功计算了{len(results)}个道对的互相关")
        
        # 绘制结果
        plot_cross_correlation_results(
            results=results,
            reference_index=reference_channel,
            dx=meta['dx'],
            max_lag=2.0  # 只显示±5秒的滞后
        )
    
        # 可选：保存结果
        output_file = f"cross_correlation_ref{reference_channel}.npz"
        np.savez(output_file, 
                results=results, 
                reference_index=reference_channel,
                sampling_rate=meta['fs'],
                dx=meta['dx'])
        print(f"结果已保存到: {output_file}")
# seismocorr/utils/io.py

import glob
from pathlib import Path
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import h5py

# 支持的时间格式定义（可扩展）
_TIMESTAMP_PATTERNS = [
    # 示例：data_20240101_1200.h5
    {
        "regex": r"(\d{8})_(\d{4})",
        "format": "%Y%m%d_%H%M"
    },
    # 示例：2024-01-01T12:00:00Z.h5
    {
        "regex": r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", 
        "format": "%Y-%m-%dT%H:%M:%S"
    },
    # 示例：chunk_2024_01_01_12_00.h5
    {
        "regex": r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})",
        "format": "%Y_%m_%d_%H_%M"
    },
    # 示例：signal_202401011200.h5 （无分隔符）
    {
        "regex": r"(\d{12})", 
        "format": "%Y%m%d%H%M"
    }
]

def _parse_filename_timestamp(filename: str) -> Optional[datetime]:
    """
    从文件名中提取时间戳，返回 datetime 对象（本地时间，无 tz）
    """
    name = Path(filename).stem  # 去掉 .h5
    for pattern in _TIMESTAMP_PATTERNS:
        match = re.search(pattern["regex"], name)
        if match:
            try:
                dt_str = "".join(match.groups())
                # 移除分隔符以便统一解析
                clean_str = re.sub(r'[^0-9]', '', dt_str)[:12]  # 取前12位：YYYYMMDDHHMM
                fmt = "%Y%m%d%H%M"
                return datetime.strptime(clean_str, fmt)
            except Exception:
                continue
    return None


def scan_h5_files(
    directory: str,
    pattern: str = "*.h5",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[str]:
    """
    扫描目录下的 H5 文件，按时间戳排序并可选时间段裁剪

    Args:
        directory: 目录路径
        pattern: glob 模式，默认 "*.h5"
        start_time: 起始时间 (datetime object)
        end_time: 结束时间 (datetime object)

    Returns:
        排好序且在时间范围内的文件路径列表
    """
    search_path = str(Path(directory) / pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"Warning: No files found in {search_path}")
        return []

    # 提取时间戳并关联文件
    file_times = []
    failed_files = []

    for f in files:
        dt = _parse_filename_timestamp(f)
        if dt is not None:
            file_times.append((f, dt))
        else:
            failed_files.append(f)

    if failed_files:
        print(f"Warning: Could not parse timestamp from {len(failed_files)} files (using filename sort)")
        # 如果部分失败，仅对成功者排序
        sorted_files = [f for f, _ in sorted(file_times, key=lambda x: x[1])]
        # 将无法解析的文件放在末尾（按名字排序）
        failed_sorted = sorted(failed_files)
        return sorted_files + failed_sorted

    # 全部成功解析，直接排序
    sorted_files = [f for f, dt in sorted(file_times, key=lambda x: x[1])]

    # 时间裁剪
    if start_time or end_time:
        filtered = []
        for f, dt in file_times:
            if start_time and dt < start_time:
                continue
            if end_time and dt >= end_time:  # 注意：[start, end)
                continue
            filtered.append(f)
        return filtered

    return sorted_files

def read_zdh5(h5_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, datetime, Dict[str, Any]]:
    """
    读取ZDH5格式的DAS数据文件
    
    Args:
        h5_path: ZDH5文件路径
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, datetime, Dict[str, Any]]:
            - DAS_data: 处理后的DAS数据，形状为(nx, nt)
            - x: 空间坐标数组 (m)
            - t: 时间坐标数组 (s)
            - begin_time: 数据起始时间
            - meta: 元数据字典，包含采样率、道间距等信息
            
    Raises:
        FileNotFoundError: 当指定的文件不存在时
        KeyError: 当文件缺少必要的属性或数据集时
        IOError: 当文件读取过程中出现错误时
    """
    meta: Dict[str, Any] = {}
    
    with h5py.File(h5_path, "r") as fp:
        # 读取原始数据
        data = fp["Acquisition/Raw[0]/RawData"][...].T
        
        # 计算时间采样间隔和空间采样间隔
        dt = 1 / fp['Acquisition/Raw[0]'].attrs['OutputDataRate']
        dx = fp['Acquisition'].attrs['SpatialSamplingInterval']
        
        # 解析起始时间
        begin_time_str = str(fp['Acquisition'].attrs['MeasurementStartTime'])[2:-15]
        meta['starttime'] = begin_time_str + '0'
        
        # 读取其他元数据
        meta['lamb'] = fp['Acquisition'].attrs['PulseWidth']
        
        # 解析时间字符串
        # 支持ISO格式，如：YYYY-MM-DDTHH:MM:SS.ssssssZ
        if begin_time_str.endswith('Z'):
            begin_time_str = begin_time_str[:-1]
        
        try:
            begin_time = datetime.fromisoformat(begin_time_str)
        except ValueError:
            # 如果解析失败，使用当前时间
            begin_time = datetime.now()
        
        # 计算转换因子
        gl = fp['Acquisition'].attrs['GaugeLength']  # m
        n = 1.45  # 光纤折射率
        lamd = 3e8 / n / fp['Acquisition'].attrs['PulseWidth'] * 1e-9  # 波长计算
        eta = 0.78  # 耦合效率
        factor = 4.0 * np.pi * eta * n * gl / lamd  # 转换因子
        radconv = 10430.378850470453  # 弧度转换系数
        
        # 数据转换
        DAS_data = data / factor / radconv * 1e6  # 转换为应变率
        
        # 生成坐标数组
        nx, nt = data.shape
        x = np.arange(nx) * dx
        t = np.arange(nt) * dt
    
    # 完善元数据
    meta['dx'] = dx
    meta['fs'] = 1 / dt
    meta['GaugeLength'] = gl
    meta['nch'] = data.shape[0]
    meta['time_length'] = data.shape[1] * dt
    
    return DAS_data, x, t, begin_time, meta
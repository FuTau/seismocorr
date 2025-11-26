# seismocorr/utils/io.py

import glob
from pathlib import Path
import re
from datetime import datetime, timezone
from typing import List, Optional

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

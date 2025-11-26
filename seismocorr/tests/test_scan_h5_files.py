# tests/test_scan_h5_files.py
from pathlib import Path
import unittest
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.io import scan_h5_files

class TestScanH5Files(unittest.TestCase):

    def setUp(self):
        self.test_dir = "./2024091912"
        if not os.path.exists(self.test_dir):
            raise FileNotFoundError(f"Test directory '{self.test_dir}' not found. "
                                  "Please run generate_test_files.py first.")

    def test_basic_scan_and_sort(self):
        """测试基本扫描和按时间排序"""
        files = scan_h5_files(self.test_dir, pattern="*.h5")
        self.assertGreater(len(files), 1, "Should find multiple files")

        # 检查是否已排序
        timestamps = []
        for f in files:
            basename = os.path.basename(f)
            self.assertIn(".h5", basename)
            match = f"FDLY_916m_4m_1m_1000Hz_1000Hz_UTC8_(\\d{{12}}).h5"
            m = os.path.splitext(basename)[0].split('_')[-1]
            self.assertEqual(len(m), 12, f"Timestamp part should be 12 digits: {m}")
            dt = datetime.strptime(m, "%Y%m%d%H%M")
            timestamps.append(dt)

        # 验证时间递增
        self.assertEqual(timestamps, sorted(timestamps), "Files should be sorted by time")

    def test_time_window_filtering(self):
        """测试时间窗口过滤功能"""
        start = datetime(2024, 9, 19, 12, 30)
        end = datetime(2024, 9, 19, 12, 50)

        files = scan_h5_files(self.test_dir, start_time=start, end_time=end)
        
        self.assertGreater(len(files), 0, "Should find files in time window")

        for f in files:
            m = os.path.splitext(os.path.basename(f))[0].split('_')[-1]
            dt = datetime.strptime(m, "%Y%m%d%H%M")
            self.assertGreaterEqual(dt, start, f"{dt} < {start}")
            self.assertLess(dt, end, f"{dt} >= {end}")

        # 额外检查：不应包含 12:00 或 13:30 之后的
        first_file = os.path.splitext(os.path.basename(files[0]))[0].split('_')[-1]
        first_dt = datetime.strptime(first_file, "%Y%m%d%H%M")
        self.assertGreaterEqual(first_dt.minute, 30 if first_dt.hour == 12 else 0)

    def test_pattern_filtering(self):
        """测试 glob 模式匹配"""
        # 假设有些文件是 .tmp.h5
        extra_file = os.path.join(self.test_dir, "extra.tmp")
        with open(extra_file, 'w') as f:
            f.write("dummy")

        files_all = scan_h5_files(self.test_dir, pattern="*")
        self.assertTrue(any("extra.tmp" in f for f in files_all))

        files_h5 = scan_h5_files(self.test_dir, pattern="*.h5")
        self.assertFalse(any("extra.tmp" in f for f in files_h5), "Should not include .tmp.h5 when *.h5")

        # 清理
        os.remove(extra_file)

    def test_empty_directory(self):
        """测试空目录"""
        temp_dir = "test_empty"
        Path(temp_dir).mkdir(exist_ok=True)
        files = scan_h5_files(temp_dir)
        self.assertEqual(len(files), 0)
        Path(temp_dir).rmdir()


if __name__ == "__main__":
    unittest.main()

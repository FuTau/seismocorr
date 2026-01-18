# tests/test_scan_h5_files.py
from pathlib import Path
import unittest
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from seismocorr.utils.io import scan_h5_files

class TestScanH5Files(unittest.TestCase):

    def setUp(self):
        # 创建临时测试目录
        self.test_dir = "./test_scan_h5_temp"
        Path(self.test_dir).mkdir(exist_ok=True)
        
        # 创建模拟H5文件（仅文件名，不需要实际内容）
        self.test_files = [
            "FDLY_916m_4m_1m_1000Hz_1000Hz_UTC8_202409191200.h5",
            "FDLY_916m_4m_1m_1000Hz_1000Hz_UTC8_202409191205.h5",
            "FDLY_916m_4m_1m_1000Hz_1000Hz_UTC8_202409191210.h5",
            "FDLY_916m_4m_1m_1000Hz_1000Hz_UTC8_202409191215.h5",
            "FDLY_916m_4m_1m_1000Hz_1000Hz_UTC8_202409191220.h5"
        ]
        
        # 写入空文件
        for f in self.test_files:
            file_path = os.path.join(self.test_dir, f)
            with open(file_path, 'w') as fp:
                fp.write("dummy content")

    def tearDown(self):
        # 清理临时文件和目录
        if os.path.exists(self.test_dir):
            for f in os.listdir(self.test_dir):
                file_path = os.path.join(self.test_dir, f)
                os.remove(file_path)
            os.rmdir(self.test_dir)

    def test_basic_scan_and_sort(self):
        """测试基本扫描和按时间排序"""
        files = scan_h5_files(self.test_dir, pattern="*.h5")
        self.assertEqual(len(files), len(self.test_files), f"Should find {len(self.test_files)} files")

        # 检查是否已排序
        timestamps = []
        for f in files:
            basename = os.path.basename(f)
            self.assertIn(".h5", basename)
            m = os.path.splitext(basename)[0].split('_')[-1]
            self.assertEqual(len(m), 12, f"Timestamp part should be 12 digits: {m}")
            dt = datetime.strptime(m, "%Y%m%d%H%M")
            timestamps.append(dt)

        # 验证时间递增
        self.assertEqual(timestamps, sorted(timestamps), "Files should be sorted by time")

    def test_time_window_filtering(self):
        """测试时间窗口过滤功能"""
        start = datetime(2024, 9, 19, 12, 5)
        end = datetime(2024, 9, 19, 12, 15)

        files = scan_h5_files(self.test_dir, start_time=start, end_time=end)
        
        self.assertEqual(len(files), 2, "Should find 2 files in time window")

        for f in files:
            m = os.path.splitext(os.path.basename(f))[0].split('_')[-1]
            dt = datetime.strptime(m, "%Y%m%d%H%M")
            self.assertGreaterEqual(dt, start, f"{dt} < {start}")
            self.assertLess(dt, end, f"{dt} >= {end}")

    def test_pattern_filtering(self):
        """测试 glob 模式匹配"""
        # 创建不同类型的文件
        extra_file = os.path.join(self.test_dir, "extra.tmp")
        with open(extra_file, 'w') as f:
            f.write("dummy")

        files_all = scan_h5_files(self.test_dir, pattern="*")
        self.assertEqual(len(files_all), len(self.test_files) + 1, "Should include all files")
        self.assertTrue(any("extra.tmp" in f for f in files_all))

        files_h5 = scan_h5_files(self.test_dir, pattern="*.h5")
        self.assertEqual(len(files_h5), len(self.test_files), "Should only include .h5 files")
        self.assertFalse(any("extra.tmp" in f for f in files_h5), "Should not include .tmp when *.h5")

        # 清理额外文件
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

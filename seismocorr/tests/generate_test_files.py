# tests/generate_test_files.py
from pathlib import Path
from datetime import datetime, timedelta

def create_dummy_files(base_dir="test_data"):
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True)

    # 模拟从 2024-09-19 12:00 到 13:59 的文件，每分钟一个
    current = datetime(2024, 9, 19, 12, 0)
    end = datetime(2024, 9, 19, 13, 59)

    while current <= end:
        ts_str = current.strftime("%Y%m%d%H%M")  # 202409191200
        filename = f"FDLY_916m_4m_1m_1000Hz_1000Hz_UTC8_{ts_str}.h5"
        filepath = base_dir / filename
        filepath.touch()  # 创建空文件
        print(f"Created: {filepath}")
        current += timedelta(minutes=1)

    print(f"\n✅ Created {len(list(base_dir.glob('*.h5')))} test files in {base_dir}/")


if __name__ == "__main__":
    create_dummy_files()

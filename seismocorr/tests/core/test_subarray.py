import pytest
import numpy as np

from seismocorr.core.subarray import (
    get_subarray,
    _validate_sensor_xy,
    SUPPORTED_GEOMETRY,
)


# ============================
# Fixtures：构造测试数据
# ============================

@pytest.fixture
def sensor_xy_2d_50():
    """
    生成 50 个 2D 平面坐标点（单位：米），范围大约 2km x 2km。
    """
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 2000.0, size=50)
    y = rng.uniform(0.0, 2000.0, size=50)
    return np.column_stack([x, y]).astype(np.float64)


@pytest.fixture
def sensor_s_1d_200():
    """
    生成 200 个 1D 沿线坐标（单位：米），0~1990，每隔 10m 一个通道。
    注意：这里用 200 个点是为了让随机滑窗更稳定地生成子阵列。
    """
    return (np.arange(200, dtype=np.float64) * 10.0).reshape(-1)


@pytest.fixture
def sensor_s_1d_200_n1(sensor_s_1d_200):
    """
    生成 1D 沿线坐标的 (n,1) 形式。
    """
    return sensor_s_1d_200.reshape(-1, 1)


# ============================
# 1) 工厂函数 get_subarray
# ============================

class TestGetSubarray:
    """测试子阵列构建器工厂函数"""

    def test_get_subarray_supported(self):
        """测试支持的 geometry 都能返回实例"""
        for g in SUPPORTED_GEOMETRY:
            builder = get_subarray(g)
            assert builder is not None

    def test_get_subarray_invalid_geometry(self):
        """测试非法 geometry 抛错"""
        with pytest.raises(ValueError):
            get_subarray("3d")


# ============================
# 2) _validate_sensor_xy
# ============================

class TestValidateSensorPosition:
    """测试输入坐标合法性检查"""

    def test_validate_2d_ok(self, sensor_xy_2d_50):
        """2D：合法 (n,2)"""
        xy = _validate_sensor_xy(sensor_xy_2d_50, geometry="2d")
        assert xy.shape == (50, 2)
        assert xy.dtype == np.float64

    def test_validate_2d_wrong_shape(self):
        """2D：形状错误应报错"""
        bad = np.zeros((50, 3))
        with pytest.raises(ValueError):
            _validate_sensor_xy(bad, geometry="2d")

    def test_validate_2d_too_few(self):
        """2D：少于2个点应报错"""
        bad = np.zeros((1, 2))
        with pytest.raises(ValueError):
            _validate_sensor_xy(bad, geometry="2d")

    def test_validate_2d_nan_inf(self):
        """2D：包含 NaN/Inf 应报错"""
        bad = np.array([[0.0, 1.0], [np.nan, 2.0]])
        with pytest.raises(ValueError):
            _validate_sensor_xy(bad, geometry="2d")

        bad2 = np.array([[0.0, 1.0], [np.inf, 2.0]])
        with pytest.raises(ValueError):
            _validate_sensor_xy(bad2, geometry="2d")

    def test_validate_1d_ok_n(self, sensor_s_1d_200):
        """1D：合法 (n,)"""
        s = _validate_sensor_xy(sensor_s_1d_200, geometry="1d")
        assert s.shape == (200,)
        assert s.dtype == np.float64

    def test_validate_1d_ok_n1(self, sensor_s_1d_200_n1):
        """1D：合法 (n,1)"""
        s = _validate_sensor_xy(sensor_s_1d_200_n1, geometry="1d")
        assert s.shape == (200,)

    def test_validate_1d_wrong_shape(self):
        """1D：不允许 (n,2)/(n,3) 等"""
        bad = np.zeros((50, 2))
        with pytest.raises(ValueError):
            _validate_sensor_xy(bad, geometry="1d")

    def test_validate_1d_all_equal(self):
        """1D：全部相等会报错"""
        bad = np.ones(50)
        with pytest.raises(ValueError):
            _validate_sensor_xy(bad, geometry="1d")

    def test_validate_1d_too_few(self):
        """1D：少于2个点报错"""
        bad = np.array([0.0])
        with pytest.raises(ValueError):
            _validate_sensor_xy(bad, geometry="1d")

    def test_validate_1d_nan_inf(self):
        """1D：包含 NaN/Inf 报错"""
        bad = np.array([0.0, np.nan, 2.0])
        with pytest.raises(ValueError):
            _validate_sensor_xy(bad, geometry="1d")

        bad2 = np.array([0.0, np.inf, 2.0])
        with pytest.raises(ValueError):
            _validate_sensor_xy(bad2, geometry="1d")

    def test_validate_invalid_geometry(self):
        """geometry 参数非法应报错"""
        with pytest.raises(ValueError):
            _validate_sensor_xy(np.zeros((2, 2)), geometry="xxx")


# ============================
# 3) Voronoi 2D 子阵列生成
# ============================

class TestVoronoi2DSubarray:
    """测试 2D Voronoi 子阵列生成"""

    def test_voronoi_generate_nonempty(self, sensor_xy_2d_50):
        """正常情况下应生成非空子阵列列表"""
        builder = get_subarray("2d")
        subs = builder(
            sensor_xy_2d_50,
            n_realizations=30,
            kmin=10,
            kmax=15,
            min_sensors=6,
            random_state=0,
        )
        assert isinstance(subs, list)
        assert len(subs) > 0
        assert all(np.asarray(s).size >= 2 for s in subs)

    def test_voronoi_reproducible_seed(self, sensor_xy_2d_50):
        """相同 random_state 应可复现（结果完全一致）"""
        builder = get_subarray("2d")
        subs1 = builder(
            sensor_xy_2d_50,
            n_realizations=20,
            kmin=10,
            kmax=12,
            min_sensors=6,
            random_state=123,
        )
        subs2 = builder(
            sensor_xy_2d_50,
            n_realizations=20,
            kmin=10,
            kmax=12,
            min_sensors=6,
            random_state=123,
        )
        assert len(subs1) == len(subs2)
        for a, b in zip(subs1, subs2):
            assert np.array_equal(a, b)

    def test_voronoi_fail_when_min_sensors_too_large(self, sensor_xy_2d_50):
        """min_sensors 太大时可能生成失败，应抛 ValueError"""
        builder = get_subarray("2d")
        with pytest.raises(ValueError):
            builder(
                sensor_xy_2d_50,
                n_realizations=10,
                kmin=10,
                kmax=12,
                min_sensors=1000,
                random_state=0,
            )


# ============================
# 4) RandomWindow 1D 子阵列生成
# ============================

class TestRandomWindow1DSubarray:
    """测试 1D：随机滑窗 + 窗内随机抽取通道子阵列生成"""

    def test_random_window_generate_nonempty(self, sensor_s_1d_200):
        """正常情况下应生成非空子阵列列表"""
        builder = get_subarray("1d")
        subs = builder(
            sensor_s_1d_200,
            n_realizations=500,
            window_length=200.0,  # 200m 窗口
            kmin=5,
            kmax=10,
            random_state=0,
        )
        assert isinstance(subs, list)
        assert len(subs) > 0
        assert all(np.asarray(s).size >= 2 for s in subs)

    def test_random_window_accepts_n1(self, sensor_s_1d_200_n1):
        """(n,1) 形式应可用"""
        builder = get_subarray("1d")
        subs = builder(
            sensor_s_1d_200_n1,
            n_realizations=300,
            window_length=200.0,
            kmin=5,
            kmax=10,
            random_state=0,
        )
        assert len(subs) > 0

    def test_random_window_shuffled_input_ok(self, sensor_s_1d_200):
        """输入打乱顺序也应能生成（内部会排序）"""
        rng = np.random.default_rng(0)
        shuffled = sensor_s_1d_200.copy()
        rng.shuffle(shuffled)

        builder = get_subarray("1d")
        subs = builder(
            shuffled,
            n_realizations=300,
            window_length=200.0,
            kmin=5,
            kmax=10,
            random_state=0,
        )
        assert len(subs) > 0

    def test_random_window_reproducible_seed(self, sensor_s_1d_200):
        """相同 random_state 应可复现（结果完全一致）"""
        builder = get_subarray("1d")
        subs1 = builder(
            sensor_s_1d_200,
            n_realizations=500,
            window_length=200.0,
            kmin=5,
            kmax=10,
            random_state=123,
        )
        subs2 = builder(
            sensor_s_1d_200,
            n_realizations=500,
            window_length=200.0,
            kmin=5,
            kmax=10,
            random_state=123,
        )
        assert len(subs1) == len(subs2)
        for a, b in zip(subs1, subs2):
            assert np.array_equal(a, b)

    def test_random_window_indices_and_size_constraints(self, sensor_s_1d_200):
        """每个子阵列索引范围正确；大小满足 kmin~kmax（除去极端情况后应该都满足）"""
        builder = get_subarray("1d")
        kmin, kmax = 5, 10
        subs = builder(
            sensor_s_1d_200,
            n_realizations=500,
            window_length=200.0,
            kmin=kmin,
            kmax=kmax,
            random_state=0,
        )
        n = sensor_s_1d_200.size
        for idx in subs:
            idx = np.asarray(idx)
            assert idx.ndim == 1
            assert np.all(idx >= 0) and np.all(idx < n)
            assert idx.size >= kmin
            assert idx.size <= kmax
            # 返回的是 np.unique 后的结果，应严格递增
            assert np.all(np.diff(idx) > 0)

    def test_random_window_deduplicate(self, sensor_s_1d_200):
        """子阵列应去重（key=tuple(idx) 唯一）"""
        builder = get_subarray("1d")
        subs = builder(
            sensor_s_1d_200,
            n_realizations=2000,
            window_length=100.0,
            kmin=5,
            kmax=10,
            random_state=0,
        )
        keys = [tuple(np.asarray(s, dtype=np.int64).tolist()) for s in subs]
        assert len(keys) == len(set(keys))

    def test_random_window_fail_window_too_large(self, sensor_s_1d_200):
        """window_length 大于沿线总长度会报错"""
        builder = get_subarray("1d")
        with pytest.raises(ValueError):
            builder(
                sensor_s_1d_200,
                n_realizations=10,
                window_length=1e9,
                kmin=5,
                kmax=10,
                random_state=0,
            )

    def test_random_window_fail_bad_kmin(self, sensor_s_1d_200):
        """kmin<2 报错"""
        builder = get_subarray("1d")
        with pytest.raises(ValueError):
            builder(
                sensor_s_1d_200,
                n_realizations=10,
                window_length=200.0,
                kmin=1,
                kmax=10,
                random_state=0,
            )

    def test_random_window_fail_kmax_lt_kmin(self, sensor_s_1d_200):
        """kmax<kmin 报错"""
        builder = get_subarray("1d")
        with pytest.raises(ValueError):
            builder(
                sensor_s_1d_200,
                n_realizations=10,
                window_length=200.0,
                kmin=8,
                kmax=7,
                random_state=0,
            )

    def test_random_window_fail_bad_n_realizations(self, sensor_s_1d_200):
        """n_realizations<=0 报错"""
        builder = get_subarray("1d")
        with pytest.raises(ValueError):
            builder(
                sensor_s_1d_200,
                n_realizations=0,
                window_length=200.0,
                kmin=5,
                kmax=10,
                random_state=0,
            )

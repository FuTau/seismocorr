# seismocorr/tests/spfi/test_assumption.py
import pytest
import numpy as np

from seismocorr.core.spfi.assumption import (
    get_assumption,
    StationAvgBuilder,
    RayAvgBuilder,
    SUPPORTED_ASSUMPTION,
    SUPPORTED_GEOMETRY,
    _validate_geometry,
    _infer_n_sensors,
    _normalize_subarray,
    _ensure_edges_1d,
    _pairs_from_indices,
    _rows_ray_2d,
    _ray_grid_intersections_2d,
)


# =========================
# Fixtures: 构造可复用测试数据
# =========================

@pytest.fixture
def sensor_xy_2d():
    # 4个台站，组成一个小方形（单位：米）
    return np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def sensor_s_1d():
    # 1D 沿线坐标（单位：米）
    return np.linspace(0.0, 90.0, 10, dtype=np.float64)


@pytest.fixture
def subarray_basic():
    # 注意：包含“需要被过滤/去重”的情况
    # - [0] 会被过滤（size<2）
    # - [1,1,2] 会去重 -> [1,2]
    return [
        [0, 1, 2],
        [1, 1, 2],
        [3],  # 会被过滤
    ]


@pytest.fixture
def grid_centers():
    # 2D 网格中心点（严格递增）
    grid_x = np.array([0.0, 5.0, 10.0], dtype=np.float64)
    grid_y = np.array([0.0, 5.0, 10.0], dtype=np.float64)
    return grid_x, grid_y


# =========================
# get_assumption / geometry
# =========================

class TestFactoryAndGeometry:
    def test_get_assumption_ok(self):
        for a in SUPPORTED_ASSUMPTION:
            builder = get_assumption(a)
            assert builder is not None

    def test_get_assumption_invalid(self):
        with pytest.raises(ValueError):
            get_assumption("not_exist")

    def test_validate_geometry_ok(self):
        for g in SUPPORTED_GEOMETRY:
            _validate_geometry(g)

    def test_validate_geometry_invalid(self):
        with pytest.raises(ValueError):
            _validate_geometry("3d")


# =========================
# _infer_n_sensors / _normalize_subarray
# =========================

class TestInferAndNormalize:
    def test_infer_n_sensors_2d(self, sensor_xy_2d):
        n = _infer_n_sensors(sensor_xy_2d, geometry="2d")
        assert n == 4

    def test_infer_n_sensors_1d_vector(self, sensor_s_1d):
        n = _infer_n_sensors(sensor_s_1d, geometry="1d")
        assert n == sensor_s_1d.size

    def test_infer_n_sensors_1d_n1(self, sensor_s_1d):
        s = sensor_s_1d.reshape(-1, 1)
        n = _infer_n_sensors(s, geometry="1d")
        assert n == sensor_s_1d.size

    def test_normalize_subarray_filters_and_dedup(self):
        subs = _normalize_subarray([[0, 1, 2], [1, 1, 2], [3]], n_sensors=10)
        assert len(subs) == 2
        assert np.array_equal(subs[1], np.array([1, 2], dtype=np.int64))

    def test_normalize_subarray_out_of_range(self):
        with pytest.raises(ValueError):
            _normalize_subarray([[0, 99]], n_sensors=10)

        with pytest.raises(ValueError):
            _normalize_subarray([[-1, 1]], n_sensors=10)


# =========================
# StationAvgBuilder
# =========================

class TestStationAvgBuilder:
    def test_station_avg_shape_and_row_sum(self, sensor_xy_2d, subarray_basic):
        A = StationAvgBuilder().assumption(
            subarray=subarray_basic,
            sensor_xy=sensor_xy_2d,
            geometry="2d",
        )
        # subarray_basic 里有 2 个有效子阵列
        assert A.shape == (2, 4)

        # 每行应该是“平均算子”，行和为 1
        row_sum = np.asarray(A.sum(axis=1)).reshape(-1)
        assert np.allclose(row_sum, 1.0)

    def test_station_avg_invalid_geometry(self, sensor_xy_2d, subarray_basic):
        with pytest.raises(ValueError):
            StationAvgBuilder().assumption(
                subarray=subarray_basic,
                sensor_xy=sensor_xy_2d,
                geometry="3d",
            )

    def test_station_avg_empty_after_filter(self, sensor_xy_2d):
        # 全部 size<2 -> 会被过滤为空 -> 抛错
        with pytest.raises(ValueError):
            StationAvgBuilder().assumption(
                subarray=[[0], [1]],
                sensor_xy=sensor_xy_2d,
                geometry="2d",
            )

    def test_station_avg_accepts_1d_geometry(self, sensor_s_1d, subarray_basic):
        # 1d 情况下只用来推断 n_sensors（不做 shape 检查）
        A = StationAvgBuilder().assumption(
            subarray=subarray_basic,
            sensor_xy=sensor_s_1d,
            geometry="1d",
        )
        assert A.shape[0] == 2
        assert A.shape[1] == sensor_s_1d.size
        row_sum = np.asarray(A.sum(axis=1)).reshape(-1)
        assert np.allclose(row_sum, 1.0)


# =========================
# RayAvgBuilder: 1d downgrade
# =========================

class TestRayAvgDowngrade1D:
    def test_ray_avg_downgrade_to_station_avg(self, sensor_s_1d, subarray_basic):
        A_ray = RayAvgBuilder().assumption(
            subarray=subarray_basic,
            sensor_xy=sensor_s_1d,
            geometry="1d",
        )
        A_sta = StationAvgBuilder().assumption(
            subarray=subarray_basic,
            sensor_xy=sensor_s_1d,
            geometry="1d",
        )

        # shape 一致，并且矩阵内容应一致
        assert A_ray.shape == A_sta.shape
        assert (A_ray != A_sta).nnz == 0


# =========================
# RayAvgBuilder: 2d ray_avg
# =========================

class TestRayAvg2D:
    def test_ray_avg_requires_grid(self, sensor_xy_2d, subarray_basic):
        with pytest.raises(ValueError):
            RayAvgBuilder().assumption(
                subarray=subarray_basic,
                sensor_xy=sensor_xy_2d,
                geometry="2d",
                grid_x=None,
                grid_y=None,
            )

    def test_ray_avg_grid_must_be_strictly_increasing(self, sensor_xy_2d, subarray_basic):
        grid_x = np.array([0.0, 0.0, 10.0])  # 非严格递增
        grid_y = np.array([0.0, 5.0, 10.0])
        with pytest.raises(ValueError):
            RayAvgBuilder().assumption(
                subarray=subarray_basic,
                sensor_xy=sensor_xy_2d,
                geometry="2d",
                grid_x=grid_x,
                grid_y=grid_y,
            )

    def test_ray_avg_shape_and_row_sum(self, sensor_xy_2d, subarray_basic, grid_centers):
        grid_x, grid_y = grid_centers
        A = RayAvgBuilder().assumption(
            subarray=subarray_basic,
            sensor_xy=sensor_xy_2d,
            geometry="2d",
            grid_x=grid_x,
            grid_y=grid_y,
            pair_sampling=None,
            random_state=0,
        )
        # 2 个有效子阵列；网格 cell 数=(len(grid_x))*(len(grid_y)) 这里 edges->nx=3, ny=3 => 9
        assert A.shape == (2, 9)

        # ray_avg 每行权重归一化，行和为 1（允许少量数值误差）
        row_sum = np.asarray(A.sum(axis=1)).reshape(-1)
        assert np.allclose(row_sum, 1.0, atol=1e-10)

    def test_ray_avg_pair_sampling_reproducible(self, sensor_xy_2d, subarray_basic, grid_centers):
        grid_x, grid_y = grid_centers

        A1 = RayAvgBuilder().assumption(
            subarray=subarray_basic,
            sensor_xy=sensor_xy_2d,
            geometry="2d",
            grid_x=grid_x,
            grid_y=grid_y,
            pair_sampling=1,
            random_state=123,
        )
        A2 = RayAvgBuilder().assumption(
            subarray=subarray_basic,
            sensor_xy=sensor_xy_2d,
            geometry="2d",
            grid_x=grid_x,
            grid_y=grid_y,
            pair_sampling=1,
            random_state=123,
        )
        assert A1.shape == A2.shape
        assert (A1 != A2).nnz == 0  # 同 seed 必须一致

    def test_ray_avg_pair_sampling_different_seed_may_differ(self, sensor_xy_2d, subarray_basic, grid_centers):
        grid_x, grid_y = grid_centers

        A1 = RayAvgBuilder().assumption(
            subarray=subarray_basic,
            sensor_xy=sensor_xy_2d,
            geometry="2d",
            grid_x=grid_x,
            grid_y=grid_y,
            pair_sampling=1,
            random_state=1,
        )
        A2 = RayAvgBuilder().assumption(
            subarray=subarray_basic,
            sensor_xy=sensor_xy_2d,
            geometry="2d",
            grid_x=grid_x,
            grid_y=grid_y,
            pair_sampling=1,
            random_state=2,
        )
        # 不强行要求一定不同（可能碰巧抽到同一对），但至少允许不同
        assert A1.shape == A2.shape


# =========================
# 内部函数单测：edges / pairs / intersections
# =========================

class TestInternalHelpers:
    def test_ensure_edges_1d_ok(self):
        centers = np.array([0.0, 10.0, 20.0], dtype=np.float64)
        edges = _ensure_edges_1d(centers)
        assert edges.shape == (4,)
        assert np.all(np.diff(edges) > 0)

    def test_ensure_edges_1d_invalid(self):
        with pytest.raises(ValueError):
            _ensure_edges_1d(np.array([0.0]))  # 太短
        with pytest.raises(ValueError):
            _ensure_edges_1d(np.array([0.0, 0.0, 1.0]))  # 非严格递增

    def test_pairs_from_indices_all(self):
        idx = np.array([0, 1, 2, 3], dtype=np.int64)
        rng = np.random.default_rng(0)
        pairs = _pairs_from_indices(idx, pair_sampling=None, rng=rng)
        assert len(pairs) == 6  # C(4,2)=6

    def test_pairs_from_indices_sampling(self):
        idx = np.array([0, 1, 2, 3], dtype=np.int64)
        rng = np.random.default_rng(0)
        pairs = _pairs_from_indices(idx, pair_sampling=2, rng=rng)
        assert len(pairs) == 2
        # 不重复
        assert len(set(pairs)) == 2

    def test_ray_grid_intersections_basic(self):
        x_edges = np.array([0.0, 5.0, 10.0], dtype=np.float64)
        y_edges = np.array([0.0, 5.0, 10.0], dtype=np.float64)

        # 斜线穿过多个 cell
        pts, seg_len, cell_ij = _ray_grid_intersections_2d(
            x_edges=x_edges,
            y_edges=y_edges,
            x1=0.0,
            y1=0.0,
            x2=10.0,
            y2=10.0,
        )
        assert pts.shape[0] >= 2
        assert seg_len.size == pts.shape[0] - 1
        assert cell_ij.shape == (seg_len.size, 2)
        assert np.all(seg_len >= 0.0)

    def test_rows_ray_2d_outputs(self, sensor_xy_2d, grid_centers):
        grid_x, grid_y = grid_centers
        x_edges = _ensure_edges_1d(grid_x)
        y_edges = _ensure_edges_1d(grid_y)

        subs = [np.array([0, 1, 2], dtype=np.int64)]
        rng = np.random.default_rng(0)

        rows, cols, data = _rows_ray_2d(
            subarray=subs,
            sensor_xy=sensor_xy_2d,
            x_edges=x_edges,
            y_edges=y_edges,
            nx=x_edges.size - 1,
            ny=y_edges.size - 1,
            pair_sampling=None,
            rng=rng,
        )

        assert rows.ndim == 1 and cols.ndim == 1 and data.ndim == 1
        assert rows.size == cols.size == data.size
        assert np.all(data > 0.0)
        # 只有一个子阵列 -> rows 里应该全是 0
        assert np.all(rows == 0)

import pytest
import numpy as np
import sys
import os
from scipy import signal
from seismocorr.preprocessing.das_process import (
    _as_2d,
    _robust_zscore,
    _cosine_taper_1d,
    _channel_score,
    _repair_bad_channels,
    get_das_preprocessor,
    CMMedian,
    CMMean,
    CMPCA,
    CMFK,
    FKFanFilter,
    MedianFilterSciPy,
    BadRobust,
)


class TestHelpers:
    def test_as_2d_channel_axis_0(self, das_data_ct):
        X, transposed = _as_2d(das_data_ct, channel_axis=0)
        assert X.shape == das_data_ct.shape
        assert transposed is False
        assert X is das_data_ct  # view/np.asarray -> same object for ndarray

    def test_as_2d_channel_axis_1(self, das_data_tc):
        X, transposed = _as_2d(das_data_tc, channel_axis=1)
        assert X.shape == (das_data_tc.shape[1], das_data_tc.shape[0])
        assert transposed is True
        assert np.array_equal(X, das_data_tc.T)

    def test_as_2d_errors(self):
        with pytest.raises(ValueError, match="data must be 2D"):
            _as_2d(np.zeros((2, 3, 4)), channel_axis=0)
        with pytest.raises(ValueError, match="channel_axis must be 0 or 1"):
            _as_2d(np.zeros((2, 3)), channel_axis=2)

    def test_robust_zscore_basic(self):
        v = np.array([0, 0, 0, 10], dtype=float)
        z = _robust_zscore(v)
        assert z.shape == v.shape
        assert np.isfinite(z).all()
        # 中位数应在 0 附近
        assert np.isclose(np.median(z), 0.0, atol=1e-12)

    def test_cosine_taper_bounds(self):
        w = _cosine_taper_1d(100, frac=0.1)
        assert w.shape == (100,)
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)
        # 中间应接近 1
        assert np.isclose(w[50], 1.0, atol=1e-12)

    def test_channel_score_methods(self, das_data_ct):
        for m in ["rms", "std", "mad", "kurtosis"]:
            s = _channel_score(das_data_ct, m)
            assert s.shape == (das_data_ct.shape[0],)
            assert np.isfinite(s).all()

        with pytest.raises(ValueError, match="Unknown score method"):
            _channel_score(das_data_ct, "bad_method")  # type: ignore

    def test_repair_bad_channels(self):
        x = np.arange(5 * 10, dtype=float).reshape(5, 10)
        bad = np.array([False, True, False, True, False])

        y0 = _repair_bad_channels(x, bad, "none")
        assert np.array_equal(y0, x)
        assert y0 is not x

        y1 = _repair_bad_channels(x, bad, "zero")
        assert np.all(y1[bad] == 0.0)

        y2 = _repair_bad_channels(x, bad, "interp_linear")
        # 1 应该在 0 和 2 之间
        assert np.allclose(y2[1], 0.5 * (x[0] + x[2]))
        # 3 应该在 2 和 4 之间
        assert np.allclose(y2[3], 0.5 * (x[2] + x[4]))

        y3 = _repair_bad_channels(x, bad, "interp_median")
        assert np.allclose(y3[1], np.median(np.stack([x[0], x[2]]), axis=0))
        assert np.allclose(y3[3], np.median(np.stack([x[2], x[4]]), axis=0))


class TestCommonMode:
    def test_cm_median_removes_common_mode(self, das_data_ct):
        p = CMMedian(return_cm=True)
        y = p.apply(das_data_ct)

        assert y.shape == das_data_ct.shape
        assert p.info["method"] == "cm_median"
        assert "cm" in p.info

        # 每个时刻跨道中位数应接近 0
        cm_after = np.median(y, axis=0)
        assert np.isclose(np.median(cm_after), 0.0, atol=1e-10)

    def test_cm_mean_removes_common_mode(self, das_data_ct):
        p = CMMean(return_cm=True)
        y = p.apply(das_data_ct)

        assert y.shape == das_data_ct.shape
        assert p.info["method"] == "cm_mean"
        assert "cm" in p.info

        # 每个时刻跨道均值应接近 0
        mean_after = np.mean(y, axis=0)
        assert np.isclose(np.mean(mean_after), 0.0, atol=1e-10)

    def test_cm_axis_handling(self, das_data_tc):
        p = CMMedian(channel_axis=1)
        y = p.apply(das_data_tc)
        assert y.shape == das_data_tc.shape  # 输出保持 (time, channels)

    def test_cm_rejects_naninf(self, das_with_naninf):
        for cls in (CMMedian, CMMean, CMPCA, CMFK):
            if cls is CMFK:
                p = cls(dx=10.0, k0=0.1)
            else:
                p = cls()
            with pytest.raises(ValueError, match="data contains nan/inf"):
                p.apply(das_with_naninf)

    def test_cm_pca_basic(self, das_data_ct):
        p = CMPCA(n_components=1, center="median", return_info=True)
        y = p.apply(das_data_ct)

        assert y.shape == das_data_ct.shape
        assert p.info["method"] == "cm_pca"
        assert "info" in p.info
        assert p.info["info"]["rank_removed"] >= 1
        frac = p.info["info"]["explained_energy_fraction"]
        assert 0.0 <= frac <= 1.0

    def test_cm_pca_invalid_params(self):
        with pytest.raises(ValueError, match="n_components must be"):
            CMPCA(n_components=0)
        with pytest.raises(ValueError, match="center must be"):
            CMPCA(n_components=1, center="bad")  # type: ignore

    def test_cm_fk_basic(self, das_data_ct):
        # k0 取较小值，仅去掉非常慢变的空间分量
        p = CMFK(dx=10.0, k0=0.05, taper=0.1, return_info=True)
        y = p.apply(das_data_ct)

        assert y.shape == das_data_ct.shape
        assert p.info["method"] == "cm_fk"
        assert "info" in p.info
        assert np.isfinite(y).all()

    def test_cm_fk_invalid_params(self):
        with pytest.raises(ValueError, match="dx must be > 0"):
            CMFK(dx=0, k0=0.1)
        with pytest.raises(ValueError, match="k0 must be > 0"):
            CMFK(dx=10, k0=0)
        with pytest.raises(ValueError, match="taper must be >= 0"):
            CMFK(dx=10, k0=0.1, taper=-1)


class TestFKFanFilter:
    def test_fk_fan_basic(self, das_data_ct, dt_dx):
        dt, dx = dt_dx
        p = FKFanFilter(dt=dt, dx=dx, fmin=0.1, fmax=30.0, vmin=100.0, vmax=5000.0)
        y = p.apply(das_data_ct)

        assert y.shape == das_data_ct.shape
        assert p.info["method"] == "fk_fan"
        assert np.isfinite(y).all()

    def test_fk_fan_axis_handling(self, das_data_tc, dt_dx):
        dt, dx = dt_dx
        p = FKFanFilter(dt=dt, dx=dx, channel_axis=1)
        y = p.apply(das_data_tc)
        assert y.shape == das_data_tc.shape

    def test_fk_fan_invalid_params(self):
        with pytest.raises(ValueError, match="dt/dx must be > 0"):
            FKFanFilter(dt=0.0, dx=10.0)
        with pytest.raises(ValueError, match='mode'):
            FKFanFilter(dt=0.01, dx=10.0, mode="bad")  # type: ignore
        with pytest.raises(ValueError, match='direction'):
            FKFanFilter(dt=0.01, dx=10.0, direction="bad")  # type: ignore


class TestMedianFilterSciPy:
    def test_median_filter_basic(self, das_data_ct):
        p = MedianFilterSciPy(k_time=5, k_chan=3, nan_policy="propagate")
        y = p.apply(das_data_ct)

        assert y.shape == das_data_ct.shape
        assert p.info["method"] == "median_scipy"
        assert np.isfinite(y).all()

    def test_median_filter_nan_omit(self, das_data_ct):
        x = das_data_ct.copy()
        x[0, 0] = np.nan
        p = MedianFilterSciPy(k_time=3, k_chan=3, nan_policy="omit")
        y = p.apply(x)

        assert y.shape == x.shape
        # omit 模式下通常可消掉局部 NaN（不保证全无 NaN，但应更稳健）
        assert np.isfinite(y[1:, 1:]).all()

    def test_median_filter_invalid_params(self):
        with pytest.raises(ValueError, match="必须是正奇数"):
            MedianFilterSciPy(k_time=4, k_chan=3)
        with pytest.raises(ValueError, match="nan_policy"):
            MedianFilterSciPy(nan_policy="bad")  # type: ignore


class TestBadRobust:
    def test_badrobust_detects_naninf(self, das_with_naninf):
        p = BadRobust(score_method="rms", z_thresh=6.0, repair="none")
        y = p.apply(das_with_naninf)

        assert y.shape == das_with_naninf.shape
        assert p.info["method"] == "bad_robust"
        assert "bad_mask" in p.info
        bad = p.info["bad_mask"]
        assert bad.dtype == bool
        # 第0、1通道应被判坏（含 NaN/Inf）
        assert bool(bad[0]) is True
        assert bool(bad[1]) is True
        assert np.isfinite(y).all()  # none 也会 finite-clean 成 0

    def test_badrobust_repair_zero(self, das_with_naninf):
        p = BadRobust(repair="zero")
        y = p.apply(das_with_naninf)
        bad = p.info["bad_mask"]
        assert np.all(y[bad] == 0.0)

    def test_badrobust_repair_interp_linear(self, rng):
        # 构造：中间一条坏道（高能量）让 robust z 识别
        n_ch, n_t = 7, 500
        x = 0.1 * rng.standard_normal((n_ch, n_t))
        x[3] += 20.0  # 强异常通道

        p = BadRobust(score_method="rms", z_thresh=3.0, repair="interp_linear")
        y = p.apply(x)

        bad = p.info["bad_mask"]
        assert bool(bad[3]) is True
        # 线性插值：3 应该在 2 和 4 之间
        assert np.allclose(y[3], 0.5 * (y[2] + y[4]), atol=1e-6)

    def test_badrobust_saturation(self, rng):
        n_ch, n_t = 8, 400
        x = 0.1 * rng.standard_normal((n_ch, n_t))
        # 第5道大量饱和
        x[5, :] = 100.0

        p = BadRobust(
            score_method="rms",
            z_thresh=10.0,  # 让 score 不一定抓到，主要测 sat
            repair="zero",
            sat_value=50.0,
            sat_frac_thresh=0.5,
        )
        y = p.apply(x)
        bad = p.info["bad_mask"]
        assert bool(bad[5]) is True
        assert np.all(y[5] == 0.0)


class TestFactory:
    @pytest.mark.parametrize(
        "name,kwargs,cls",
        [
            ("cm_median", {}, CMMedian),
            ("cm_mean", {}, CMMean),
            ("cm_pca", {"n_components": 1}, CMPCA),
            ("cm_fk", {"dx": 10.0, "k0": 0.1}, CMFK),
            ("fk_fan", {"dt": 0.01, "dx": 10.0}, FKFanFilter),
            ("median", {"k_time": 3, "k_chan": 3}, MedianFilterSciPy),
            ("bad_robust", {}, BadRobust),
        ],
    )
    def test_get_das_preprocessor(self, name, kwargs, cls):
        p = get_das_preprocessor(name, **kwargs)
        assert isinstance(p, cls)
        assert hasattr(p, "apply")

    def test_get_das_preprocessor_invalid(self):
        with pytest.raises(ValueError, match="Unknown DAS preprocessor"):
            get_das_preprocessor("not_exists")


def test_preprocessors_do_not_modify_input(das_data_ct, dt_dx):
    x = das_data_ct.copy()
    orig = x.copy()
    dt, dx = dt_dx

    procs = [
        CMMedian(),
        CMMean(),
        CMPCA(n_components=1),
        CMFK(dx=dx, k0=0.1),
        FKFanFilter(dt=dt, dx=dx),
        MedianFilterSciPy(k_time=3, k_chan=3),
        BadRobust(repair="none"),
    ]

    for p in procs:
        y = p.apply(x)
        assert np.array_equal(x, orig)
        assert y is not x
        assert y.shape == x.shape
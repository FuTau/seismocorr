import pytest
import numpy as np
import sys
import os
from typing import List, Dict, Any, Tuple

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from seismocorr.plugins.disper import (
    DispersionConfig,
    PlotConfig,
    DispersionMethod,
    DispersionStrategy,
    FJ,
    FJ_RR,
    MFJ_RR,
    SlantStack,
    MASW,
    DispersionFactory,
    DispersionAnalyzer
)


class TestDispersionConfig:
    """测试频散成像配置类"""
    
    def test_dispersion_config_defaults(self):
        """测试默认配置"""
        config = DispersionConfig()
        
        assert config.freqmin == 0.1
        assert config.freqmax == 10.0
        assert config.vmin == 100.0
        assert config.vmax == 5000.0
        assert config.vnum == 100
        assert config.sampling_rate == 100.0
    
    def test_dispersion_config_custom(self):
        """测试自定义配置"""
        config = DispersionConfig(
            freqmin=0.5,
            freqmax=5.0,
            vmin=500.0,
            vmax=3000.0,
            vnum=200,
            sampling_rate=50.0
        )
        
        assert config.freqmin == 0.5
        assert config.freqmax == 5.0
        assert config.vmin == 500.0
        assert config.vmax == 3000.0
        assert config.vnum == 200
        assert config.sampling_rate == 50.0


class TestPlotConfig:
    """测试绘图配置类"""
    
    def test_plot_config_defaults(self):
        """测试默认绘图配置"""
        config = PlotConfig()
        
        assert config.fig_width == 10
        assert config.fig_height == 6
        assert config.font_size == 12
        assert config.cmap == 'jet'
        assert config.vmin == 0.0
        assert config.vmax == 0.8
    
    def test_plot_config_custom(self):
        """测试自定义绘图配置"""
        config = PlotConfig(
            fig_width=12,
            fig_height=8,
            font_size=14,
            cmap='viridis',
            vmin=0.1,
            vmax=1.0
        )
        
        assert config.fig_width == 12
        assert config.fig_height == 8
        assert config.font_size == 14
        assert config.cmap == 'viridis'
        assert config.vmin == 0.1
        assert config.vmax == 1.0


class TestDispersionMethod:
    """测试频散成像方法枚举"""
    
    def test_dispersion_method_values(self):
        """测试枚举值"""
        assert DispersionMethod.FJ.value == "fj"
        assert DispersionMethod.FJ_RR.value == "fj_rr"
        assert DispersionMethod.MFJ_RR.value == "mfj_rr"
        assert DispersionMethod.SLANT_STACK.value == "slant_stack"
        assert DispersionMethod.MASW.value == "masw"
    
    def test_dispersion_method_from_string(self):
        """测试从字符串创建枚举"""
        methods = {
            "fj": DispersionMethod.FJ,
            "fj_rr": DispersionMethod.FJ_RR,
            "mfj_rr": DispersionMethod.MFJ_RR,
            "slant_stack": DispersionMethod.SLANT_STACK,
            "masw": DispersionMethod.MASW
        }
        
        for name, method in methods.items():
            assert DispersionMethod(name) == method


@pytest.fixture
def sample_dispersion_data():
    """生成测试用的频散成像数据"""
    np.random.seed(42)
    
    # 频率数组
    n_freq = 50
    f = np.linspace(0.1, 10.0, n_freq)
    
    # 台站距离数组
    n_stations = 8
    dist = np.linspace(100, 1000, n_stations)
    
    # 频域互相关数据（复数）
    cc_array_f = np.random.randn(n_stations, n_freq) + 1j * np.random.randn(n_stations, n_freq)
    
    return {
        'f': f,
        'dist': dist,
        'cc_array_f': cc_array_f
    }


@pytest.fixture
def dispersion_config():
    """生成测试配置"""
    return DispersionConfig(
        freqmin=0.5,
        freqmax=5.0,
        vmin=500.0,
        vmax=3000.0,
        vnum=50,
        sampling_rate=100.0
    )


@pytest.fixture
def masw_time_domain_data():
    """生成MASW方法需要的时域数据"""
    np.random.seed(42)
    
    # 时域信号
    n_samples = 1000
    n_stations = 50
    sampling_rate = 100.0
    
    # 创建包含面波信号的时域数据
    t = np.linspace(0, 10, n_samples)
    u = np.zeros((n_samples, n_stations))
    
    for i in range(n_stations):
        # 模拟面波信号
        freq = 5.0  # 5 Hz
        velocity = 1000.0  # 1000 m/s
        delay = i * 50 / velocity  # 根据距离延迟
        
        signal = np.sin(2 * np.pi * freq * (t - delay)) * np.exp(-0.5 * (t - 2 - delay)**2)
        noise = 0.1 * np.random.randn(n_samples)
        u[:, i] = signal + noise
    
    return {
        'u': u,
        'sampling_rate': sampling_rate,
        'dist': np.linspace(50, 300, n_stations)  # 台站距离
    }


class TestDispersionStrategies:
    """测试频散成像策略基类"""
    
    def test_strategy_abstract_method(self):
        """测试策略基类的抽象方法"""
        # 应该不能直接实例化抽象基类
        with pytest.raises(TypeError):
            strategy = DispersionStrategy()
    
    def test_calculate_weights_method(self, sample_dispersion_data):
        """测试权重计算方法"""
        dist = sample_dispersion_data['dist']
        
        # 创建一个具体策略实例来测试权重计算
        class TestStrategy(DispersionStrategy):
            def compute(self, cc_array_f, f, dist, config):
                return np.zeros((len(f), config.vnum))
        
        strategy = TestStrategy()
        weights = strategy._calculate_weights(dist)
        
        # 检查权重计算
        assert len(weights) == len(dist)
        assert np.all(weights >= 0)  # 权重应为非负
        assert np.sum(weights) > 0  # 权重和应大于0


class TestFJStrategy:
    """测试FJ策略"""
    
    def test_fj_compute_basic(self, sample_dispersion_data, dispersion_config):
        """测试FJ基本计算"""
        strategy = FJ()
        result = strategy.compute(
            sample_dispersion_data['cc_array_f'],
            sample_dispersion_data['f'],
            sample_dispersion_data['dist'],
            dispersion_config
        )
        
        # 检查结果形状
        assert result.shape == (len(sample_dispersion_data['f']), dispersion_config.vnum)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_fj_compute_zero_frequency(self, sample_dispersion_data, dispersion_config):
        """测试FJ处理零频率"""
        # 在频率数组中添加零频率
        f_with_zero = np.concatenate([[0.0], sample_dispersion_data['f']])
        cc_array_f_extended = np.zeros((sample_dispersion_data['cc_array_f'].shape[0], 
                                       len(f_with_zero)), dtype=complex)
        cc_array_f_extended[:, 1:] = sample_dispersion_data['cc_array_f']
        
        strategy = FJ()
        result = strategy.compute(
            cc_array_f_extended,
            f_with_zero,
            sample_dispersion_data['dist'],
            dispersion_config
        )
        
        # 应该能正确处理零频率（跳过计算）
        assert result.shape == (dispersion_config.vnum, len(f_with_zero))


class TestFJ_RRStrategy:
    """测试FJ_RR策略"""
    
    def test_fj_rr_compute_basic(self, sample_dispersion_data, dispersion_config):
        """测试FJ_RR基本计算"""
        strategy = FJ_RR()
        result = strategy.compute(
            sample_dispersion_data['cc_array_f'],
            sample_dispersion_data['f'],
            sample_dispersion_data['dist'],
            dispersion_config
        )
        
        # 检查结果形状
        assert result.shape == (len(sample_dispersion_data['f']), dispersion_config.vnum)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_fj_rr_vs_fj_difference(self, sample_dispersion_data, dispersion_config):
        """测试FJ_RR与FJ的差异"""
        fj_strategy = FJ()
        fj_rr_strategy = FJ_RR()
        
        fj_result = fj_strategy.compute(
            sample_dispersion_data['cc_array_f'],
            sample_dispersion_data['f'],
            sample_dispersion_data['dist'],
            dispersion_config
        )
        
        fj_rr_result = fj_rr_strategy.compute(
            sample_dispersion_data['cc_array_f'],
            sample_dispersion_data['f'],
            sample_dispersion_data['dist'],
            dispersion_config
        )
        
        # 两种方法的结果应该不同（使用不同的贝塞尔函数）
        assert not np.array_equal(fj_result, fj_rr_result)


class TestMFJ_RRStrategy:
    """测试MFJ_RR策略"""
    
    def test_mfj_rr_compute_basic(self, sample_dispersion_data, dispersion_config):
        """测试MFJ_RR基本计算"""
        strategy = MFJ_RR()
        result = strategy.compute(
            sample_dispersion_data['cc_array_f'],
            sample_dispersion_data['f'],
            sample_dispersion_data['dist'],
            dispersion_config
        )
        
        # 检查结果形状
        assert result.shape == (len(sample_dispersion_data['f']), dispersion_config.vnum)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestSlantStackStrategy:
    """测试SlantStack策略"""
    
    def test_slant_stack_compute_basic(self, sample_dispersion_data, dispersion_config):
        """测试SlantStack基本计算"""
        strategy = SlantStack()
        result = strategy.compute(
            sample_dispersion_data['cc_array_f'],
            sample_dispersion_data['f'],
            sample_dispersion_data['dist'],
            dispersion_config
        )
        
        # 检查结果形状
        assert result.shape == (len(sample_dispersion_data['f']), dispersion_config.vnum)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        


class TestMASWStrategy:
    """测试MASW策略"""
    
    def test_masw_compute_basic(self, masw_time_domain_data, dispersion_config):
        """测试MASW基本计算"""
        strategy = MASW()
        u = masw_time_domain_data['u']
        dist = masw_time_domain_data['dist']
        
        # MASW需要不同的参数
        A = strategy.compute(u, None, dist, dispersion_config)
        
        # 检查结果形状
        assert A.shape == (dispersion_config.vnum, len(np.fft.fftfreq(u.shape[0], d=1/dispersion_config.sampling_rate)[:u.shape[0]//2]))
        assert not np.any(np.isnan(A))
        assert not np.any(np.isinf(A))
    
    def test_masw_frequency_range(self, masw_time_domain_data, dispersion_config):
        """测试MASW频率范围"""
        strategy = MASW()
        u = masw_time_domain_data['u']
        dist = masw_time_domain_data['dist']
        sampling_rate = masw_time_domain_data['sampling_rate']
        
        dispersion_config.sampling_rate = sampling_rate
        A = strategy.compute(u, None, dist, dispersion_config)
        
        # 频率范围应该合理
        # 检查结果形状
        assert A.shape == (dispersion_config.vnum, len(np.fft.fftfreq(u.shape[0], d=1/dispersion_config.sampling_rate)[:u.shape[0]//2]))
        assert not np.any(np.isnan(A))
        assert not np.any(np.isinf(A))


class TestDispersionFactory:
    """测试频散成像工厂类"""
    
    def test_create_all_strategies(self):
        """测试创建所有支持的策略"""
        methods = [
            DispersionMethod.FJ,
            DispersionMethod.FJ_RR,
            DispersionMethod.MFJ_RR,
            DispersionMethod.SLANT_STACK,
            DispersionMethod.MASW,
        ]
        
        for method in methods:
            strategy = DispersionFactory.create_strategy(method)
            assert isinstance(strategy, DispersionStrategy)
            assert hasattr(strategy, 'compute')
    
    def test_create_invalid_method(self):
        """测试创建无效方法"""
        with pytest.raises(ValueError, match="不支持的频散成像方法"):
            DispersionFactory.create_strategy("invalid_method")


class TestDispersionAnalyzer:
    """测试频散分析器主类"""
    
    def test_analyzer_initialization(self):
        """测试分析器初始化"""
        analyzer = DispersionAnalyzer(DispersionMethod.FJ)
        assert analyzer.method == DispersionMethod.FJ
        assert isinstance(analyzer.config, DispersionConfig)
        assert isinstance(analyzer.strategy, DispersionStrategy)
    
    def test_analyzer_with_custom_config(self):
        """测试带自定义配置的分析器"""
        config = DispersionConfig(
            freqmin=1.0,
            freqmax=8.0,
            vmin=800.0,
            vmax=2500.0,
            vnum=150
        )
        
        analyzer = DispersionAnalyzer(DispersionMethod.FJ_RR, config)
        assert analyzer.config.freqmin == 1.0
        assert analyzer.config.vnum == 150
    
    def test_analyzer_analyze_method(self, sample_dispersion_data, dispersion_config):
        """测试分析器的analyze方法"""
        analyzer = DispersionAnalyzer(DispersionMethod.FJ, dispersion_config)
        result = analyzer.analyze(
            sample_dispersion_data['cc_array_f'],
            sample_dispersion_data['f'],
            sample_dispersion_data['dist'],
            dispersion_config
        )
        
        # 检查结果形状
        expected_shape = (len(sample_dispersion_data['f']), dispersion_config.vnum)
        assert result.shape == expected_shape
    
    def test_analyzer_all_methods(self, sample_dispersion_data, dispersion_config):
        """测试分析器的所有方法"""
        methods = [
            DispersionMethod.FJ,
            DispersionMethod.FJ_RR,
            DispersionMethod.MFJ_RR,
            DispersionMethod.SLANT_STACK,
        ]
        
        for method in methods:
            analyzer = DispersionAnalyzer(method, dispersion_config)
            result = analyzer.analyze(
                sample_dispersion_data['cc_array_f'],
                sample_dispersion_data['f'],
                sample_dispersion_data['dist'],
                dispersion_config
            )
            
            assert result.shape == (len(sample_dispersion_data['f']), dispersion_config.vnum)
    
    def test_analyzer_masw_special_case(self, masw_time_domain_data, dispersion_config):
        """测试MASW方法的特殊处理"""
        analyzer = DispersionAnalyzer(DispersionMethod.MASW, dispersion_config)
        u = masw_time_domain_data['u']
        dist = masw_time_domain_data['dist']
        
        # MASW需要不同的调用方式
        A = analyzer.analyze(u, None, dist, dispersion_config)
        
        # 检查结果形状
        assert A.shape == (dispersion_config.vnum, len(np.fft.fftfreq(u.shape[0], d=1/dispersion_config.sampling_rate)[:u.shape[0]//2]))
        assert not np.any(np.isnan(A))
        assert not np.any(np.isinf(A))


class TestEdgeCases:
    """测试边界情况"""
    
    def test_empty_input_data(self, dispersion_config):
        """测试空输入数据"""
        empty_cc = np.array([])
        empty_f = np.array([])
        empty_dist = np.array([])
        
        strategies = [FJ(), FJ_RR(), MFJ_RR(), SlantStack()]
        
        for strategy in strategies:
            result = strategy.compute(empty_cc, empty_f, empty_dist, dispersion_config)
            # 应该返回空数组或适当处理
            assert len(result) == 0 or result.size == 0
    
    def test_single_station(self, dispersion_config):
        """测试单台站情况"""
        n_freq = 20
        f = np.linspace(0.1, 5.0, n_freq)
        dist = np.array([100.0])  # 单台站
        cc_array_f = np.random.randn(1, n_freq) + 1j * np.random.randn(1, n_freq)
        
        strategy = FJ()
        result = strategy.compute(cc_array_f, f, dist, dispersion_config)
        
        # 应该能处理单台站
        assert result.shape == (dispersion_config.vnum, n_freq)
    
    def test_single_frequency(self, sample_dispersion_data, dispersion_config):
        """测试单频率情况"""
        single_f = np.array([2.5])  # 单频率
        cc_array_f_single = sample_dispersion_data['cc_array_f'][:, 0:1]  # 单频率切片
        
        strategy = FJ()
        result = strategy.compute(cc_array_f_single, single_f, 
                                sample_dispersion_data['dist'], dispersion_config)
        
        # 应该能处理单频率
        assert result.shape == (dispersion_config.vnum, 1)
    
    def test_very_short_arrays(self, dispersion_config):
        """测试非常短的数组"""
        short_f = np.array([1.0, 2.0])
        short_dist = np.array([100.0, 200.0])
        short_cc = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        
        strategy = FJ()
        result = strategy.compute(short_cc, short_f, short_dist, dispersion_config)
        
        # 应该能处理短数组
        assert result.shape == (dispersion_config.vnum, len(short_f))


class TestPerformance:
    """测试性能"""
    
    def test_large_data_performance(self):
        """测试大数据性能"""
        # 创建较大的测试数据
        n_freq = 200
        n_stations = 20
        n_velocity = 100
        
        f = np.linspace(0.1, 10.0, n_freq)
        dist = np.linspace(100, 2000, n_stations)
        cc_array_f = np.random.randn(n_stations, n_freq) + 1j * np.random.randn(n_stations, n_freq)
        
        config = DispersionConfig(vnum=n_velocity)
        strategy = FJ()
        
        import time
        start_time = time.time()
        
        result = strategy.compute(cc_array_f, f, dist, config)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 应该在合理时间内完成
        assert processing_time < 30.0  # 30秒内完成
        assert result.shape == (n_velocity, n_freq)


class TestIntegration:
    """测试集成功能"""
    
    def test_full_workflow(self, sample_dispersion_data, dispersion_config):
        """测试完整工作流程"""
        # 创建分析器
        analyzer = DispersionAnalyzer(DispersionMethod.FJ, dispersion_config)
        
        # 执行分析
        spectrum = analyzer.analyze(
            sample_dispersion_data['cc_array_f'],
            sample_dispersion_data['f'],
            sample_dispersion_data['dist'],
            dispersion_config
        )
        
        # 检查结果
        assert spectrum.shape == (len(sample_dispersion_data['f']), dispersion_config.vnum)
        assert not np.any(np.isnan(spectrum))
        assert not np.any(np.isinf(spectrum))
        
        # 检查频散谱的基本特性
        assert np.max(spectrum) > 0  # 应该有正值
        assert np.min(spectrum) <= 0  # 可能有负值（取决于方法）
    
    def test_different_methods_comparison(self, sample_dispersion_data, dispersion_config):
        """测试不同方法的比较"""
        methods = [DispersionMethod.FJ, DispersionMethod.FJ_RR, DispersionMethod.SLANT_STACK]
        results = {}
        
        for method in methods:
            analyzer = DispersionAnalyzer(method, dispersion_config)
            spectrum = analyzer.analyze(
                sample_dispersion_data['cc_array_f'],
                sample_dispersion_data['f'],
                sample_dispersion_data['dist'],
                dispersion_config
            )
            results[method] = spectrum
        
        # 不同方法应该产生不同的结果
        assert not np.array_equal(results[DispersionMethod.FJ], results[DispersionMethod.FJ_RR])
        assert not np.array_equal(results[DispersionMethod.FJ], results[DispersionMethod.SLANT_STACK])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
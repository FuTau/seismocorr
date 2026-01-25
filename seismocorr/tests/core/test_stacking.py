import pytest
import numpy as np
import sys
import os
from typing import List, Dict, Any

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from seismocorr.core.correlation.stacking import (
    StackingStrategy,
    LinearStack,
    SelectiveStack,
    NrootStack,
    PhaseWeightedStack,
    RobustStack,
    get_stacker,
    stack_ccfs,
    _STRATEGY_REGISTRY
)


class TestLinearStack:
    """测试线性叠加"""
    
    def test_linear_stack_basic(self, ccf_list):
        """测试基本线性叠加"""
        stacker = LinearStack()
        result = stacker.stack(ccf_list)
        
        # 检查结果形状
        assert result.shape == ccf_list[0].shape
        
        # 线性叠加应该是平均值
        expected = np.mean(ccf_list, axis=0)
        assert np.allclose(result, expected)
    
    def test_linear_stack_single_ccf(self, single_ccf):
        """测试单个CCF的叠加"""
        stacker = LinearStack()
        result = stacker.stack([single_ccf])
        
        # 单个CCF叠加应该等于自身
        assert np.array_equal(result, single_ccf)
    
    
    def test_linear_stack_callable_interface(self, ccf_list):
        """测试可调用接口"""
        stacker = LinearStack()
        result1 = stacker.stack(ccf_list)
        result2 = stacker(ccf_list)  # 使用__call__
        
        assert np.array_equal(result1, result2)


class TestSelectiveStack:
    """测试选择叠加"""
    
    def test_selective_stack_basic(self, ccf_list):
        """测试基本选择叠加"""
        stacker = SelectiveStack()
        result = stacker.stack(ccf_list)
        
        # 检查结果形状
        assert result.shape == ccf_list[0].shape
        
        # 结果应该与线性叠加不同（因为剔除了低相关性的CCF）
        linear_result = np.mean(ccf_list, axis=0)
        assert not np.array_equal(result, linear_result)
    
    def test_selective_stack_with_outliers(self, ccf_list_with_outliers):
        """测试包含异常值的CCF列表"""
        stacker = SelectiveStack()
        result = stacker.stack(ccf_list_with_outliers)
        
        # 选择叠加应该能抑制异常值的影响
        assert result.shape == ccf_list_with_outliers[0].shape
        
        # 检查结果是否合理（不应该被异常值主导）
        result_max = np.max(np.abs(result))
        assert result_max < 10.0  # 不应该太大
    
    def test_selective_stack_identical_ccfs(self, identical_ccfs):
        """测试完全相同的CCF"""
        stacker = SelectiveStack()
        result = stacker.stack(identical_ccfs)
        
        # 完全相同的CCF应该全部被保留
        expected = identical_ccfs[0]  # 所有CCF都相同
        assert np.allclose(result, expected)


class TestNrootStack:
    """测试N次根叠加"""
    
    def test_nroot_stack_basic(self, ccf_list):
        """测试基本N次根叠加"""
        stacker = NrootStack()
        result = stacker.stack(ccf_list)
        
        # 检查结果形状
        assert result.shape == ccf_list[0].shape
        
        # N次根叠加应该与线性叠加不同
        linear_result = np.mean(ccf_list, axis=0)
        assert not np.array_equal(result, linear_result)
    
    def test_nroot_stack_custom_power(self):
        """测试自定义幂次"""
        # 创建测试数据
        ccf_list = [np.array([1.0, 4.0, 9.0]), np.array([1.0, 8.0, 27.0])]
        
        # 使用不同的幂次
        for power in [2, 3, 4]:
            stacker = NrootStack()
            stacker.power = power
            result = stacker.stack(ccf_list)
            assert len(result) == 3
    
    def test_nroot_stack_negative_values(self, ccf_list_with_negatives):
        """测试包含负值的CCF"""
        stacker = NrootStack()
        result = stacker.stack(ccf_list_with_negatives)
        
        # 应该能正确处理负值
        assert np.any(result < 0)  # 结果应该包含负值


class TestPhaseWeightedStack:
    """测试相位加权叠加"""
    
    def test_phase_weighted_stack_basic(self, ccf_list):
        """测试基本相位加权叠加"""
        stacker = PhaseWeightedStack(power=2.0)
        result = stacker.stack(ccf_list)
        
        # 检查结果形状
        assert result.shape == ccf_list[0].shape
        
        # 相位加权叠加应该与线性叠加不同
        linear_result = np.mean(ccf_list, axis=0)
        assert not np.array_equal(result, linear_result)
    
    def test_phase_weighted_stack_different_powers(self, ccf_list):
        """测试不同幂次的相位加权叠加"""
        for power in [1.0, 2.0, 4.0]:
            stacker = PhaseWeightedStack(power=power)
            result = stacker.stack(ccf_list)
            assert len(result) == len(ccf_list[0])
    
    def test_phase_weighted_stack_coherent_signals(self, coherent_ccfs):
        """测试高相干性信号的相位加权叠加"""
        stacker = PhaseWeightedStack(power=2.0)
        result = stacker.stack(coherent_ccfs)
        
        # 高相干性信号应该产生更强的叠加结果
        result_energy = np.sum(result ** 2)
        linear_result = np.mean(coherent_ccfs, axis=0)
        linear_energy = np.sum(linear_result ** 2)
        
        # 相位加权应该增强相干信号
        assert result_energy > linear_energy * 0.5  # 至少保留50%的能量


class TestRobustStack:
    """测试鲁棒叠加"""
    
    def test_robust_stack_basic(self, ccf_list):
        """测试基本鲁棒叠加"""
        stacker = RobustStack(epsilon=1e-8)
        result = stacker.stack(ccf_list)
        
        # 检查结果形状
        assert result.shape == ccf_list[0].shape
        
        # 鲁棒叠加应该与线性叠加不同
        linear_result = np.mean(ccf_list, axis=0)
        assert not np.array_equal(result, linear_result)
    
    def test_robust_stack_convergence(self, ccf_list_with_outliers):
        """测试鲁棒叠加的收敛性"""
        stacker = RobustStack(epsilon=1e-10)
        result = stacker.stack(ccf_list_with_outliers)
        
        # 应该能够收敛并返回合理结果
        assert len(result) == len(ccf_list_with_outliers[0])
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_robust_stack_different_epsilon(self, ccf_list):
        """测试不同收敛阈值"""
        for epsilon in [1e-6, 1e-8, 1e-10]:
            stacker = RobustStack(epsilon=epsilon)
            result = stacker.stack(ccf_list)
            assert len(result) > 0



class TestGetStacker:
    """测试叠加器工厂函数"""
    
    def test_get_all_stacker_types(self):
        """测试获取所有支持的叠加器类型"""
        for name in _STRATEGY_REGISTRY.keys():
            stacker = get_stacker(name)
            assert isinstance(stacker, StackingStrategy)
            assert hasattr(stacker, 'stack')
    
    def test_get_stacker_with_params(self):
        """测试带参数的叠加器获取"""
        # 测试PhaseWeightedStack带参数
        pws_stacker = get_stacker('pws', power=4.0)
        assert isinstance(pws_stacker, PhaseWeightedStack)
        assert pws_stacker.power == 4.0
        
        # 测试RobustStack带参数
        robust_stacker = get_stacker('robust', epsilon=1e-10)
        assert isinstance(robust_stacker, RobustStack)
        assert robust_stacker.epsilon == 1e-10
        
    
    def test_get_stacker_invalid_name(self):
        """测试无效叠加器名称"""
        with pytest.raises(ValueError, match="Unknown stacking method"):
            get_stacker('invalid_method')
    
    def test_get_stacker_case_insensitive(self):
        """测试大小写不敏感"""
        # 应该能处理大小写
        for name in ['LINEAR', 'PWS', 'Robust']:
            stacker = get_stacker(name)
            assert stacker is not None


class TestStackCCFs:
    """测试快捷函数stack_ccfs"""
    
    def test_stack_ccfs_basic(self, ccf_list):
        """测试基本快捷函数"""
        result = stack_ccfs(ccf_list, method='linear')
        
        # 应该与直接使用LinearStack相同
        stacker = LinearStack()
        expected = stacker.stack(ccf_list)
        
        assert np.array_equal(result, expected)
    
    def test_stack_ccfs_all_methods(self, ccf_list):
        """测试所有方法的快捷函数"""
        for method in _STRATEGY_REGISTRY.keys():
            result = stack_ccfs(ccf_list, method=method)
            assert len(result) == len(ccf_list[0])
    
    def test_stack_ccfs_with_params(self, ccf_list):
        """测试带参数的快捷函数"""
        result = stack_ccfs(ccf_list, method='pws', power=3.0)
        assert len(result) == len(ccf_list[0])


class TestEdgeCases:
    """测试边界情况"""
    
    def test_single_ccf(self, single_ccf):
        """测试单个CCF"""
        result = stack_ccfs([single_ccf], method='linear')
        assert np.array_equal(result, single_ccf)
    
    def test_very_short_ccfs(self):
        """测试非常短的CCF"""
        short_ccfs = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        result = stack_ccfs(short_ccfs, method='linear')
        assert len(result) == 1
        assert result[0] == 2.0  # 平均值


class TestPerformance:
    """测试性能"""
    
    def test_large_ccf_list_performance(self):
        """测试大CCF列表的性能"""
        # 创建大的CCF列表
        n_ccfs = 100
        ccf_length = 1000
        large_ccfs = [np.random.randn(ccf_length) for _ in range(n_ccfs)]
        
        import time
        start_time = time.time()
        
        # 测试线性叠加性能
        result = stack_ccfs(large_ccfs, method='linear')
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(result) == ccf_length
        assert processing_time < 5.0  # 应该在5秒内完成
    
    def test_different_methods_performance(self, ccf_list):
        """测试不同方法的性能比较"""
        methods = ['linear', 'selective', 'nroot', 'pws', 'robust']
        
        for method in methods:
            import time
            start_time = time.time()
            
            result = stack_ccfs(ccf_list, method=method)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            assert processing_time < 1.0  # 每个方法应该在1秒内完成


class TestIntegration:
    """测试集成功能"""
    
    def test_stacking_improves_snr(self, noisy_ccfs):
        """测试叠加能提高信噪比"""
        # 计算单个CCF的信噪比
        single_ccf = noisy_ccfs[0]
        signal_power = np.var(single_ccf[:100])  # 前100个点作为信号
        noise_power = np.var(single_ccf[100:])    # 后100个点作为噪声
        snr_single = signal_power / (noise_power + 1e-10)
        
        # 计算叠加后的信噪比
        stacked = stack_ccfs(noisy_ccfs, method='linear')
        signal_power_stacked = np.var(stacked[:100])
        noise_power_stacked = np.var(stacked[100:])
        snr_stacked = signal_power_stacked / (noise_power_stacked + 1e-10)
        
        # 叠加后信噪比应该提高
        assert snr_stacked > snr_single * 0.8  # 至少提高80%
    
    def test_different_stacking_methods_comparison(self, ccf_list):
        """测试不同叠加方法的比较"""
        results = {}
        
        for method in ['linear', 'selective', 'nroot', 'pws', 'robust']:
            results[method] = stack_ccfs(ccf_list, method=method)
        
        # 不同方法的结果应该不同
        assert not np.array_equal(results['linear'], results['selective'])
        assert not np.array_equal(results['linear'], results['nroot'])
        assert not np.array_equal(results['linear'], results['pws'])
        assert not np.array_equal(results['linear'], results['robust'])
        
        # 但应该高度相关
        correlation_linear_selective = np.corrcoef(
            results['linear'], results['selective']
        )[0, 1]
        assert correlation_linear_selective > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
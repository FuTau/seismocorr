# Seismocorr 代码规范

## 1. 项目概述

Seismocorr 是一个用于地震信号互相关分析的 Python 软件包，旨在提供便捷可用的 DAS（DAS-node）成像功能，集成了数据预处理、互相关计算、频散计算、空间相速度反演和反演等功能。

## 2. 命名规范

### 2.1 包和模块命名
- 包名使用小写字母，不使用下划线，如 `seismocorr`
- 模块名使用小写字母，单词之间用下划线分隔，如 `correlation.py`
- 避免使用缩写，除非是广为人知的缩写（如 `io.py`）

### 2.2 类命名
- 使用 PascalCase（大驼峰命名法），如 `StackingStrategy`
- 类名应清晰描述其功能，避免过于笼统
- 抽象基类应在名称中体现，如 `StackingStrategy` 暗示了它是一个策略基类

### 2.3 函数和方法命名
- 使用 snake_case（蛇形命名法），如 `compute_cross_correlation`
- 函数名应包含动词，清晰描述其功能
- 私有函数和方法应以单下划线开头，如 `_xcorr_time_domain`
- 避免使用过于简短的函数名，应清晰表达其功能

### 2.4 变量命名
- 使用 snake_case（蛇形命名法），如 `sampling_rate`
- 变量名应清晰描述其含义，避免使用单个字母（除非在循环中）
- 常量使用全大写，单词之间用下划线分隔，如 `SUPPORTED_METHODS`
- 类型别名使用 PascalCase，如 `ArrayLike`

### 2.5 文件命名
- 数据文件：使用有意义的名称，包含关键参数，如 `FDLY_916m_4m_1m_1000Hz_1000Hz_UTC8_202409191200.h5`
- 脚本文件：使用蛇形命名法，清晰描述其功能，如 `do_disper.py`
- 测试文件：使用 `test_` 前缀加模块名，如 `test_correlation.py`

## 3. 代码风格

### 3.1 缩进和换行
- 使用 4 个空格进行缩进，不使用制表符
- 每行长度不超过 100 个字符
- 函数和类定义之间使用两个空行分隔
- 函数内部逻辑块之间使用一个空行分隔
- 避免不必要的换行，保持代码的可读性

### 3.2 导入规范
- 导入顺序：
  1. 标准库导入（如 `numpy`, `typing`）
  2. 第三方库导入（如 `scipy`）
  3. 本地模块导入（如 `from seismocorr.preprocessing import ...`）
- 每个导入组之间使用一个空行分隔
- 避免使用 `from module import *` 形式的导入
- 对于较长的导入语句，可以使用括号换行

### 3.3 注释
- 使用中文注释，确保所有开发者都能理解
- 函数和类必须有文档字符串（docstring），描述其功能、参数和返回值
- 复杂的算法或逻辑应有注释说明
- 注释应清晰、简洁，避免冗余
- 使用 `#` 进行单行注释，注释与代码之间至少有一个空格

### 3.4 文档字符串格式
- 使用 Google 风格的文档字符串
- 包含以下部分：
  - 函数/类的简要描述
  - Args：参数列表，包括参数名、类型和描述
  - Returns：返回值类型和描述
  - Raises：可能抛出的异常
  - Example：使用示例（可选）

示例：
```python
def compute_cross_correlation(
    x: ArrayLike,
    y: ArrayLike,
    sampling_rate: float,
    method: str = 'time-domain',
    time_normalize: str = 'one-bit',
    freq_normalize: str = 'no',
    freq_band: Optional[Tuple[float, float]] = None,
    max_lag: Optional[Union[float, int]] = None,
    nfft: Optional[int] = None,
    time_norm_kwargs: Optional[Dict[str, Any]] = None,
    freq_norm_kwargs: Optional[Dict[str, Any]] = None,
) -> LagsAndCCF:
    """
    计算两个时间序列的互相关函数（CCF）

    Args:
        x, y: 时间序列数据
        sampling_rate: 采样率 (Hz)
        method: 计算方法 ('time-domain', 'freq-domain', 'deconv', 'coherency')
        time_normalize: 时域归一化方式
        freq_normalize: 频域归一化方式
        freq_band: 带通滤波范围 (fmin, fmax)，单位 Hz
        max_lag: 最大滞后时间（秒）；若为 None，则使用 min(len(x), len(y))
        nfft: FFT 长度，自动补零到 next_fast_len
        time_norm_kwargs: 时域归一化参数
        freq_norm_kwargs: 频域归一化参数

    Returns:
        lags: 时间滞后数组 (单位：秒)
        ccf: 互相关函数值
    """
```

## 4. 类型提示

- 所有公共函数和方法必须使用类型提示
- 使用 `typing` 模块提供的类型，如 `Union`, `Optional`, `List`, `Dict`, `Tuple` 等
- 对于复杂类型，定义类型别名，提高代码可读性
- 示例：
  ```python
  ArrayLike = Union[np.ndarray, List[float]]
  LagsAndCCF = Tuple[np.ndarray, np.ndarray]
  BatchResult = Dict[str, LagsAndCCF]  # {channel_pair: (lags, ccf)}
  ```

## 5. 模块化设计

### 5.1 模块划分原则
- 每个模块应专注于单一功能
- 模块之间的依赖关系应清晰、简单
- 避免循环依赖
- 核心功能与辅助功能分离

### 5.2 目录结构
```
seismocorr/
├── config/          # 配置管理
├── core/            # 核心计算功能
├── pipline/         # 流水线处理
├── plugins/         # 插件功能
├── preprocessing/   # 数据预处理
├── tests/           # 测试代码
├── utils/           # 工具函数
└── __init__.py      # 包初始化
```

### 5.3 接口设计
- 公共函数和类应提供清晰、稳定的接口
- 避免暴露内部实现细节
- 使用工厂模式（如 `get_stacker`, `get_time_normalizer`）提供对象创建接口
- 接口参数应合理，避免过多参数

## 6. 测试规范

### 6.1 测试框架
- 使用 `pytest` 进行测试
- 测试文件放在 `tests/` 目录下，与源代码结构对应
- 测试文件名使用 `test_` 前缀，如 `test_correlation.py`

### 6.2 测试覆盖
- 核心功能的测试覆盖率应达到 80% 以上
- 测试应包括正常情况和异常情况
- 测试应具有可重复性，避免依赖外部资源

### 6.3 测试用例设计
- 每个测试用例应测试单一功能
- 测试用例名称应清晰描述测试内容
- 使用 `conftest.py` 提供测试夹具（fixtures）

## 7. 性能考虑

### 7.1 并行计算
- 对于计算密集型任务，使用并行计算提高性能
- 并行实现应支持两种后端：线程池（适合 I/O 密集型）和进程池（适合 CPU 密集型）
- 并行计算应提供参数控制并行度（如 `n_jobs`）
- 示例：
  ```python
def batch_cross_correlation(
    traces: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str]],
    sampling_rate: float,
    n_jobs: int = -1,
    parallel_backend: str = "thread",  # "process" 或 "thread"
    **kwargs
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    # 实现并行计算
    pass
  ```

### 7.2 内存管理
- 对于大规模数据处理，注意内存使用
- 避免不必要的数据复制
- 使用生成器或迭代器处理流式数据

### 7.3 算法优化
- 选择高效的算法实现
- 对于频繁调用的函数，进行性能优化
- 使用合适的库函数（如 `scipy` 中的优化函数）

## 8. 异常处理

### 8.1 异常类型
- 使用适当的异常类型，如 `ValueError`, `TypeError` 等
- 避免使用过于宽泛的异常（如 `Exception`）

### 8.2 异常消息
- 异常消息应清晰、具体，说明错误原因
- 示例：
  ```python
  raise ValueError(f"Unsupported method: {method}. Choose from {SUPPORTED_METHODS}")
  ```

### 8.3 异常处理
- 对于可能出现的异常，进行适当的处理
- 避免静默失败，应记录错误信息
- 示例：
  ```python
try:
    lags, ccf = compute_cross_correlation(traces[a], traces[b], sampling_rate, **kwargs)
    key = f"{a}--{b}"
    result[key] = (lags, ccf)
except Exception as e:
    print(f"Failed on pair {a}-{b}: {e}")
  ```

## 9. 代码审查

### 9.1 审查要点
- 代码是否符合命名规范和风格指南
- 函数和类的设计是否合理
- 接口是否清晰、稳定
- 注释是否完整、准确
- 测试覆盖率是否足够
- 性能是否满足要求
- 异常处理是否适当

### 9.2 审查流程
- 代码提交前进行自我审查
- 提交后进行团队审查
- 审查意见应具体、建设性
- 及时修复审查中发现的问题

## 10. 版本控制

### 10.1 Git 规范
- 提交信息应清晰、简洁，描述本次提交的主要内容
- 提交信息格式：`[类型] 描述`，如 `[FEATURE] 添加频域归一化功能`
- 类型包括：
  - `FEATURE`：新功能
  - `FIX`：修复bug
  - `REFACTOR`：代码重构
  - `DOCS`：文档更新
  - `TEST`：测试相关
  - `CI`：CI/CD 相关

### 10.2 分支管理
- 使用 `main` 分支作为稳定分支
- 开发新功能使用 `feature/` 前缀的分支
- 修复bug使用 `fix/` 前缀的分支
- 分支命名应清晰描述分支目的

## 11. 文档

### 11.1 文档类型
- 代码注释：解释代码功能和实现细节
- 文档字符串：描述函数、类的接口和使用方法
- README.md：项目概述、安装和使用说明
- CODE_STYLE.md：代码规范
- Optimization.md：性能优化说明

### 11.2 文档更新
- 代码变更时，同步更新相关文档
- 确保文档的准确性和完整性
- 使用清晰、简洁的语言

## 12. 总结

本代码规范旨在确保 Seismocorr 项目的代码具有高可读性、可维护性和可扩展性。所有开发者应严格遵守本规范，共同提高项目的代码质量。

代码规范应随着项目的发展不断更新和完善，以适应新的需求和技术发展。
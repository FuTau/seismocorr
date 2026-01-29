#seismocorr/plugins/processing/cluster_filter.py
"""
聚类筛选器模块（基于组合距离进行样本筛选）。

数据约定：
    - lags: (n_lags,)，延时范围。
    - ccfs: (n_samples, n_lags)，Cross-Correlation Functions（CCFs）矩阵。
    - keys: (n_samples,)，样本的标识符（可选）。

功能：
    - ClusterFilter：根据给定的组合距离度量（包括 EMD 和 Energy Distance），
      对多个样本的 CCFs 进行筛选，选出符合条件的稳定片段。
      输出经过筛选后的延时（lags）、CCFs 以及样本标识符。
    
    - 核心方法：
      - fit：计算距离矩阵、生成排序 order、计算到时 arrival_times，并选择稳定片段的索引。
      - transform：返回筛选后的 CCFs（支持不重复传入 lags/ccfs）。
      - filter：结合 fit 和 transform 的一站式接口，执行整个筛选过程。
      - filter_to_list：返回 stacking.py 所需的 CCF 列表。
      
    - 支持的计算：
      - 使用 EMD 和 Energy distance 对样本进行匹配度量，构造距离矩阵。
      - 基于贪心算法对距离矩阵进行排序，尽量将相似的样本排列在一起。
      - 可选的到时计算：使用包络（Hilbert）方法计算 CCF 的到时。
      - 基于设定的百分比选择筛选后的样本。
"""


import numpy as np
from typing import Optional, Tuple, Sequence, List, Any, Dict


class ClusterFilter:
    """
    “集相邻/排序”滤波器：返回处理后的预叠加片段（不做叠加）。

    - fit: 计算距离矩阵、生成排序 order、计算到时 arrival_times，并选择稳定片段索引
    - transform: 返回筛选后的 ccfs（支持不重复传入 lags/ccfs）
    - filter: fit + transform 的一站式接口
    - filter_to_list: 一站式返回 stacking.py 需要的 ccf_list
    """

    def __init__(
        self,
        lag_window: Optional[Tuple[float, float]] = (-1.5, 1.5),
        add_constant: float = 1e-6,
        emd_weight: float = 1.0,
        energy_weight: float = 1.0,
        select_percentile: Tuple[float, float] = (0.55, 0.85),
        arrival_window: Optional[Tuple[float, float]] = None,
        arrival_use_envelope: bool = True,
    ):
        self.lag_window = lag_window
        self.add_constant = add_constant
        self.emd_weight = emd_weight
        self.energy_weight = energy_weight
        self.select_percentile = select_percentile
        self.arrival_window = arrival_window
        self.arrival_use_envelope = arrival_use_envelope
        self.order_ = None
        self.arrival_times_ = None
        self.selected_indices_ = None
        self.distance_matrix_ = None
        self.info_ = None
        self._lags_cache = None
        self._ccfs_cache = None
        self._keys_cache = None

    @staticmethod
    def as_ccf_list(ccfs_2d: np.ndarray) -> List[np.ndarray]:
        """
        将二维数组 (N, n_lags) 转为 stacking.py 需要的 List[np.ndarray]。
        """
        x = np.asarray(ccfs_2d)
        if x.ndim != 2:
            raise ValueError(f"ccfs_2d 必须为二维数组 (N, n_lags)，当前 shape={x.shape}")
        return [np.asarray(row) for row in x]

    @staticmethod
    def analytic_envelope(segments: np.ndarray) -> np.ndarray:
        """
        使用 FFT 形式 Hilbert 变换计算包络（不依赖 scipy）。
        segments: shape=(n_segments, n_samples)
        return: envelope, shape 同输入
        """
        x = np.asarray(segments, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"segments 必须为二维数组，当前 shape={x.shape}")

        n = x.shape[1]
        X = np.fft.fft(x, axis=1)

        h = np.zeros(n, dtype=float)
        if n % 2 == 0:
            h[0] = 1.0
            h[n // 2] = 1.0
            h[1:n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(n + 1) // 2] = 2.0

        Z = X * h[None, :]
        z = np.fft.ifft(Z, axis=1)
        return np.abs(z)

    @staticmethod
    def _to_nonnegative_distribution(
        x: np.ndarray,
        eps: float = 1e-12,
        add_constant: float = 1e-6,
    ) -> np.ndarray:
        """
        将 1D 波形转换为非负“分布”（用于 EMD / Energy distance）：
        1) 去最小值 -> 非负
        2) 统一加常数 -> 避免零值造成数值不稳定
        3) 归一化到 sum=1
        """
        v = np.asarray(x, dtype=float).copy()
        v -= np.min(v)
        v += float(add_constant)
        s = np.sum(v)
        if s <= eps:
            v = np.ones_like(v, dtype=float)
            v /= np.sum(v)
            return v
        v /= s
        return v

    @staticmethod
    def emd_1d_wasserstein1(p: np.ndarray, q: np.ndarray, dx: float = 1.0) -> float:
        """
        1D Earth Mover's Distance（等价于 Wasserstein-1）：
        EMD = ∫ |CDF_p - CDF_q| dx
        离散等距采样时：
        EMD ≈ sum_i |cdf_p[i] - cdf_q[i]| * dx
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        if p.ndim != 1 or q.ndim != 1 or p.size != q.size:
            raise ValueError("p, q 必须为同长度的一维数组")
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        return float(np.sum(np.abs(cdf_p - cdf_q)) * dx)

    @staticmethod
    def energy_distance_1d(p: np.ndarray, q: np.ndarray, dx: float = 1.0) -> float:
        """
        1D Energy distance（基于 CDF 差异的形式）： 
        D = sqrt( 2 * ∫ (F-G)^2 dx )
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        if p.ndim != 1 or q.ndim != 1 or p.size != q.size:
            raise ValueError("p, q 必须为同长度的一维数组")
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        val = 2.0 * np.sum((cdf_p - cdf_q) ** 2) * dx
        return float(np.sqrt(max(val, 0.0)))

    @staticmethod
    def combined_distance(
        a: np.ndarray,
        b: np.ndarray,
        dx: float = 1.0,
        add_constant: float = 1e-6,
        emd_weight: float = 1.0,
        energy_weight: float = 1.0,
    ) -> float:
        """
        组合距离：emd_weight*EMD + energy_weight*EnergyDistance
        """
        pa = ClusterFilter._to_nonnegative_distribution(a, add_constant=add_constant)
        pb = ClusterFilter._to_nonnegative_distribution(b, add_constant=add_constant)
        d1 = ClusterFilter.emd_1d_wasserstein1(pa, pb, dx=dx)
        d2 = ClusterFilter.energy_distance_1d(pa, pb, dx=dx)
        return float(emd_weight * d1 + energy_weight * d2)

    def pairwise_distance_matrix(
        self,
        X: np.ndarray,
        dx: float = 1.0,
        add_constant: float = 1e-6,
        emd_weight: float = 1.0,
        energy_weight: float = 1.0,
    ) -> np.ndarray:
        """
        计算两两距离矩阵 D，D[i,j] = 组合距离(EMD, Energy distance)。
        X: shape=(N, M)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X 必须为二维数组 (N, M)，当前 shape={X.shape}")
        N = X.shape[0]
        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                d = ClusterFilter.combined_distance(
                    X[i],
                    X[j],
                    dx=dx,
                    add_constant=add_constant,
                    emd_weight=emd_weight,
                    energy_weight=energy_weight,
                )
                D[i, j] = d
                D[j, i] = d
        return D

    def greedy_end_extension_order(self, D: np.ndarray) -> np.ndarray:
        """
        贪心两端扩展构造序列，使相似样本尽量相邻。
        返回：order（长度 N 的索引序列）
        """
        D = np.asarray(D, dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("D 必须为方阵")
        N = D.shape[0]
        if N == 0:
            return np.array([], dtype=int)
        if N == 1:
            return np.array([0], dtype=int)

        start = int(np.argmin(np.mean(D, axis=1)))
        visited = np.zeros(N, dtype=bool)
        visited[start] = True

        order = [start]
        left = start
        right = start

        for _ in range(N - 1):
            unvisited = np.where(~visited)[0]
            dl = D[left, unvisited]
            dr = D[right, unvisited]

            i_l = int(unvisited[int(np.argmin(dl))])
            i_r = int(unvisited[int(np.argmin(dr))])

            if D[left, i_l] <= D[right, i_r]:
                order.insert(0, i_l)
                visited[i_l] = True
                left = i_l
            else:
                order.append(i_r)
                visited[i_r] = True
                right = i_r

        return np.array(order, dtype=int)
    
    def fit(
        self,
        lags: np.ndarray,
        ccfs: np.ndarray,
        keys: Optional[Sequence[str]] = None,
    ) -> "ClusterFilter":
        """
        计算距离矩阵并完成排序，保存 order_ / selected_indices_ / arrival_times_ 等信息。
        同时缓存本次输入，便于 transform() 直接调用。
        """
        lags, ccfs = self._validate_inputs(lags, ccfs)
        if keys is not None and len(keys) != ccfs.shape[0]:
            raise ValueError("keys 长度必须与 ccfs 行数一致。")

        self._lags_cache = lags.copy()
        self._ccfs_cache = ccfs.copy()
        self._keys_cache = list(keys) if keys is not None else None

        win = self._window_slice(lags)
        Xwin = ccfs[:, win]

        if lags.size >= 2:
            dx = float(np.median(np.diff(lags)))
            dx = abs(dx) if dx > 0 else 1.0
        else:
            dx = 1.0

        D = self.pairwise_distance_matrix(
            Xwin,
            dx=dx,
            add_constant=self.add_constant,
            emd_weight=self.emd_weight,
            energy_weight=self.energy_weight,
        )
        order = self.greedy_end_extension_order(D)

        arr = self.batch_causal_arrival_times(
            lags,
            ccfs,
            search_window=self.arrival_window,
            use_envelope=self.arrival_use_envelope,
        )

        p0, p1 = self.select_percentile
        p0 = float(np.clip(p0, 0.0, 1.0))
        p1 = float(np.clip(p1, 0.0, 1.0))
        if p0 > p1:
            p0, p1 = p1, p0

        N = order.size
        i0 = int(np.floor(p0 * N))
        i1 = int(np.ceil(p1 * N))
        i0 = max(0, min(N - 1, i0))
        i1 = max(i0 + 1, min(N, i1))
        selected = order[i0:i1]

        self.distance_matrix_ = D
        self.order_ = order
        self.arrival_times_ = arr
        self.selected_indices_ = selected

        self.info_ = {
            "lag_window": self.lag_window,
            "distance": {
                "emd_weight": self.emd_weight,
                "energy_weight": self.energy_weight,
                "add_constant": self.add_constant,
            },
            "sequencing": {
                "method": "greedy_end_extension",
                "n_items": int(N),
            },
            "selection": {
                "select_percentile": (p0, p1),
                "selected_count": int(selected.size),
                "selected_index_range_in_order": (int(i0), int(i1)),
            },
            "arrival": {
                "arrival_window": self.arrival_window,
                "use_envelope": bool(self.arrival_use_envelope),
            },
        }
        return self

    def transform(
        self,
        lags: Optional[np.ndarray] = None,
        ccfs: Optional[np.ndarray] = None,
        keys: Optional[Sequence[str]] = None,
        return_info: bool = False,
        return_ordered: bool = False,
    ):
        """
        返回筛选后的预叠加片段。

        - 若不传 lags/ccfs/keys，则使用 fit() 缓存的输入
        - return_info=True 时返回 info 字典
        - return_ordered=True 时在 info 中附加 ordered_ccfs 与 ordered_keys
        """
        if self.order_ is None or self.selected_indices_ is None or self.info_ is None:
            raise RuntimeError("请先调用 fit(...) 再 transform(...).")

        if lags is None or ccfs is None:
            if self._lags_cache is None or self._ccfs_cache is None:
                raise ValueError("未提供 lags/ccfs，且没有可用缓存。请先 fit(...) 或直接传入 lags/ccfs。")
            lags = self._lags_cache
            ccfs = self._ccfs_cache

        lags, ccfs = self._validate_inputs(lags, ccfs)

        if keys is None:
            keys = self._keys_cache
        if keys is not None and len(keys) != ccfs.shape[0]:
            raise ValueError("keys 长度必须与 ccfs 行数一致。")

        idx = self.selected_indices_
        out_lags = lags.copy()
        out_ccfs = ccfs[idx].copy()
        out_keys = [keys[i] for i in idx] if keys is not None else None

        if not return_info and not return_ordered:
            return out_lags, out_ccfs, out_keys

        info = dict(self.info_)
        info["order"] = self.order_.copy()
        info["selected_indices"] = idx.copy()

        if self.arrival_times_ is not None:
            info["arrival_times"] = self.arrival_times_.copy()
            info["selected_arrival_times"] = self.arrival_times_[idx].copy()

        if return_ordered:
            info["ordered_ccfs"] = ccfs[self.order_].copy()
            info["ordered_keys"] = [keys[i] for i in self.order_] if keys is not None else None

        return out_lags, out_ccfs, out_keys, info

    def to_ccf_list(
        self,
        lags: Optional[np.ndarray] = None,
        ccfs: Optional[np.ndarray] = None,
        keys: Optional[Sequence[str]] = None,
        return_info: bool = False,
    ):
        """
        返回 stacking.py 可用的 ccf_list（筛选后的预叠加片段列表）。
        """
        if return_info:
            out_lags, out_ccfs, out_keys, info = self.transform(
                lags=lags, ccfs=ccfs, keys=keys, return_info=True, return_ordered=False
            )
            return self.as_ccf_list(out_ccfs), out_keys, info
        out_lags, out_ccfs, out_keys = self.transform(lags=lags, ccfs=ccfs, keys=keys)
        return self.as_ccf_list(out_ccfs)

    def filter(
        self,
        lags: np.ndarray,
        ccfs: np.ndarray,
        keys: Optional[Sequence[str]] = None,
        return_info: bool = False,
        return_ordered: bool = False,
    ):
        """
        一站式接口：fit + transform。
        """
        self.fit(lags, ccfs, keys=keys)
        return self.transform(return_info=return_info, return_ordered=return_ordered)

    def filter_to_list(
        self,
        lags: np.ndarray,
        ccfs: np.ndarray,
        keys: Optional[Sequence[str]] = None,
        return_info: bool = False,
    ):
        """
        一站式接口：返回 stacking.py 可用的 ccf_list。
        """
        self.fit(lags, ccfs, keys=keys)
        return self.to_ccf_list(return_info=return_info)

    def __call__(
        self,
        lags: np.ndarray,
        ccfs: np.ndarray,
        keys: Optional[Sequence[str]] = None,
    ):
        """
        允许直接把对象当函数用：等价于 filter(lags, ccfs, keys) 的简化返回（三元组）。
        """
        return self.filter(lags, ccfs, keys=keys, return_info=False, return_ordered=False)

    def info(self) -> Dict[str, Any]:
        """
        返回 fit 过程中记录的信息字典。
        """
        if self.info_ is None:
            return {}
        return dict(self.info_)
   

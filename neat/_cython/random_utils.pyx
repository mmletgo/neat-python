# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
高性能随机数生成缓存

使用 Cython 优化的批量随机数生成，避免频繁调用 Python random() 的开销。
主要优化：
1. 预生成大量随机数到缓存
2. 获取随机数时直接从缓存读取（使用 typed memoryview）
3. 缓存耗尽时自动批量重新填充
4. 提供无锁的单线程快速版本
"""

import numpy as np
cimport numpy as np

# NumPy 类型声明
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# 默认缓存大小
DEF DEFAULT_CACHE_SIZE = 100000


cdef class RandomCache:
    """
    随机数缓存类

    预生成 uniform [0,1) 和 gaussian 随机数缓存，
    提供高效的单值和批量获取接口。

    注意：此类不是线程安全的。在多线程环境中，
    请为每个线程创建独立的 RandomCache 实例。
    """

    # 缓存数组
    cdef np.ndarray _uniform_cache
    cdef np.ndarray _gaussian_cache

    # typed memoryview 用于快速访问
    cdef DTYPE_t[::1] uniform_view
    cdef DTYPE_t[::1] gaussian_view

    # 缓存配置
    cdef int cache_size

    # 当前索引
    cdef int uniform_idx
    cdef int gaussian_idx

    def __cinit__(self, int cache_size=DEFAULT_CACHE_SIZE):
        """初始化缓存"""
        self.cache_size = cache_size
        self.uniform_idx = 0
        self.gaussian_idx = 0

        # 初始化缓存数组
        self._uniform_cache = np.empty(cache_size, dtype=DTYPE)
        self._gaussian_cache = np.empty(cache_size, dtype=DTYPE)

        # 创建 memoryview
        self.uniform_view = self._uniform_cache
        self.gaussian_view = self._gaussian_cache

        # 填充初始缓存
        self._refill_uniform()
        self._refill_gaussian()

    cdef inline void _refill_uniform(self) noexcept:
        """重新填充均匀分布缓存"""
        self._uniform_cache[:] = np.random.random(self.cache_size)
        self.uniform_idx = 0

    cdef inline void _refill_gaussian(self) noexcept:
        """重新填充高斯分布缓存"""
        self._gaussian_cache[:] = np.random.standard_normal(self.cache_size)
        self.gaussian_idx = 0

    cdef inline double _get_uniform_fast(self) noexcept:
        """
        内联获取单个均匀分布随机数（最快，无边界检查）

        调用者需确保在调用前检查缓存是否足够。
        """
        cdef double value = self.uniform_view[self.uniform_idx]
        self.uniform_idx += 1
        return value

    cdef inline double _get_gaussian_fast(self) noexcept:
        """
        内联获取单个高斯分布随机数（最快，无边界检查）

        调用者需确保在调用前检查缓存是否足够。
        """
        cdef double value = self.gaussian_view[self.gaussian_idx]
        self.gaussian_idx += 1
        return value

    cpdef double get_uniform(self):
        """
        获取单个 [0, 1) 均匀分布随机数

        Returns:
            double: 均匀分布随机数
        """
        if self.uniform_idx >= self.cache_size:
            self._refill_uniform()
        return self._get_uniform_fast()

    cpdef double get_gaussian(self, double mean=0.0, double std=1.0):
        """
        获取单个高斯分布随机数

        Args:
            mean: 均值，默认 0.0
            std: 标准差，默认 1.0

        Returns:
            double: 高斯分布随机数
        """
        if self.gaussian_idx >= self.cache_size:
            self._refill_gaussian()
        return mean + std * self._get_gaussian_fast()

    cpdef np.ndarray get_batch_uniform_array(self, int n):
        """
        批量获取 n 个 [0, 1) 均匀分布随机数

        Args:
            n: 需要的随机数数量

        Returns:
            np.ndarray: 包含 n 个均匀分布随机数的数组
        """
        cdef np.ndarray result
        cdef int available
        cdef int remaining

        if n <= 0:
            return np.empty(0, dtype=DTYPE)

        # 如果请求数量超过缓存大小，直接生成
        if n > self.cache_size:
            return np.random.random(n).astype(DTYPE)

        available = self.cache_size - self.uniform_idx

        if n <= available:
            # 缓存足够，直接切片复制
            result = self._uniform_cache[self.uniform_idx:self.uniform_idx + n].copy()
            self.uniform_idx += n
            return result
        else:
            # 缓存不足，需要组合
            result = np.empty(n, dtype=DTYPE)

            if available > 0:
                # 先取剩余的缓存
                result[:available] = self._uniform_cache[self.uniform_idx:]

            # 重新填充缓存
            self._refill_uniform()

            # 取剩余需要的数量
            remaining = n - available
            result[available:] = self._uniform_cache[:remaining]
            self.uniform_idx = remaining

            return result

    cpdef np.ndarray get_batch_gaussian_array(self, int n, double mean=0.0, double std=1.0):
        """
        批量获取 n 个高斯分布随机数

        Args:
            n: 需要的随机数数量
            mean: 均值，默认 0.0
            std: 标准差，默认 1.0

        Returns:
            np.ndarray: 包含 n 个高斯分布随机数的数组
        """
        cdef np.ndarray result
        cdef int available
        cdef int remaining

        if n <= 0:
            return np.empty(0, dtype=DTYPE)

        # 如果请求数量超过缓存大小，直接生成
        if n > self.cache_size:
            result = np.random.standard_normal(n).astype(DTYPE)
            if mean != 0.0 or std != 1.0:
                result = mean + std * result
            return result

        available = self.cache_size - self.gaussian_idx

        if n <= available:
            # 缓存足够，直接切片复制
            result = self._gaussian_cache[self.gaussian_idx:self.gaussian_idx + n].copy()
            self.gaussian_idx += n
        else:
            # 缓存不足，需要组合
            result = np.empty(n, dtype=DTYPE)

            if available > 0:
                # 先取剩余的缓存
                result[:available] = self._gaussian_cache[self.gaussian_idx:]

            # 重新填充缓存
            self._refill_gaussian()

            # 取剩余需要的数量
            remaining = n - available
            result[available:] = self._gaussian_cache[:remaining]
            self.gaussian_idx = remaining

        # 应用均值和标准差变换
        if mean != 0.0 or std != 1.0:
            result = mean + std * result

        return result

    cpdef void reset(self):
        """
        重置缓存（用于可重复性测试）

        重新填充所有缓存并重置索引。
        """
        self._refill_uniform()
        self._refill_gaussian()


# ============================================================================
# 全局缓存实例（单线程使用）
# ============================================================================

# 全局缓存实例
cdef RandomCache _global_cache = RandomCache(DEFAULT_CACHE_SIZE)

# 全局缓存的 memoryview 引用（用于超快速访问）
cdef DTYPE_t[::1] _uniform_view = _global_cache._uniform_cache
cdef DTYPE_t[::1] _gaussian_view = _global_cache._gaussian_cache


# ============================================================================
# 模块级函数接口
# ============================================================================

cpdef double fast_random():
    """
    获取单个 [0, 1) 均匀分布随机数

    使用全局缓存实例，比 Python random.random() 更快。

    注意：此函数不是线程安全的。

    Returns:
        double: 均匀分布随机数

    Example:
        >>> value = fast_random()
        >>> 0.0 <= value < 1.0
        True
    """
    global _uniform_view

    if _global_cache.uniform_idx >= _global_cache.cache_size:
        _global_cache._refill_uniform()
        _uniform_view = _global_cache._uniform_cache

    cdef double value = _uniform_view[_global_cache.uniform_idx]
    _global_cache.uniform_idx += 1
    return value


cpdef double fast_gauss(double mean=0.0, double std=1.0):
    """
    获取单个高斯分布随机数

    使用全局缓存实例，比 Python random.gauss() 更快。

    注意：此函数不是线程安全的。

    Args:
        mean: 均值，默认 0.0
        std: 标准差，默认 1.0

    Returns:
        double: 高斯分布随机数

    Example:
        >>> value = fast_gauss(0.0, 1.0)
        >>> isinstance(value, float)
        True
    """
    global _gaussian_view

    if _global_cache.gaussian_idx >= _global_cache.cache_size:
        _global_cache._refill_gaussian()
        _gaussian_view = _global_cache._gaussian_cache

    cdef double value = _gaussian_view[_global_cache.gaussian_idx]
    _global_cache.gaussian_idx += 1
    return mean + std * value


cpdef np.ndarray get_batch_uniform(int n):
    """
    批量获取 n 个 [0, 1) 均匀分布随机数

    当需要多个随机数时，批量获取比多次调用 fast_random() 更高效。

    Args:
        n: 需要的随机数数量

    Returns:
        np.ndarray: 包含 n 个均匀分布随机数的 float64 数组

    Example:
        >>> arr = get_batch_uniform(100)
        >>> arr.shape
        (100,)
        >>> arr.dtype
        dtype('float64')
    """
    return _global_cache.get_batch_uniform_array(n)


cpdef np.ndarray get_batch_gaussian(int n, double mean=0.0, double std=1.0):
    """
    批量获取 n 个高斯分布随机数

    当需要多个随机数时，批量获取比多次调用 fast_gauss() 更高效。

    Args:
        n: 需要的随机数数量
        mean: 均值，默认 0.0
        std: 标准差，默认 1.0

    Returns:
        np.ndarray: 包含 n 个高斯分布随机数的 float64 数组

    Example:
        >>> arr = get_batch_gaussian(100, mean=0.0, std=1.0)
        >>> arr.shape
        (100,)
    """
    return _global_cache.get_batch_gaussian_array(n, mean, std)


cpdef void reset_random_cache():
    """
    重置全局随机数缓存

    用于测试时确保可重复性。在设置随机种子后调用此函数，
    可以确保后续的随机数序列是确定的。

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> reset_random_cache()
        >>> # 现在随机数序列是确定的
    """
    global _uniform_view, _gaussian_view
    _global_cache.reset()
    _uniform_view = _global_cache._uniform_cache
    _gaussian_view = _global_cache._gaussian_cache


# ============================================================================
# 额外的便捷函数
# ============================================================================

cpdef bint fast_random_choice(double probability):
    """
    根据概率返回布尔值

    等价于 random.random() < probability，但更快。

    Args:
        probability: 返回 True 的概率，范围 [0, 1]

    Returns:
        bint: True 或 False

    Example:
        >>> # 50% 概率返回 True
        >>> result = fast_random_choice(0.5)
    """
    return fast_random() < probability


cpdef double fast_uniform(double low, double high):
    """
    获取 [low, high) 范围内的均匀分布随机数

    Args:
        low: 下界（包含）
        high: 上界（不包含）

    Returns:
        double: 均匀分布随机数

    Example:
        >>> value = fast_uniform(-1.0, 1.0)
        >>> -1.0 <= value < 1.0
        True
    """
    return low + (high - low) * fast_random()


cpdef np.ndarray get_batch_uniform_range(int n, double low, double high):
    """
    批量获取 [low, high) 范围内的均匀分布随机数

    Args:
        n: 需要的随机数数量
        low: 下界（包含）
        high: 上界（不包含）

    Returns:
        np.ndarray: 包含 n 个均匀分布随机数的数组
    """
    cdef np.ndarray result = _global_cache.get_batch_uniform_array(n)
    return low + (high - low) * result


# ============================================================================
# 工厂函数（用于创建独立的缓存实例）
# ============================================================================

def create_random_cache(int cache_size=DEFAULT_CACHE_SIZE) -> RandomCache:
    """
    创建一个新的 RandomCache 实例

    当需要线程独立的随机数缓存时使用此函数。

    Args:
        cache_size: 缓存大小，默认 100000

    Returns:
        RandomCache: 新的随机数缓存实例

    Example:
        >>> cache = create_random_cache(10000)
        >>> value = cache.get_uniform()
    """
    return RandomCache(cache_size)

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
高性能 Innovation 批量查询

使用 Cython 优化 get_innovation_numbers_batch 的核心循环。

主要优化：
1. 使用 C API 直接操作字典
2. 使用 typed memoryview 存储结果
3. 减少 Python 对象操作开销
"""

from cpython.dict cimport PyDict_GetItem, PyDict_SetItem
from cpython.ref cimport PyObject

import numpy as np
cimport numpy as np

# NumPy 类型声明
DTYPE = np.int64
ctypedef np.int64_t DTYPE_t


cpdef tuple fast_get_innovations_batch(
    dict generation_innovations,
    list connection_pairs,
    str mutation_type,
    int start_counter
):
    """
    批量获取 innovation numbers 的 Cython 优化版本

    直接操作 generation_innovations 字典（原地修改），
    返回 innovation 列表和新的计数器值。

    Args:
        generation_innovations: 代际创新字典，会被原地修改
        connection_pairs: 连接对列表 [(input_node, output_node), ...]
        mutation_type: 变异类型
        start_counter: 起始计数器值

    Returns:
        tuple: (innovations_list, new_counter_value)

    Example:
        >>> gen_innovations = {}
        >>> pairs = [(1, 2), (1, 3), (2, 3), (1, 2)]
        >>> innovations, counter = fast_get_innovations_batch(
        ...     gen_innovations, pairs, 'initial_connection', 0
        ... )
        >>> print(innovations)
        [1, 2, 3, 1]
        >>> print(counter)
        3
    """
    cdef:
        int counter = start_counter
        Py_ssize_t n = len(connection_pairs)
        Py_ssize_t i
        tuple pair
        tuple key
        PyObject* existing_ptr
        int innovation_number
        np.ndarray[DTYPE_t, ndim=1] result_array
        DTYPE_t[::1] result_view

    if n == 0:
        return ([], counter)

    # 使用 NumPy 数组存储结果
    result_array = np.empty(n, dtype=DTYPE)
    result_view = result_array

    for i in range(n):
        pair = <tuple>connection_pairs[i]
        # 创建 key tuple
        key = (pair[0], pair[1], mutation_type)

        # 使用 C API 直接查找字典，避免返回 None 的开销
        existing_ptr = PyDict_GetItem(generation_innovations, key)
        if existing_ptr != NULL:
            result_view[i] = <DTYPE_t><object>existing_ptr
        else:
            # 新 innovation - 递增计数器并记录
            counter += 1
            innovation_number = counter
            PyDict_SetItem(generation_innovations, key, innovation_number)
            result_view[i] = innovation_number

    return (result_array.tolist(), counter)


cpdef tuple fast_get_innovations_batch_array(
    dict generation_innovations,
    list connection_pairs,
    str mutation_type,
    int start_counter
):
    """
    批量获取 innovation numbers，返回 NumPy 数组版本

    与 fast_get_innovations_batch 相同，但返回 NumPy 数组而非列表。
    适用于后续需要 NumPy 数组操作的场景。

    Args:
        generation_innovations: 代际创新字典，会被原地修改
        connection_pairs: 连接对列表 [(input_node, output_node), ...]
        mutation_type: 变异类型
        start_counter: 起始计数器值

    Returns:
        tuple: (innovations_array, new_counter_value)
    """
    cdef:
        int counter = start_counter
        Py_ssize_t n = len(connection_pairs)
        Py_ssize_t i
        tuple pair
        tuple key
        PyObject* existing_ptr
        int innovation_number
        np.ndarray[DTYPE_t, ndim=1] result_array
        DTYPE_t[::1] result_view

    if n == 0:
        return (np.array([], dtype=DTYPE), counter)

    # 使用 NumPy 数组存储结果
    result_array = np.empty(n, dtype=DTYPE)
    result_view = result_array

    for i in range(n):
        pair = <tuple>connection_pairs[i]
        key = (pair[0], pair[1], mutation_type)

        existing_ptr = PyDict_GetItem(generation_innovations, key)
        if existing_ptr != NULL:
            result_view[i] = <DTYPE_t><object>existing_ptr
        else:
            counter += 1
            innovation_number = counter
            PyDict_SetItem(generation_innovations, key, innovation_number)
            result_view[i] = innovation_number

    return (result_array, counter)

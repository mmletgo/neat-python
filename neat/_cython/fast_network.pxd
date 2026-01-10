# cython: language_level=3
"""FastFeedForwardNetwork 类声明文件

此文件允许其他 Cython 模块 cimport FastFeedForwardNetwork 类，
并调用其 cdef 方法（如 activate_nogil）。
"""

cimport numpy as np
from numpy cimport float64_t as DTYPE_t

cdef class FastFeedForwardNetwork:
    # 输入输出节点信息
    cdef public int num_inputs
    cdef public int num_outputs
    cdef public np.ndarray input_keys
    cdef public np.ndarray output_keys

    # 节点计算信息
    cdef public int num_nodes
    cdef public np.ndarray node_ids
    cdef public np.ndarray biases
    cdef public np.ndarray responses
    cdef public np.ndarray act_types

    # 连接信息（CSR 格式）
    cdef public np.ndarray conn_indptr
    cdef public np.ndarray conn_sources
    cdef public np.ndarray conn_weights

    # 节点 ID 到索引的映射
    cdef dict id_to_idx

    # 值数组
    cdef public np.ndarray values

    # 预计算的输出索引
    cdef public np.ndarray output_indices

    # 方法声明
    cpdef np.ndarray activate(self, inputs)
    cdef void _forward_pass(self) noexcept
    cdef void activate_nogil(self, double[:] inputs, double[:] outputs) noexcept nogil
    cdef void _forward_pass_with_io(self, double[:] inputs, double[:] outputs) noexcept


# 批量并行处理函数声明（def 函数不需要在 .pxd 中声明，但这里文档化）
# def batch_activate_parallel(
#     list networks,
#     np.ndarray[DTYPE_t, ndim=2] inputs,
#     np.ndarray[DTYPE_t, ndim=2] outputs,
#     int num_threads=0,
# )

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
快速基因工厂函数

提供批量创建基因对象的优化实现，绕过 __init__ 断言检查。
使用 object.__new__(cls) 直接创建对象并设置 __slots__ 属性。
"""

import numpy as np
cimport numpy as np

# NumPy 类型声明
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def create_connection_genes_batch(
    conn_cls,
    list keys,
    list innovations,
    np.ndarray[DTYPE_t, ndim=1] weights,
    np.ndarray[np.uint8_t, ndim=1] enabled
) -> dict:
    """
    批量创建连接基因对象

    绕过 DefaultConnectionGene.__init__ 的断言检查，直接创建对象并设置属性。
    直接返回字典，避免调用方再次遍历添加。

    Args:
        conn_cls: DefaultConnectionGene 类
        keys: 连接键列表 [(input, output), ...]
        innovations: innovation number 列表
        weights: 权重数组 (numpy float64)
        enabled: 启用状态数组 (numpy uint8)

    Returns:
        连接基因字典 {key: gene, ...}
    """
    cdef int n = len(keys)
    cdef int i
    cdef dict result = {}
    cdef object gene
    cdef tuple key

    for i in range(n):
        key = keys[i]
        # 使用 object.__new__ 绕过 __init__ 断言
        gene = object.__new__(conn_cls)
        # 直接设置 __slots__ 属性
        gene.key = key
        gene.innovation = innovations[i]
        gene.weight = weights[i]
        gene.enabled = bool(enabled[i])
        result[key] = gene

    return result


def create_node_genes_batch(
    node_cls,
    list keys,
    np.ndarray[DTYPE_t, ndim=1] biases,
    np.ndarray[DTYPE_t, ndim=1] responses,
    list activations,
    list aggregations
) -> list:
    """
    批量创建节点基因对象

    绕过 DefaultNodeGene.__init__ 的断言检查，直接创建对象并设置属性。

    Args:
        node_cls: DefaultNodeGene 类
        keys: 节点键列表 [int, ...]
        biases: 偏置数组 (numpy float64)
        responses: 响应系数数组 (numpy float64)
        activations: 激活函数名称列表 [str, ...]
        aggregations: 聚合函数名称列表 [str, ...]

    Returns:
        节点基因对象列表
    """
    cdef int n = len(keys)
    cdef int i
    cdef list result = []
    cdef object gene

    for i in range(n):
        # 使用 object.__new__ 绕过 __init__ 断言
        gene = object.__new__(node_cls)
        # 直接设置 __slots__ 属性
        gene.key = keys[i]
        gene.bias = biases[i]
        gene.response = responses[i]
        gene.activation = activations[i]
        gene.aggregation = aggregations[i]
        result.append(gene)

    return result


def create_single_connection_gene(
    conn_cls,
    tuple key,
    int innovation,
    double weight,
    bint enabled
) -> object:
    """
    快速创建单个连接基因对象

    绕过 DefaultConnectionGene.__init__ 的断言检查。

    Args:
        conn_cls: DefaultConnectionGene 类
        key: 连接键 (input, output)
        innovation: innovation number
        weight: 权重
        enabled: 启用状态

    Returns:
        连接基因对象
    """
    cdef object gene = object.__new__(conn_cls)
    gene.key = key
    gene.innovation = innovation
    gene.weight = weight
    gene.enabled = enabled
    return gene


def create_single_node_gene(
    node_cls,
    int key,
    double bias,
    double response,
    str activation,
    str aggregation
) -> object:
    """
    快速创建单个节点基因对象

    绕过 DefaultNodeGene.__init__ 的断言检查。

    Args:
        node_cls: DefaultNodeGene 类
        key: 节点键
        bias: 偏置
        response: 响应系数
        activation: 激活函数名称
        aggregation: 聚合函数名称

    Returns:
        节点基因对象
    """
    cdef object gene = object.__new__(node_cls)
    gene.key = key
    gene.bias = bias
    gene.response = response
    gene.activation = activation
    gene.aggregation = aggregation
    return gene

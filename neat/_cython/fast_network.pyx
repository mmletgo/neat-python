# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
高性能前向传播神经网络 (Cython 优化版本)

使用 Cython 优化的 NEAT 神经网络前向传播实现。
主要优化：
1. 预构建 NumPy 数组替代 Python dict 和 list
2. 使用 Cython 类型声明加速循环
3. 避免运行时的 list.append 操作
4. 使用 CSR 格式存储稀疏连接矩阵
5. 支持 nogil 多线程并行
"""

import numpy as np
cimport numpy as np
from libc.math cimport tanh, exp, sin, cos, fabs
from cython.parallel cimport prange, parallel

# 【关键修复】将导入移到模块级别，避免多线程并发导入导致死锁
# Python 的导入机制有全局锁，多线程并发导入可能导致死锁
try:
    from neat._cython.fast_graphs import fast_feed_forward_layers as _feed_forward_layers
except ImportError:
    from neat.graphs import feed_forward_layers as _feed_forward_layers

# NumPy 类型声明
DTYPE = np.float64
# DTYPE_t 已在 numpy.pxd 中定义为 float64_t
from numpy cimport float64_t as DTYPE_t

# 激活函数类型枚举
DEF ACT_TANH = 0
DEF ACT_SIGMOID = 1
DEF ACT_RELU = 2
DEF ACT_IDENTITY = 3
DEF ACT_SIN = 4
DEF ACT_GAUSS = 5


cdef inline double activate(double x, int act_type) noexcept nogil:
    """内联激活函数"""
    if act_type == ACT_TANH:
        return tanh(x)
    elif act_type == ACT_SIGMOID:
        return 1.0 / (1.0 + exp(-x))
    elif act_type == ACT_RELU:
        return x if x > 0 else 0.0
    elif act_type == ACT_IDENTITY:
        return x
    elif act_type == ACT_SIN:
        return sin(x)
    elif act_type == ACT_GAUSS:
        return exp(-x * x)
    else:
        return tanh(x)  # 默认 tanh


cdef inline double aggregate_sum(double* arr, int n) noexcept nogil:
    """求和聚合"""
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += arr[i]
    return total


cdef class FastFeedForwardNetwork:
    """
    高性能前向传播网络

    在创建时将 NEAT 网络结构转换为优化的数组表示，
    前向传播时使用 Cython 循环加速计算。

    属性（在 fast_network.pxd 中声明）:
    - num_inputs, num_outputs: 输入输出节点数
    - input_keys, output_keys: 输入输出节点 ID 数组
    - num_nodes: 总节点数（不含输入）
    - node_ids, biases, responses, act_types: 节点信息数组
    - conn_indptr, conn_sources, conn_weights: CSR 格式连接信息
    - id_to_idx: 节点 ID 到索引的映射
    - values: 前向传播值数组
    - output_indices: 预计算的输出索引
    """

    def __init__(self):
        """初始化空网络，由 create 方法填充"""
        pass

    @staticmethod
    def create(genome, config):
        """
        从 NEAT 基因组创建快速网络

        Args:
            genome: NEAT 基因组
            config: NEAT 配置

        Returns:
            FastFeedForwardNetwork 实例
        """
        # 使用模块级别导入的 feed_forward_layers（避免多线程并发导入死锁）
        feed_forward_layers = _feed_forward_layers

        cdef FastFeedForwardNetwork network = FastFeedForwardNetwork()

        # 获取输入输出键
        input_keys = config.genome_config.input_keys
        output_keys = config.genome_config.output_keys

        cdef int num_inputs = len(input_keys)
        cdef int num_outputs = len(output_keys)
        network.num_inputs = num_inputs
        network.num_outputs = num_outputs
        network.input_keys = np.array(input_keys, dtype=np.int32)
        network.output_keys = np.array(output_keys, dtype=np.int32)

        # 单次遍历：获取启用连接并同时构建 conn_map
        # conn_map: 目标节点 -> [(源节点, 权重), ...]
        cdef dict conn_map = {}
        cdef list connections = []
        cdef tuple key
        cdef int inode, onode

        for cg in genome.connections.values():
            if cg.enabled:
                key = cg.key
                connections.append(key)
                inode, onode = key
                if onode not in conn_map:
                    conn_map[onode] = [(inode, cg.weight)]
                else:
                    conn_map[onode].append((inode, cg.weight))

        # 获取拓扑层
        layers, required = feed_forward_layers(input_keys, output_keys, connections)

        # 构建节点 ID 到索引的映射
        # 输入节点: 0 ~ num_inputs-1
        # 其他节点: num_inputs ~ ...
        cdef dict id_to_idx = {}
        cdef int idx = 0
        for key_int in input_keys:
            id_to_idx[key_int] = idx
            idx += 1

        # 计算非输入节点数量并预分配数组
        cdef int num_nodes = sum(len(layer) for layer in layers)

        if num_nodes == 0:
            # 无隐藏/输出节点（退化情况）
            network.id_to_idx = id_to_idx
            network.num_nodes = 0
            network.values = np.zeros(num_inputs, dtype=DTYPE)
            network.node_ids = np.array([], dtype=np.int32)
            network.biases = np.array([], dtype=DTYPE)
            network.responses = np.array([], dtype=DTYPE)
            network.act_types = np.array([], dtype=np.int32)
            network.conn_indptr = np.zeros(1, dtype=np.int32)
            network.conn_sources = np.array([], dtype=np.int32)
            network.conn_weights = np.array([], dtype=DTYPE)
            network.output_indices = np.array([], dtype=np.int32)
            return network

        # 预分配节点相关数组
        cdef np.ndarray[np.int32_t, ndim=1] node_ids = np.empty(num_nodes, dtype=np.int32)
        cdef np.ndarray[DTYPE_t, ndim=1] biases = np.empty(num_nodes, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] responses = np.empty(num_nodes, dtype=DTYPE)
        cdef np.ndarray[np.int32_t, ndim=1] act_types = np.empty(num_nodes, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] conn_indptr = np.empty(num_nodes + 1, dtype=np.int32)

        # 激活函数名到类型的映射
        cdef dict act_map = {
            'tanh': ACT_TANH,
            'sigmoid': ACT_SIGMOID,
            'relu': ACT_RELU,
            'identity': ACT_IDENTITY,
            'sin': ACT_SIN,
            'gauss': ACT_GAUSS,
        }

        # 只保留 required 节点的连接（过滤 conn_map）
        cdef set required_with_inputs = required.union(set(input_keys))

        # 缓存 genome.nodes 避免重复属性访问
        cdef dict genome_nodes = genome.nodes

        # 第一遍：计算总连接数并填充节点信息
        # 同时缓存每个节点的连接列表，避免第二遍重复查找
        cdef int i = 0
        cdef int total_conns = 0
        cdef list node_conns
        cdef list all_node_conns = []  # 缓存每个节点的连接列表

        conn_indptr[0] = 0
        for layer in layers:
            for node in layer:
                id_to_idx[node] = idx
                node_ids[i] = node

                # 获取节点基因
                ng = genome_nodes[node]
                biases[i] = ng.bias
                responses[i] = ng.response
                act_types[i] = act_map.get(ng.activation, ACT_TANH)

                # 获取并缓存该节点的连接（使用 get 避免两次查找）
                node_conns = conn_map.get(node)
                all_node_conns.append(node_conns)

                # 计算该节点的有效连接数
                if node_conns is not None:
                    for src, _ in node_conns:
                        if src in required_with_inputs:
                            total_conns += 1

                conn_indptr[i + 1] = total_conns
                idx += 1
                i += 1

        # 预分配连接数组
        cdef np.ndarray[np.int32_t, ndim=1] conn_sources = np.empty(total_conns, dtype=np.int32)
        cdef np.ndarray[DTYPE_t, ndim=1] conn_weights = np.empty(total_conns, dtype=DTYPE)

        # 第二遍：填充连接数组（使用缓存的连接列表）
        cdef int conn_idx = 0
        cdef double weight

        for i in range(num_nodes):
            node_conns = all_node_conns[i]
            if node_conns is not None:
                for src, weight in node_conns:
                    if src in required_with_inputs:
                        conn_sources[conn_idx] = id_to_idx[src]
                        conn_weights[conn_idx] = weight
                        conn_idx += 1

        # 设置网络属性
        network.id_to_idx = id_to_idx
        network.num_nodes = num_nodes
        network.values = np.zeros(num_inputs + num_nodes, dtype=DTYPE)
        network.node_ids = node_ids
        network.biases = biases
        network.responses = responses
        network.act_types = act_types
        network.conn_indptr = conn_indptr
        network.conn_sources = conn_sources
        network.conn_weights = conn_weights

        # 预计算输出索引（避免每次 activate 时访问 dict）
        cdef np.ndarray[np.int32_t, ndim=1] output_idx_arr = np.empty(num_outputs, dtype=np.int32)
        for i in range(num_outputs):
            output_idx_arr[i] = id_to_idx[output_keys[i]]
        network.output_indices = output_idx_arr

        return network

    cpdef np.ndarray activate(self, inputs):
        """
        前向传播（Python/Cython 混合接口）

        优化：
        - 使用 cpdef 减少 Python 调用开销
        - 输入复制和计算都尽量释放 GIL
        - 使用预计算的输出索引，避免每次访问 dict

        Args:
            inputs: 输入值列表或 ndarray

        Returns:
            输出值 numpy 数组
        """
        cdef int i, num_inputs, num_outputs
        cdef double[:] values_view = self.values
        cdef np.ndarray[DTYPE_t, ndim=1] input_arr
        cdef double[:] input_view
        cdef np.ndarray[DTYPE_t, ndim=1] result
        cdef int[:] output_indices_view = self.output_indices
        cdef int out_idx

        num_inputs = self.num_inputs
        num_outputs = self.num_outputs

        # 处理输入（需要 GIL 进行类型检查）
        if isinstance(inputs, np.ndarray):
            if len(inputs) != num_inputs:
                raise RuntimeError(f"Expected {num_inputs} inputs, got {len(inputs)}")
            input_arr = inputs.astype(DTYPE, copy=False)
            input_view = input_arr
        else:
            if len(inputs) != num_inputs:
                raise RuntimeError(f"Expected {num_inputs} inputs, got {len(inputs)}")
            input_arr = np.array(inputs, dtype=DTYPE)
            input_view = input_arr

        # 预分配输出数组（需要 GIL）
        result = np.empty(num_outputs, dtype=DTYPE)
        cdef double[:] result_view = result

        # 释放 GIL 执行核心计算
        with nogil:
            # 复制输入数据
            for i in range(num_inputs):
                values_view[i] = input_view[i]

        # 调用 Cython 优化的前向传播（内部也释放 GIL）
        self._forward_pass()

        # 释放 GIL 提取输出（使用预计算的索引）
        with nogil:
            for i in range(num_outputs):
                out_idx = output_indices_view[i]
                result_view[i] = values_view[out_idx]

        return result

    cdef void _forward_pass(self) noexcept:
        """
        Cython 优化的前向传播核心（释放 GIL 支持多线程并行）
        """
        # 使用 typed memoryview 替代 ndarray，支持 nogil
        cdef double[:] values = self.values
        cdef double[:] biases = self.biases
        cdef double[:] responses = self.responses
        cdef int[:] act_types = self.act_types
        cdef int[:] conn_indptr = self.conn_indptr
        cdef int[:] conn_sources = self.conn_sources
        cdef double[:] conn_weights = self.conn_weights

        cdef int num_nodes = self.num_nodes
        cdef int num_inputs = self.num_inputs
        cdef int i, j, start, end, src_idx
        cdef double node_sum, weighted_input

        # 释放 GIL 执行计算，允许多线程真正并行
        with nogil:
            # 按拓扑顺序计算每个节点
            for i in range(num_nodes):
                # 计算加权输入和
                start = conn_indptr[i]
                end = conn_indptr[i + 1]
                node_sum = 0.0

                for j in range(start, end):
                    src_idx = conn_sources[j]
                    node_sum += values[src_idx] * conn_weights[j]

                # 应用偏置、响应系数和激活函数
                weighted_input = biases[i] + responses[i] * node_sum
                values[num_inputs + i] = activate(weighted_input, act_types[i])

    cdef void activate_nogil(
        self,
        double[:] inputs,
        double[:] outputs,
    ) noexcept nogil:
        """完全 nogil 的前向传播，可从外部 nogil 块调用

        此方法允许在多线程环境中真正并行执行神经网络计算。
        调用者需要确保 inputs 和 outputs 的大小正确。

        Args:
            inputs: 输入数组 (memoryview, 大小 = num_inputs)
            outputs: 输出数组 (memoryview, 大小 = num_outputs)
        """
        cdef double[:] values = self.values
        cdef double[:] biases = self.biases
        cdef double[:] responses = self.responses
        cdef int[:] act_types = self.act_types
        cdef int[:] conn_indptr = self.conn_indptr
        cdef int[:] conn_sources = self.conn_sources
        cdef double[:] conn_weights = self.conn_weights
        cdef int[:] output_indices = self.output_indices

        cdef int num_nodes = self.num_nodes
        cdef int num_inputs = self.num_inputs
        cdef int num_outputs = self.num_outputs
        cdef int i, j, start, end, src_idx, out_idx
        cdef double node_sum, weighted_input

        # 1. 复制输入到 values 数组
        for i in range(num_inputs):
            values[i] = inputs[i]

        # 2. 前向传播
        for i in range(num_nodes):
            start = conn_indptr[i]
            end = conn_indptr[i + 1]
            node_sum = 0.0

            for j in range(start, end):
                src_idx = conn_sources[j]
                node_sum += values[src_idx] * conn_weights[j]

            weighted_input = biases[i] + responses[i] * node_sum
            values[num_inputs + i] = activate(weighted_input, act_types[i])

        # 3. 提取输出
        for i in range(num_outputs):
            out_idx = output_indices[i]
            outputs[i] = values[out_idx]

    cdef void _forward_pass_with_io(
        self,
        double[:] inputs,
        double[:] outputs,
    ) noexcept:
        """带输入输出的前向传播（Cython 优化，需要 GIL）

        与 activate_nogil 类似，但不使用 nogil，因为需要访问 Python 对象属性。
        主要用于批量处理场景。

        Args:
            inputs: 输入数组 (memoryview, 大小 = num_inputs)
            outputs: 输出数组 (memoryview, 大小 = num_outputs)
        """
        cdef double[:] values = self.values
        cdef double[:] biases = self.biases
        cdef double[:] responses = self.responses
        cdef int[:] act_types = self.act_types
        cdef int[:] conn_indptr = self.conn_indptr
        cdef int[:] conn_sources = self.conn_sources
        cdef double[:] conn_weights = self.conn_weights
        cdef int[:] output_indices = self.output_indices

        cdef int num_nodes = self.num_nodes
        cdef int num_inputs = self.num_inputs
        cdef int num_outputs = self.num_outputs
        cdef int i, j, start, end, src_idx, out_idx
        cdef double node_sum, weighted_input

        # 1. 复制输入到 values 数组
        for i in range(num_inputs):
            values[i] = inputs[i]

        # 2. 前向传播
        for i in range(num_nodes):
            start = conn_indptr[i]
            end = conn_indptr[i + 1]
            node_sum = 0.0

            for j in range(start, end):
                src_idx = conn_sources[j]
                node_sum += values[src_idx] * conn_weights[j]

            weighted_input = biases[i] + responses[i] * node_sum
            values[num_inputs + i] = activate(weighted_input, act_types[i])

        # 3. 提取输出
        for i in range(num_outputs):
            out_idx = output_indices[i]
            outputs[i] = values[out_idx]


def batch_activate_parallel(
    list networks,
    np.ndarray[DTYPE_t, ndim=2] inputs,
    np.ndarray[DTYPE_t, ndim=2] outputs,
    int num_threads=0,
):
    """批量激活多个网络

    虽然名为 parallel，但由于 Python GIL 和对象访问限制，
    目前实现为串行执行。比纯 Python 循环更快，因为使用了
    Cython 类型化和 nogil 块。

    Args:
        networks: FastFeedForwardNetwork 实例列表
        inputs: 2D 输入数组 [num_networks, num_inputs]
        outputs: 2D 输出数组 [num_networks, num_outputs]（会被填充）
        num_threads: 线程数（未使用）

    Returns:
        None (结果写入 outputs 数组)
    """
    cdef int n = len(networks)
    cdef int i
    cdef FastFeedForwardNetwork net
    cdef double[:, :] inputs_view = inputs
    cdef double[:, :] outputs_view = outputs

    # 串行执行，但使用 Cython 优化
    for i in range(n):
        net = <FastFeedForwardNetwork>networks[i]
        # 调用优化的 activate 方法（不使用 nogil 版本，因为需要访问 Python 对象）
        net._forward_pass_with_io(inputs_view[i], outputs_view[i])

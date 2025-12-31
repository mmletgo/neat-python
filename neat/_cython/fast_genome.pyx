# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
高性能基因组变异模块

使用 Cython 向量化实现 NEAT 基因组的变异操作。
主要优化：
1. MutationConfig 缓存所有变异配置参数，避免重复 getattr 调用
2. 批量提取基因属性到 NumPy 数组
3. 向量化变异操作
4. 批量写回结果

原始问题：
- 逐基因调用 mutate()，大量 getattr 调用
- 没有向量化，无法利用 SIMD

优化策略：
- 提取数组 -> 向量化操作 -> 写回的模式
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt, log, cos, sin

# NumPy 类型声明
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t INT32_t

# 常量
cdef double TWO_PI = 6.283185307179586
cdef double RAND_MAX_INV = 1.0 / <double>RAND_MAX


# ============================================================================
# 随机数生成辅助函数
# ============================================================================

cdef inline double fast_random() noexcept nogil:
    """快速生成 [0, 1) 均匀分布随机数"""
    return <double>rand() * RAND_MAX_INV


cdef inline double fast_gauss(double mean, double stdev) noexcept nogil:
    """
    快速生成高斯分布随机数（Box-Muller 变换）
    """
    cdef double u1 = fast_random()
    cdef double u2 = fast_random()
    # 避免 log(0)
    if u1 < 1e-10:
        u1 = 1e-10
    cdef double z = sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2)
    return mean + stdev * z


cdef inline double clamp(double value, double min_val, double max_val) noexcept nogil:
    """限制值在范围内"""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


# ============================================================================
# MutationConfig - 缓存变异配置参数
# ============================================================================

cdef class MutationConfig:
    """
    缓存所有变异配置参数的类

    避免每次变异时通过 getattr 动态获取配置，
    在创建时一次性从 NEAT 配置对象提取所有参数。

    应该被缓存复用，避免每次变异都创建新实例。
    """

    # ========== 节点基因参数 ==========
    # bias 属性
    cdef public double bias_mutate_rate
    cdef public double bias_replace_rate
    cdef public double bias_mutate_power
    cdef public double bias_min_value
    cdef public double bias_max_value
    cdef public double bias_init_mean
    cdef public double bias_init_stdev

    # response 属性
    cdef public double response_mutate_rate
    cdef public double response_replace_rate
    cdef public double response_mutate_power
    cdef public double response_min_value
    cdef public double response_max_value
    cdef public double response_init_mean
    cdef public double response_init_stdev

    # activation 和 aggregation（字符串属性，变异率）
    cdef public double activation_mutate_rate
    cdef public double aggregation_mutate_rate

    # ========== 连接基因参数 ==========
    # weight 属性
    cdef public double weight_mutate_rate
    cdef public double weight_replace_rate
    cdef public double weight_mutate_power
    cdef public double weight_min_value
    cdef public double weight_max_value
    cdef public double weight_init_mean
    cdef public double weight_init_stdev

    # enabled 属性（布尔属性）
    cdef public double enabled_mutate_rate
    cdef public double enabled_rate_to_true_add
    cdef public double enabled_rate_to_false_add

    # activation 和 aggregation 选项列表（Python 对象）
    cdef public list activation_options
    cdef public list aggregation_options

    def __init__(self):
        """初始化默认值为 0"""
        # bias
        self.bias_mutate_rate = 0.0
        self.bias_replace_rate = 0.0
        self.bias_mutate_power = 0.0
        self.bias_min_value = -30.0
        self.bias_max_value = 30.0
        self.bias_init_mean = 0.0
        self.bias_init_stdev = 1.0

        # response
        self.response_mutate_rate = 0.0
        self.response_replace_rate = 0.0
        self.response_mutate_power = 0.0
        self.response_min_value = -30.0
        self.response_max_value = 30.0
        self.response_init_mean = 1.0
        self.response_init_stdev = 0.0

        # activation/aggregation
        self.activation_mutate_rate = 0.0
        self.aggregation_mutate_rate = 0.0

        # weight
        self.weight_mutate_rate = 0.0
        self.weight_replace_rate = 0.0
        self.weight_mutate_power = 0.0
        self.weight_min_value = -30.0
        self.weight_max_value = 30.0
        self.weight_init_mean = 0.0
        self.weight_init_stdev = 1.0

        # enabled
        self.enabled_mutate_rate = 0.0
        self.enabled_rate_to_true_add = 0.0
        self.enabled_rate_to_false_add = 0.0

        # 选项列表
        self.activation_options = []
        self.aggregation_options = []

    @staticmethod
    def from_neat_config(genome_config):
        """
        从 NEAT 配置对象提取所有变异参数

        Args:
            genome_config: DefaultGenomeConfig 实例

        Returns:
            MutationConfig 实例
        """
        cdef MutationConfig mc = MutationConfig()

        # ========== 提取 bias 参数 ==========
        mc.bias_mutate_rate = getattr(genome_config, 'bias_mutate_rate', 0.0)
        mc.bias_replace_rate = getattr(genome_config, 'bias_replace_rate', 0.0)
        mc.bias_mutate_power = getattr(genome_config, 'bias_mutate_power', 0.5)
        mc.bias_min_value = getattr(genome_config, 'bias_min_value', -30.0)
        mc.bias_max_value = getattr(genome_config, 'bias_max_value', 30.0)
        mc.bias_init_mean = getattr(genome_config, 'bias_init_mean', 0.0)
        mc.bias_init_stdev = getattr(genome_config, 'bias_init_stdev', 1.0)

        # ========== 提取 response 参数 ==========
        mc.response_mutate_rate = getattr(genome_config, 'response_mutate_rate', 0.0)
        mc.response_replace_rate = getattr(genome_config, 'response_replace_rate', 0.0)
        mc.response_mutate_power = getattr(genome_config, 'response_mutate_power', 0.5)
        mc.response_min_value = getattr(genome_config, 'response_min_value', -30.0)
        mc.response_max_value = getattr(genome_config, 'response_max_value', 30.0)
        mc.response_init_mean = getattr(genome_config, 'response_init_mean', 1.0)
        mc.response_init_stdev = getattr(genome_config, 'response_init_stdev', 0.0)

        # ========== 提取 activation/aggregation 参数 ==========
        mc.activation_mutate_rate = getattr(genome_config, 'activation_mutate_rate', 0.0)
        mc.aggregation_mutate_rate = getattr(genome_config, 'aggregation_mutate_rate', 0.0)

        # 获取选项列表
        mc.activation_options = list(getattr(genome_config, 'activation_options', ['tanh']))
        mc.aggregation_options = list(getattr(genome_config, 'aggregation_options', ['sum']))

        # ========== 提取 weight 参数 ==========
        mc.weight_mutate_rate = getattr(genome_config, 'weight_mutate_rate', 0.0)
        mc.weight_replace_rate = getattr(genome_config, 'weight_replace_rate', 0.0)
        mc.weight_mutate_power = getattr(genome_config, 'weight_mutate_power', 0.5)
        mc.weight_min_value = getattr(genome_config, 'weight_min_value', -30.0)
        mc.weight_max_value = getattr(genome_config, 'weight_max_value', 30.0)
        mc.weight_init_mean = getattr(genome_config, 'weight_init_mean', 0.0)
        mc.weight_init_stdev = getattr(genome_config, 'weight_init_stdev', 1.0)

        # ========== 提取 enabled 参数 ==========
        mc.enabled_mutate_rate = getattr(genome_config, 'enabled_mutate_rate', 0.0)
        mc.enabled_rate_to_true_add = getattr(genome_config, 'enabled_rate_to_true_add', 0.0)
        mc.enabled_rate_to_false_add = getattr(genome_config, 'enabled_rate_to_false_add', 0.0)

        return mc


# ============================================================================
# 向量化变异核心函数
# ============================================================================

cdef void _mutate_float_array(
    double[:] values,
    double mutate_rate,
    double replace_rate,
    double mutate_power,
    double min_value,
    double max_value,
    double init_mean,
    double init_stdev,
    int n
) noexcept nogil:
    """
    向量化浮点属性变异（nogil 版本）

    对数组中的每个值进行变异：
    1. 以 mutate_rate 概率进行小幅变异（加高斯噪声）
    2. 以 replace_rate 概率完全替换为新初始值
    3. 否则保持不变

    Args:
        values: 待变异的值数组（原地修改）
        mutate_rate: 变异概率
        replace_rate: 替换概率
        mutate_power: 变异强度（高斯标准差）
        min_value: 最小值
        max_value: 最大值
        init_mean: 初始化均值
        init_stdev: 初始化标准差
        n: 数组长度
    """
    cdef int i
    cdef double r, new_val
    cdef double combined_rate = mutate_rate + replace_rate

    for i in range(n):
        r = fast_random()

        if r < mutate_rate:
            # 小幅变异：加高斯噪声
            new_val = values[i] + fast_gauss(0.0, mutate_power)
            values[i] = clamp(new_val, min_value, max_value)
        elif r < combined_rate:
            # 完全替换：用新初始值
            new_val = fast_gauss(init_mean, init_stdev)
            values[i] = clamp(new_val, min_value, max_value)
        # else: 保持不变


cdef void _mutate_bool_array(
    np.int8_t[:] values,
    double mutate_rate,
    double rate_to_true_add,
    double rate_to_false_add,
    int n
) noexcept nogil:
    """
    向量化布尔属性变异（nogil 版本）

    Args:
        values: 布尔值数组（以 int8 存储，0 或 1）
        mutate_rate: 基础变异概率
        rate_to_true_add: 当前为 False 时的额外变异概率
        rate_to_false_add: 当前为 True 时的额外变异概率
        n: 数组长度
    """
    cdef int i
    cdef double r, actual_rate

    for i in range(n):
        if values[i]:
            # 当前为 True，可能变为 False
            actual_rate = mutate_rate + rate_to_false_add
        else:
            # 当前为 False，可能变为 True
            actual_rate = mutate_rate + rate_to_true_add

        if actual_rate > 0:
            r = fast_random()
            if r < actual_rate:
                # 随机选择新值（与原实现一致）
                values[i] = 1 if fast_random() < 0.5 else 0


# ============================================================================
# 节点基因变异
# ============================================================================

cpdef void fast_mutate_node_genes(
    np.ndarray[DTYPE_t, ndim=1] biases,
    np.ndarray[DTYPE_t, ndim=1] responses,
    MutationConfig config
):
    """
    向量化变异节点基因的数值属性（bias, response）

    Args:
        biases: 偏置数组（原地修改）
        responses: 响应系数数组（原地修改）
        config: 变异配置
    """
    cdef int n = biases.shape[0]
    if n == 0:
        return

    # 变异 bias
    _mutate_float_array(
        biases,
        config.bias_mutate_rate,
        config.bias_replace_rate,
        config.bias_mutate_power,
        config.bias_min_value,
        config.bias_max_value,
        config.bias_init_mean,
        config.bias_init_stdev,
        n
    )

    # 变异 response
    _mutate_float_array(
        responses,
        config.response_mutate_rate,
        config.response_replace_rate,
        config.response_mutate_power,
        config.response_min_value,
        config.response_max_value,
        config.response_init_mean,
        config.response_init_stdev,
        n
    )


def mutate_string_attributes(
    list items,
    str attr_name,
    double mutate_rate,
    list options
):
    """
    变异字符串属性（activation, aggregation）

    由于字符串操作无法在 nogil 中进行，这个函数使用 Python 实现。

    Args:
        items: 基因对象列表
        attr_name: 属性名称
        mutate_rate: 变异概率
        options: 可选值列表
    """
    import random

    if not options or mutate_rate <= 0:
        return

    for item in items:
        if random.random() < mutate_rate:
            setattr(item, attr_name, random.choice(options))


# ============================================================================
# 连接基因变异
# ============================================================================

cpdef void fast_mutate_connection_genes(
    np.ndarray[DTYPE_t, ndim=1] weights,
    np.ndarray[np.int8_t, ndim=1] enabled,
    MutationConfig config
):
    """
    向量化变异连接基因属性（weight, enabled）

    Args:
        weights: 权重数组（原地修改）
        enabled: 启用状态数组（原地修改，int8 表示 bool）
        config: 变异配置
    """
    cdef int n = weights.shape[0]
    if n == 0:
        return

    # 变异 weight
    _mutate_float_array(
        weights,
        config.weight_mutate_rate,
        config.weight_replace_rate,
        config.weight_mutate_power,
        config.weight_min_value,
        config.weight_max_value,
        config.weight_init_mean,
        config.weight_init_stdev,
        n
    )

    # 变异 enabled
    _mutate_bool_array(
        enabled,
        config.enabled_mutate_rate,
        config.enabled_rate_to_true_add,
        config.enabled_rate_to_false_add,
        n
    )


# ============================================================================
# 主入口：基因组变异
# ============================================================================

cpdef void fast_mutate_genome(
    dict nodes,
    dict connections,
    MutationConfig config
):
    """
    向量化变异整个基因组

    实现流程：
    1. 提取所有节点属性到数组
    2. 向量化变异节点属性
    3. 写回节点属性
    4. 提取所有连接属性到数组
    5. 向量化变异连接属性
    6. 写回连接属性

    注意：这个函数只处理数值属性的变异，
    字符串属性（activation, aggregation）需要单独处理。

    Args:
        nodes: 节点基因字典 {node_id: DefaultNodeGene}
        connections: 连接基因字典 {conn_key: DefaultConnectionGene}
        config: 变异配置
    """
    cdef int n_nodes, n_conns
    cdef int i
    cdef list node_ids, conn_keys, node_list
    cdef np.ndarray[DTYPE_t, ndim=1] biases
    cdef np.ndarray[DTYPE_t, ndim=1] responses
    cdef np.ndarray[DTYPE_t, ndim=1] weights
    cdef np.ndarray[np.int8_t, ndim=1] enabled

    # ========== 处理节点基因 ==========
    n_nodes = len(nodes)
    if n_nodes > 0:
        node_ids = list(nodes.keys())

        # 提取属性到数组
        biases = np.empty(n_nodes, dtype=DTYPE)
        responses = np.empty(n_nodes, dtype=DTYPE)

        for i in range(n_nodes):
            biases[i] = nodes[node_ids[i]].bias
            responses[i] = nodes[node_ids[i]].response

        # 向量化变异
        fast_mutate_node_genes(biases, responses, config)

        # 写回
        for i in range(n_nodes):
            nodes[node_ids[i]].bias = biases[i]
            nodes[node_ids[i]].response = responses[i]

        # 变异字符串属性
        node_list = list(nodes.values())
        mutate_string_attributes(
            node_list,
            'activation',
            config.activation_mutate_rate,
            config.activation_options
        )
        mutate_string_attributes(
            node_list,
            'aggregation',
            config.aggregation_mutate_rate,
            config.aggregation_options
        )

    # ========== 处理连接基因 ==========
    n_conns = len(connections)
    if n_conns > 0:
        conn_keys = list(connections.keys())

        # 提取属性到数组
        weights = np.empty(n_conns, dtype=DTYPE)
        enabled = np.empty(n_conns, dtype=np.int8)

        for i in range(n_conns):
            weights[i] = connections[conn_keys[i]].weight
            enabled[i] = 1 if connections[conn_keys[i]].enabled else 0

        # 向量化变异
        fast_mutate_connection_genes(weights, enabled, config)

        # 写回
        for i in range(n_conns):
            connections[conn_keys[i]].weight = weights[i]
            connections[conn_keys[i]].enabled = bool(enabled[i])


# ============================================================================
# 批量基因组距离计算（可选优化）
# ============================================================================

cpdef double fast_genome_distance(
    np.ndarray[DTYPE_t, ndim=1] weights1,
    np.ndarray[DTYPE_t, ndim=1] weights2,
    np.ndarray[np.int8_t, ndim=1] enabled1,
    np.ndarray[np.int8_t, ndim=1] enabled2,
    double weight_coeff
):
    """
    向量化计算两个基因组的连接基因距离

    用于计算同源连接基因的距离。假设输入数组已按相同顺序对齐
    （即 weights1[i] 和 weights2[i] 是同源基因的权重）。

    计算公式：
    distance = (权重差绝对值之和 + 启用状态差异数) * weight_coeff

    Args:
        weights1: 第一个基因组的权重数组
        weights2: 第二个基因组的权重数组
        enabled1: 第一个基因组的启用状态数组
        enabled2: 第二个基因组的启用状态数组
        weight_coeff: 权重系数

    Returns:
        距离值
    """
    cdef int n = weights1.shape[0]
    if n == 0:
        return 0.0

    cdef double weight_diff = 0.0
    cdef int enabled_diff = 0
    cdef int i
    cdef double diff

    for i in range(n):
        # 权重差绝对值
        diff = weights1[i] - weights2[i]
        if diff < 0:
            diff = -diff
        weight_diff += diff

        # 启用状态差异
        if enabled1[i] != enabled2[i]:
            enabled_diff += 1

    return (weight_diff + <double>enabled_diff) * weight_coeff


cpdef double fast_node_distance(
    np.ndarray[DTYPE_t, ndim=1] biases1,
    np.ndarray[DTYPE_t, ndim=1] biases2,
    np.ndarray[DTYPE_t, ndim=1] responses1,
    np.ndarray[DTYPE_t, ndim=1] responses2,
    double weight_coeff
):
    """
    向量化计算两个基因组的节点基因距离

    用于计算同源节点基因的距离。假设输入数组已按相同顺序对齐。

    注意：这个函数只计算数值属性（bias, response）的距离，
    字符串属性（activation, aggregation）的差异需要另外计算。

    Args:
        biases1: 第一个基因组的偏置数组
        biases2: 第二个基因组的偏置数组
        responses1: 第一个基因组的响应系数数组
        responses2: 第二个基因组的响应系数数组
        weight_coeff: 权重系数

    Returns:
        距离值（不含字符串属性差异）
    """
    cdef int n = biases1.shape[0]
    if n == 0:
        return 0.0

    cdef double total_diff = 0.0
    cdef int i
    cdef double bias_diff, response_diff

    for i in range(n):
        # 偏置差绝对值
        bias_diff = biases1[i] - biases2[i]
        if bias_diff < 0:
            bias_diff = -bias_diff

        # 响应系数差绝对值
        response_diff = responses1[i] - responses2[i]
        if response_diff < 0:
            response_diff = -response_diff

        total_diff += bias_diff + response_diff

    return total_diff * weight_coeff


# ============================================================================
# 批量基因组变异（用于整个种群）
# ============================================================================

def fast_mutate_population(
    list genomes,
    MutationConfig config
):
    """
    批量变异整个种群的基因组

    对种群中的每个基因组应用 fast_mutate_genome。
    这个函数主要是为了方便调用，实际的向量化优化
    在 fast_mutate_genome 中完成。

    Args:
        genomes: 基因组列表（DefaultGenome 实例）
        config: 变异配置
    """
    for genome in genomes:
        fast_mutate_genome(genome.nodes, genome.connections, config)


# ============================================================================
# 完整的基因组距离计算（一次性完成，避免临时数组）
# ============================================================================

cpdef double fast_genome_distance_full(
    dict self_nodes,
    dict self_connections,
    dict other_nodes,
    dict other_connections,
    object config
):
    """
    一次性计算两个基因组的完整距离

    将整个距离计算逻辑移到 Cython 中，避免：
    1. Python 层创建临时 numpy 数组
    2. Python 层遍历 dict 构建同源基因列表
    3. 多次 Python-Cython 边界穿越

    距离计算公式（与原始实现保持一致）：
    node_distance = (同源节点距离和 + disjoint_coefficient * 非同源节点数) / max_nodes
    conn_distance = (同源连接距离和 + disjoint_coefficient * 非同源连接数) / max_conns
    total_distance = node_distance + conn_distance

    Args:
        self_nodes: 第一个基因组的节点字典 {node_id: NodeGene}
        self_connections: 第一个基因组的连接字典 {conn_key: ConnectionGene}
        other_nodes: 第二个基因组的节点字典
        other_connections: 第二个基因组的连接字典
        config: 配置对象，需要有以下属性:
            - compatibility_weight_coefficient
            - compatibility_disjoint_coefficient
            - activation_to_int (dict)
            - aggregation_to_int (dict)

    Returns:
        两个基因组的总距离
    """
    cdef double weight_coeff = config.compatibility_weight_coefficient
    cdef double disjoint_coeff = config.compatibility_disjoint_coefficient
    cdef dict act_to_int = config.activation_to_int
    cdef dict agg_to_int = config.aggregation_to_int

    cdef double node_distance = 0.0
    cdef double connection_distance = 0.0
    cdef int disjoint_nodes = 0
    cdef int disjoint_connections = 0
    cdef int max_nodes, max_conns
    cdef int len_self_nodes = len(self_nodes)
    cdef int len_other_nodes = len(other_nodes)
    cdef int len_self_conns = len(self_connections)
    cdef int len_other_conns = len(other_connections)

    cdef object key, n1, n2, c1, c2
    cdef double diff, bias_diff, response_diff, weight_diff
    cdef int act1_int, act2_int, agg1_int, agg2_int

    # ========== 计算节点基因距离 ==========
    if len_self_nodes > 0 or len_other_nodes > 0:
        # 遍历 other_nodes，找出不在 self_nodes 中的节点
        for key in other_nodes:
            if key not in self_nodes:
                disjoint_nodes += 1

        # 遍历 self_nodes，计算同源节点距离或计数非同源节点
        for key in self_nodes:
            n2 = other_nodes.get(key)
            if n2 is None:
                disjoint_nodes += 1
            else:
                n1 = self_nodes[key]
                # 计算节点距离: |bias1 - bias2| + |response1 - response2| + activation_diff + aggregation_diff
                bias_diff = n1.bias - n2.bias
                if bias_diff < 0:
                    bias_diff = -bias_diff

                response_diff = n1.response - n2.response
                if response_diff < 0:
                    response_diff = -response_diff

                diff = bias_diff + response_diff

                # 字符串属性距离（使用整数编码比较）
                act1_int = act_to_int[n1.activation]
                act2_int = act_to_int[n2.activation]
                if act1_int != act2_int:
                    diff += 1.0

                agg1_int = agg_to_int[n1.aggregation]
                agg2_int = agg_to_int[n2.aggregation]
                if agg1_int != agg2_int:
                    diff += 1.0

                node_distance += diff * weight_coeff

        max_nodes = len_self_nodes if len_self_nodes > len_other_nodes else len_other_nodes
        node_distance = (node_distance + disjoint_coeff * disjoint_nodes) / max_nodes

    # ========== 计算连接基因距离 ==========
    if len_self_conns > 0 or len_other_conns > 0:
        # 遍历 other_connections，找出不在 self_connections 中的连接
        for key in other_connections:
            if key not in self_connections:
                disjoint_connections += 1

        # 遍历 self_connections，计算同源连接距离或计数非同源连接
        for key in self_connections:
            c2 = other_connections.get(key)
            if c2 is None:
                disjoint_connections += 1
            else:
                c1 = self_connections[key]
                # 计算连接距离: |weight1 - weight2| + enabled_diff
                weight_diff = c1.weight - c2.weight
                if weight_diff < 0:
                    weight_diff = -weight_diff

                diff = weight_diff
                if c1.enabled != c2.enabled:
                    diff += 1.0

                connection_distance += diff * weight_coeff

        max_conns = len_self_conns if len_self_conns > len_other_conns else len_other_conns
        connection_distance = (connection_distance + disjoint_coeff * disjoint_connections) / max_conns

    return node_distance + connection_distance


# ============================================================================
# 快速交叉操作
# ============================================================================

cpdef void fast_configure_crossover(
    dict child_nodes,
    dict child_connections,
    dict parent1_nodes,
    dict parent1_connections,
    dict parent2_nodes,
    dict parent2_connections,
    object node_gene_type,
    object conn_gene_type,
    list activation_options,
    list aggregation_options
):
    """
    将整个交叉操作移到 Cython 中执行

    直接在 Cython 层遍历父代字典，找出同源基因进行交叉，
    处理 disjoint/excess 基因，创建新基因对象填充子代字典。

    Args:
        child_nodes: 子代节点字典（待填充，应为空字典）
        child_connections: 子代连接字典（待填充，应为空字典）
        parent1_nodes: 较优父代的节点字典
        parent1_connections: 较优父代的连接字典
        parent2_nodes: 较差父代的节点字典
        parent2_connections: 较差父代的连接字典
        node_gene_type: 节点基因类型（DefaultNodeGene 类）
        conn_gene_type: 连接基因类型（DefaultConnectionGene 类）
        activation_options: 可用的 activation 函数列表（未使用，保留兼容性）
        aggregation_options: 可用的 aggregation 函数列表（未使用，保留兼容性）
    """
    cdef double r
    cdef object cg1, cg2, ng1, ng2, new_gene, new_node
    cdef tuple key
    cdef int innovation_num
    cdef int node_key

    # ========== 处理连接基因 ==========
    # 建立 innovation number 到连接基因的映射
    cdef dict parent1_innovations = {}
    cdef dict parent2_innovations = {}

    for key in parent1_connections:
        cg1 = parent1_connections[key]
        parent1_innovations[cg1.innovation] = cg1

    for key in parent2_connections:
        cg2 = parent2_connections[key]
        parent2_innovations[cg2.innovation] = cg2

    # 找出同源和 disjoint innovations
    cdef set p1_innovations_set = set(parent1_innovations.keys())
    cdef set p2_innovations_set = set(parent2_innovations.keys())
    cdef set homologous_innovations_set = p1_innovations_set & p2_innovations_set
    cdef set disjoint_innovations_set = p1_innovations_set - p2_innovations_set

    # 处理同源连接基因
    for innovation_num in homologous_innovations_set:
        cg1 = parent1_innovations[innovation_num]
        cg2 = parent2_innovations[innovation_num]

        # 检查 key 是否相同（处理 innovation 碰撞）
        if cg1.key != cg2.key:
            # Innovation 碰撞：当作 disjoint 处理，只取 parent1
            # 使用 object.__new__() 绕过 __init__ 断言检查
            new_gene = object.__new__(conn_gene_type)
            new_gene.key = cg1.key
            new_gene.innovation = cg1.innovation
            new_gene.weight = cg1.weight
            new_gene.enabled = cg1.enabled
            child_connections[new_gene.key] = new_gene
        else:
            # 同源基因：随机选择属性
            # 使用 object.__new__() 绕过 __init__ 断言检查
            new_gene = object.__new__(conn_gene_type)
            new_gene.key = cg1.key
            new_gene.innovation = innovation_num

            # 随机选择 weight
            r = fast_random()
            if r > 0.5:
                new_gene.weight = cg1.weight
            else:
                new_gene.weight = cg2.weight

            # 随机选择 enabled
            r = fast_random()
            if r > 0.5:
                new_gene.enabled = cg1.enabled
            else:
                new_gene.enabled = cg2.enabled

            # 75% 禁用规则：如果任一父代禁用，75% 概率子代也禁用
            if not cg1.enabled or not cg2.enabled:
                r = fast_random()
                if r < 0.75:
                    new_gene.enabled = False

            child_connections[new_gene.key] = new_gene

    # 处理 disjoint/excess 连接基因（只从 parent1 继承）
    for innovation_num in disjoint_innovations_set:
        cg1 = parent1_innovations[innovation_num]
        # 使用 object.__new__() 绕过 __init__ 断言检查
        new_gene = object.__new__(conn_gene_type)
        new_gene.key = cg1.key
        new_gene.innovation = cg1.innovation
        new_gene.weight = cg1.weight
        new_gene.enabled = cg1.enabled
        child_connections[new_gene.key] = new_gene

    # ========== 处理节点基因 ==========
    cdef set p1_node_keys = set(parent1_nodes.keys())
    cdef set p2_node_keys = set(parent2_nodes.keys())
    cdef set homologous_node_keys = p1_node_keys & p2_node_keys
    cdef set disjoint_node_keys = p1_node_keys - p2_node_keys

    # 处理同源节点基因
    for node_key in homologous_node_keys:
        ng1 = parent1_nodes[node_key]
        ng2 = parent2_nodes[node_key]

        # 使用 object.__new__() 绕过 __init__ 断言检查
        new_node = object.__new__(node_gene_type)
        new_node.key = node_key

        # 随机选择 bias
        r = fast_random()
        if r > 0.5:
            new_node.bias = ng1.bias
        else:
            new_node.bias = ng2.bias

        # 随机选择 response
        r = fast_random()
        if r > 0.5:
            new_node.response = ng1.response
        else:
            new_node.response = ng2.response

        # 随机选择 activation
        r = fast_random()
        if r > 0.5:
            new_node.activation = ng1.activation
        else:
            new_node.activation = ng2.activation

        # 随机选择 aggregation
        r = fast_random()
        if r > 0.5:
            new_node.aggregation = ng1.aggregation
        else:
            new_node.aggregation = ng2.aggregation

        child_nodes[node_key] = new_node

    # 处理 disjoint/excess 节点基因（只从 parent1 继承）
    for node_key in disjoint_node_keys:
        ng1 = parent1_nodes[node_key]
        # 使用 object.__new__() 绕过 __init__ 断言检查
        new_node = object.__new__(node_gene_type)
        new_node.key = node_key
        new_node.bias = ng1.bias
        new_node.response = ng1.response
        new_node.activation = ng1.activation
        new_node.aggregation = ng1.aggregation
        child_nodes[node_key] = new_node

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
向量化基因操作

提供批量基因变异和交叉的 Cython 优化实现。
所有操作使用 NumPy 向量化，避免 Python 循环。
"""

import numpy as np
cimport numpy as np

# NumPy 类型声明
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cpdef void fast_mutate_node_genes(
    np.ndarray[DTYPE_t, ndim=1] biases,
    np.ndarray[DTYPE_t, ndim=1] responses,
    double bias_mutate_rate,
    double bias_replace_rate,
    double bias_mutate_power,
    double bias_min,
    double bias_max,
    double bias_init_mean,
    double bias_init_stdev,
    double response_mutate_rate,
    double response_replace_rate,
    double response_mutate_power,
    double response_min,
    double response_max,
    double response_init_mean,
    double response_init_stdev
):
    """
    批量变异节点基因的 bias 和 response 属性

    使用向量化操作同时处理多个节点基因的变异。
    遵循 NEAT 的变异逻辑：
    1. 按 mutate_rate 概率进行高斯扰动
    2. 按 replace_rate 概率用新初始化值替换
    3. 确保值在 [min, max] 范围内

    Args:
        biases: 偏置数组 (in-place 修改)
        responses: 响应系数数组 (in-place 修改)
        bias_mutate_rate: bias 高斯扰动概率
        bias_replace_rate: bias 替换概率
        bias_mutate_power: bias 高斯扰动标准差
        bias_min: bias 最小值
        bias_max: bias 最大值
        bias_init_mean: bias 初始化均值
        bias_init_stdev: bias 初始化标准差
        response_mutate_rate: response 高斯扰动概率
        response_replace_rate: response 替换概率
        response_mutate_power: response 高斯扰动标准差
        response_min: response 最小值
        response_max: response 最大值
        response_init_mean: response 初始化均值
        response_init_stdev: response 初始化标准差
    """
    cdef int n = biases.shape[0]
    if n == 0:
        return

    # 生成随机数
    cdef np.ndarray[DTYPE_t, ndim=1] rand_bias = np.random.random(n)
    cdef np.ndarray[DTYPE_t, ndim=1] rand_response = np.random.random(n)
    cdef np.ndarray[DTYPE_t, ndim=1] gauss_bias = np.random.normal(0, bias_mutate_power, n)
    cdef np.ndarray[DTYPE_t, ndim=1] gauss_response = np.random.normal(0, response_mutate_power, n)

    # --- Bias 变异 ---
    # 1. 高斯扰动 (rand < mutate_rate)
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] mutate_mask = rand_bias < bias_mutate_rate
    biases[mutate_mask] = biases[mutate_mask] + gauss_bias[mutate_mask]

    # 2. 替换 (mutate_rate <= rand < mutate_rate + replace_rate)
    cdef double bias_total_rate = bias_mutate_rate + bias_replace_rate
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] replace_mask = (rand_bias >= bias_mutate_rate) & (rand_bias < bias_total_rate)
    cdef int replace_count = np.sum(replace_mask)
    if replace_count > 0:
        biases[replace_mask] = np.random.normal(bias_init_mean, bias_init_stdev, replace_count)

    # 3. 裁剪到范围
    np.clip(biases, bias_min, bias_max, out=biases)

    # --- Response 变异 ---
    # 1. 高斯扰动
    mutate_mask = rand_response < response_mutate_rate
    responses[mutate_mask] = responses[mutate_mask] + gauss_response[mutate_mask]

    # 2. 替换
    cdef double response_total_rate = response_mutate_rate + response_replace_rate
    replace_mask = (rand_response >= response_mutate_rate) & (rand_response < response_total_rate)
    replace_count = np.sum(replace_mask)
    if replace_count > 0:
        responses[replace_mask] = np.random.normal(response_init_mean, response_init_stdev, replace_count)

    # 3. 裁剪到范围
    np.clip(responses, response_min, response_max, out=responses)


cpdef void fast_mutate_connection_genes(
    np.ndarray[DTYPE_t, ndim=1] weights,
    np.ndarray[np.uint8_t, ndim=1] enabled,
    double weight_mutate_rate,
    double weight_replace_rate,
    double weight_mutate_power,
    double weight_min,
    double weight_max,
    double weight_init_mean,
    double weight_init_stdev,
    double enabled_mutate_rate,
    double enabled_rate_to_true,
    double enabled_rate_to_false
):
    """
    批量变异连接基因的 weight 和 enabled 属性

    使用向量化操作同时处理多个连接基因的变异。
    遵循 NEAT 的变异逻辑：
    - weight: 高斯扰动或替换
    - enabled: 按条件概率翻转

    Args:
        weights: 权重数组 (in-place 修改)
        enabled: 启用状态数组 (in-place 修改, uint8 作为 bool)
        weight_mutate_rate: weight 高斯扰动概率
        weight_replace_rate: weight 替换概率
        weight_mutate_power: weight 高斯扰动标准差
        weight_min: weight 最小值
        weight_max: weight 最大值
        weight_init_mean: weight 初始化均值
        weight_init_stdev: weight 初始化标准差
        enabled_mutate_rate: enabled 基础变异概率
        enabled_rate_to_true: 当 enabled=False 时额外增加的变异概率
        enabled_rate_to_false: 当 enabled=True 时额外增加的变异概率
    """
    cdef int n = weights.shape[0]
    if n == 0:
        return

    # 生成随机数
    cdef np.ndarray[DTYPE_t, ndim=1] rand_weight = np.random.random(n)
    cdef np.ndarray[DTYPE_t, ndim=1] rand_enabled = np.random.random(n)
    cdef np.ndarray[DTYPE_t, ndim=1] rand_flip = np.random.random(n)
    cdef np.ndarray[DTYPE_t, ndim=1] gauss_weight = np.random.normal(0, weight_mutate_power, n)

    # --- Weight 变异 ---
    # 1. 高斯扰动 (rand < mutate_rate)
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] mutate_mask = rand_weight < weight_mutate_rate
    weights[mutate_mask] = weights[mutate_mask] + gauss_weight[mutate_mask]

    # 2. 替换 (mutate_rate <= rand < mutate_rate + replace_rate)
    cdef double weight_total_rate = weight_mutate_rate + weight_replace_rate
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] replace_mask = (rand_weight >= weight_mutate_rate) & (rand_weight < weight_total_rate)
    cdef int replace_count = np.sum(replace_mask)
    if replace_count > 0:
        weights[replace_mask] = np.random.normal(weight_init_mean, weight_init_stdev, replace_count)

    # 3. 裁剪到范围
    np.clip(weights, weight_min, weight_max, out=weights)

    # --- Enabled 变异 ---
    # 计算每个连接的实际变异概率
    # 当 enabled=True 时: mutate_rate + rate_to_false
    # 当 enabled=False 时: mutate_rate + rate_to_true
    cdef np.ndarray[DTYPE_t, ndim=1] actual_rate = np.full(n, enabled_mutate_rate, dtype=DTYPE)
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] enabled_mask = enabled.astype(np.bool_)

    # enabled=True 的增加 rate_to_false
    actual_rate[enabled_mask] = actual_rate[enabled_mask] + enabled_rate_to_false
    # enabled=False 的增加 rate_to_true
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] disabled_mask = ~enabled_mask
    actual_rate[disabled_mask] = actual_rate[disabled_mask] + enabled_rate_to_true

    # 按概率触发变异
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] should_mutate = rand_enabled < actual_rate

    # 变异时随机选择新值 (random() < 0.5 为 True)
    cdef np.ndarray[np.uint8_t, ndim=1] new_values = (rand_flip < 0.5).astype(np.uint8)
    enabled[should_mutate] = new_values[should_mutate]


cpdef tuple fast_crossover_genes(
    np.ndarray[DTYPE_t, ndim=1] parent1_weights,
    np.ndarray[DTYPE_t, ndim=1] parent2_weights,
    np.ndarray[np.uint8_t, ndim=1] parent1_enabled,
    np.ndarray[np.uint8_t, ndim=1] parent2_enabled
):
    """
    批量交叉连接基因

    从两个父代基因组中随机选择基因属性，
    并应用 NEAT 的 75% 禁用规则。

    NEAT 禁用规则：如果任一父代的基因被禁用，
    则有 75% 的概率子代也禁用该基因。

    Args:
        parent1_weights: 父代1的权重数组
        parent2_weights: 父代2的权重数组 (长度必须与父代1相同)
        parent1_enabled: 父代1的启用状态数组
        parent2_enabled: 父代2的启用状态数组

    Returns:
        (new_weights, new_enabled): 子代的权重和启用状态数组
    """
    cdef int n = parent1_weights.shape[0]
    if n == 0:
        return (
            np.array([], dtype=DTYPE),
            np.array([], dtype=np.uint8)
        )

    # 随机选择继承自哪个父代 (random() > 0.5 选择 parent1)
    cdef np.ndarray[DTYPE_t, ndim=1] rand = np.random.random(n)
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] choices = rand > 0.5

    # 向量化选择 weight
    cdef np.ndarray[DTYPE_t, ndim=1] new_weights = np.where(
        choices, parent1_weights, parent2_weights
    )

    # 向量化选择 enabled
    cdef np.ndarray[np.uint8_t, ndim=1] new_enabled = np.where(
        choices, parent1_enabled, parent2_enabled
    ).astype(np.uint8)

    # 75% 禁用规则：
    # 如果任一父代禁用 (~p1_enabled | ~p2_enabled)，按 75% 概率禁用子代
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] p1_enabled = parent1_enabled.astype(np.bool_)
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] p2_enabled = parent2_enabled.astype(np.bool_)
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] either_disabled = (~p1_enabled) | (~p2_enabled)

    cdef np.ndarray[DTYPE_t, ndim=1] rand_disable = np.random.random(n)
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] disable_mask = either_disabled & (rand_disable < 0.75)

    new_enabled[disable_mask] = 0

    return (new_weights, new_enabled)


cpdef tuple fast_crossover_node_genes(
    np.ndarray[DTYPE_t, ndim=1] parent1_biases,
    np.ndarray[DTYPE_t, ndim=1] parent2_biases,
    np.ndarray[DTYPE_t, ndim=1] parent1_responses,
    np.ndarray[DTYPE_t, ndim=1] parent2_responses
):
    """
    批量交叉节点基因

    从两个父代基因组中随机选择节点属性 (bias, response)。

    注意：activation 和 aggregation 为字符串类型，
    无法向量化处理，需在 Python 层单独处理。

    Args:
        parent1_biases: 父代1的偏置数组
        parent2_biases: 父代2的偏置数组
        parent1_responses: 父代1的响应系数数组
        parent2_responses: 父代2的响应系数数组

    Returns:
        (new_biases, new_responses): 子代的偏置和响应系数数组
    """
    cdef int n = parent1_biases.shape[0]
    if n == 0:
        return (
            np.array([], dtype=DTYPE),
            np.array([], dtype=DTYPE)
        )

    # 为每个属性独立选择 (遵循原始 NEAT 逻辑)
    cdef np.ndarray[DTYPE_t, ndim=1] rand_bias = np.random.random(n)
    cdef np.ndarray[DTYPE_t, ndim=1] rand_response = np.random.random(n)

    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] choice_bias = rand_bias > 0.5
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] choice_response = rand_response > 0.5

    cdef np.ndarray[DTYPE_t, ndim=1] new_biases = np.where(
        choice_bias, parent1_biases, parent2_biases
    )
    cdef np.ndarray[DTYPE_t, ndim=1] new_responses = np.where(
        choice_response, parent1_responses, parent2_responses
    )

    return (new_biases, new_responses)


cpdef np.ndarray[DTYPE_t, ndim=1] fast_init_float_genes(
    int n,
    double init_mean,
    double init_stdev,
    double min_value,
    double max_value,
    str init_type = "gaussian"
):
    """
    批量初始化浮点型基因值

    Args:
        n: 基因数量
        init_mean: 初始化均值
        init_stdev: 初始化标准差
        min_value: 最小值
        max_value: 最大值
        init_type: 初始化类型 ("gaussian" 或 "uniform")

    Returns:
        初始化后的值数组
    """
    if n <= 0:
        return np.array([], dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] values
    cdef double low, high

    if "uniform" in init_type.lower():
        # 均匀分布：在 [mean - 2*stdev, mean + 2*stdev] 内
        low = max(min_value, init_mean - 2 * init_stdev)
        high = min(max_value, init_mean + 2 * init_stdev)
        values = np.random.uniform(low, high, n)
    else:
        # 高斯分布
        values = np.random.normal(init_mean, init_stdev, n)
        np.clip(values, min_value, max_value, out=values)

    return values


cpdef np.ndarray[np.uint8_t, ndim=1] fast_init_bool_genes(
    int n,
    str default_value = "random"
):
    """
    批量初始化布尔型基因值

    Args:
        n: 基因数量
        default_value: 默认值 ("true", "false", "random")

    Returns:
        初始化后的值数组 (uint8 类型, 0 或 1)
    """
    if n <= 0:
        return np.array([], dtype=np.uint8)

    cdef str lower = default_value.lower()

    if lower in ('1', 'on', 'yes', 'true'):
        return np.ones(n, dtype=np.uint8)
    elif lower in ('0', 'off', 'no', 'false'):
        return np.zeros(n, dtype=np.uint8)
    else:
        # random 或 none
        return (np.random.random(n) < 0.5).astype(np.uint8)


cpdef double fast_node_distance(
    np.ndarray[DTYPE_t, ndim=1] biases1,
    np.ndarray[DTYPE_t, ndim=1] biases2,
    np.ndarray[DTYPE_t, ndim=1] responses1,
    np.ndarray[DTYPE_t, ndim=1] responses2,
    double compatibility_weight_coefficient
):
    """
    批量计算节点基因距离

    计算两组节点基因的总距离 (不包括 activation/aggregation 差异)。

    Args:
        biases1: 第一组节点的偏置
        biases2: 第二组节点的偏置
        responses1: 第一组节点的响应系数
        responses2: 第二组节点的响应系数
        compatibility_weight_coefficient: 兼容性权重系数

    Returns:
        总距离
    """
    cdef int n = biases1.shape[0]
    if n == 0:
        return 0.0

    cdef double bias_diff = np.sum(np.abs(biases1 - biases2))
    cdef double response_diff = np.sum(np.abs(responses1 - responses2))

    return (bias_diff + response_diff) * compatibility_weight_coefficient


cpdef double fast_connection_distance(
    np.ndarray[DTYPE_t, ndim=1] weights1,
    np.ndarray[DTYPE_t, ndim=1] weights2,
    np.ndarray[np.uint8_t, ndim=1] enabled1,
    np.ndarray[np.uint8_t, ndim=1] enabled2,
    double compatibility_weight_coefficient
):
    """
    批量计算连接基因距离

    计算两组连接基因的总距离。

    Args:
        weights1: 第一组连接的权重
        weights2: 第二组连接的权重
        enabled1: 第一组连接的启用状态
        enabled2: 第二组连接的启用状态
        compatibility_weight_coefficient: 兼容性权重系数

    Returns:
        总距离
    """
    cdef int n = weights1.shape[0]
    if n == 0:
        return 0.0

    cdef double weight_diff = np.sum(np.abs(weights1 - weights2))

    # enabled 差异：不同则 +1
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] enabled_diff = enabled1 != enabled2
    cdef double enabled_count = np.sum(enabled_diff)

    return (weight_diff + enabled_count) * compatibility_weight_coefficient


cpdef double fast_string_distance(
    np.ndarray[np.int32_t, ndim=1] acts1,
    np.ndarray[np.int32_t, ndim=1] acts2,
    np.ndarray[np.int32_t, ndim=1] aggs1,
    np.ndarray[np.int32_t, ndim=1] aggs2,
    double weight_coeff
):
    """
    计算字符串属性差异的距离

    用于计算 activation 和 aggregation 属性的差异。
    字符串已预先编码为整数，避免字符串比较开销。

    Args:
        acts1: 第一组节点的 activation 编码
        acts2: 第二组节点的 activation 编码
        aggs1: 第一组节点的 aggregation 编码
        aggs2: 第二组节点的 aggregation 编码
        weight_coeff: 兼容性权重系数

    Returns:
        总距离（activation 和 aggregation 差异数 * weight_coeff）
    """
    cdef int n = acts1.shape[0]
    if n == 0:
        return 0.0

    cdef double dist = 0.0
    cdef int i

    for i in range(n):
        if acts1[i] != acts2[i]:
            dist += weight_coeff
        if aggs1[i] != aggs2[i]:
            dist += weight_coeff

    return dist


cpdef double fast_full_node_distance(
    list nodes1,
    list nodes2,
    dict act_to_int,
    dict agg_to_int,
    double weight_coeff
):
    """
    一次性计算所有节点属性的距离（数值 + 字符串）

    将数值属性和字符串属性的距离计算合并到一个函数中，
    避免多次 Python-Cython 边界穿越和重复的列表遍历。

    Args:
        nodes1: 第一组节点对象列表
        nodes2: 第二组节点对象列表（长度必须与 nodes1 相同）
        act_to_int: activation 名称到整数的映射
        agg_to_int: aggregation 名称到整数的映射
        weight_coeff: 兼容性权重系数

    Returns:
        总节点距离
    """
    cdef int n = len(nodes1)
    if n == 0:
        return 0.0

    cdef double dist = 0.0
    cdef int i
    cdef object n1, n2
    cdef int act1_int, act2_int, agg1_int, agg2_int

    for i in range(n):
        n1 = nodes1[i]
        n2 = nodes2[i]

        # 数值属性距离
        dist += abs(n1.bias - n2.bias) * weight_coeff
        dist += abs(n1.response - n2.response) * weight_coeff

        # 字符串属性距离（使用整数编码比较）
        act1_int = act_to_int[n1.activation]
        act2_int = act_to_int[n2.activation]
        if act1_int != act2_int:
            dist += weight_coeff

        agg1_int = agg_to_int[n1.aggregation]
        agg2_int = agg_to_int[n2.aggregation]
        if agg1_int != agg2_int:
            dist += weight_coeff

    return dist

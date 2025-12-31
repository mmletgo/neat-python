# cython: language_level=3
"""
Cython 优化模块

提供 NEAT 算法热点函数的 Cython 优化实现。
优化模块包括：
- random_utils: 批量随机数生成缓存
- fast_attributes: 向量化属性变异
- fast_genes: 向量化基因操作
- fast_genome: 向量化基因组变异
- fast_graphs: 优化图算法
"""

# 尝试导入 Cython 模块
_CYTHON_AVAILABLE = False

try:
    from neat._cython.random_utils import (
        fast_random,
        fast_gauss,
        get_batch_uniform,
        get_batch_gaussian,
        reset_random_cache,
        fast_random_choice,
        fast_uniform,
        get_batch_uniform_range,
        create_random_cache,
        RandomCache,
    )
    from neat._cython.fast_attributes import (
        fast_float_mutate,
        fast_float_mutate_batch,
        fast_float_init_batch,
        fast_int_mutate,
        fast_int_mutate_batch,
        fast_bool_mutate,
        fast_bool_mutate_batch,
        fast_bool_init_batch,
        fast_mutate_genome_floats,
    )
    from neat._cython.fast_genes import (
        fast_mutate_node_genes,
        fast_mutate_connection_genes,
        fast_crossover_genes,
        fast_crossover_node_genes,
        fast_init_float_genes,
        fast_init_bool_genes,
        fast_node_distance,
        fast_connection_distance,
    )
    from neat._cython.fast_genome import (
        MutationConfig,
        fast_mutate_genome,
        fast_mutate_node_genes as fast_mutate_node_genes_genome,
        fast_mutate_connection_genes as fast_mutate_connection_genes_genome,
        fast_genome_distance,
        fast_node_distance as fast_node_distance_genome,
        fast_mutate_population,
    )
    from neat._cython.fast_graphs import (
        fast_creates_cycle,
        fast_required_for_output,
        fast_feed_forward_layers,
    )
    from neat._cython.fast_network import FastFeedForwardNetwork
    _CYTHON_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """检查 Cython 优化模块是否可用"""
    return _CYTHON_AVAILABLE

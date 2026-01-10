# cython: language_level=3
"""
fast_reproduction.pyx 的声明文件

允许其他 Cython 模块 cimport 这里定义的类型和函数。
"""

cimport numpy as np


# ReproductionConfig 类声明
cdef class ReproductionConfig:
    cdef public int elitism
    cdef public double survival_threshold
    cdef public int min_species_size


# 核心函数声明
cpdef list compute_spawn_fast(
    np.ndarray adjusted_fitness,
    np.ndarray previous_sizes,
    int pop_size,
    int min_species_size
)

cpdef list adjust_spawn_exact_fast(
    list spawn_amounts,
    int pop_size,
    int min_species_size
)

cpdef tuple compute_adjusted_fitness_fast(
    list species_list,
    list all_fitnesses
)

cpdef dict reproduce_species_fast(
    object species,
    int spawn_count,
    int elitism,
    double survival_threshold,
    object genome_indexer,
    object config,
    object mutation_config,
    dict ancestors
)

cpdef dict reproduce_fast(
    object reproduction,
    object config,
    object species_set,
    int pop_size,
    int generation
)

cpdef dict create_new_genomes_fast(
    object genome_type,
    object genome_config,
    object genome_indexer,
    int num_genomes,
    dict ancestors
)

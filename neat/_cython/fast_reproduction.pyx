# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
高性能繁殖模块

使用 Cython 实现 NEAT 算法的繁殖过程优化。
主要优化：
1. ReproductionConfig 缓存所有繁殖配置参数
2. 向量化 spawn 数量计算
3. 批量创建后代基因组
4. 利用已有的 fast_genome 优化函数

核心流程：
1. 停滞检查 - 移除停滞的 species
2. 计算调整后适应度 - 归一化到 [0, 1]
3. 计算 spawn 数量 - 每个 species 的后代数量
4. 精英保留 - 保留最优个体
5. 选择父代 - 使用 survival_threshold 选择
6. 交叉产生后代 - configure_crossover
7. 变异 - mutate
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport ceil, floor
import math
import random

# 导入已有的 fast_genome 优化函数
from neat._cython.fast_genome import (
    MutationConfig,
    fast_configure_crossover,
    fast_mutate_genome,
)

# NumPy 类型声明
DTYPE = np.float64

# 常量
cdef double RAND_MAX_INV = 1.0 / <double>RAND_MAX


# ============================================================================
# 辅助函数 - 用于替代 lambda
# ============================================================================

def _get_fitness_key(item):
    """用于排序的 fitness 提取函数"""
    return (item[1].fitness, item[0])


def _get_spawn_value(item):
    """用于排序的 spawn 值提取函数"""
    return item[1]


def _get_spawn_value_reverse(item):
    """用于反向排序的 spawn 值提取函数（返回负值）"""
    return -item[1]


def _sum_member_fitness(members):
    """计算成员适应度总和"""
    cdef double total = 0.0
    for m in members.values():
        total += m.fitness
    return total


def _extend_fitnesses(all_fitnesses, members):
    """将成员适应度添加到列表"""
    for m in members.values():
        all_fitnesses.append(m.fitness)


def _get_member_sizes(remaining_species):
    """获取各 species 的成员数量列表"""
    return [len(s.members) for s in remaining_species]


# ============================================================================
# 随机数生成辅助函数
# ============================================================================

cdef inline double fast_random() noexcept nogil:
    """快速生成 [0, 1) 均匀分布随机数"""
    return <double>rand() * RAND_MAX_INV


cdef inline int fast_randint(int low, int high) noexcept nogil:
    """快速生成 [low, high) 区间的随机整数"""
    return low + <int>(fast_random() * (high - low))


# ============================================================================
# ReproductionConfig - 缓存繁殖配置参数
# ============================================================================

cdef class ReproductionConfig:
    """
    缓存所有繁殖配置参数的类

    避免每次繁殖时通过 getattr 动态获取配置，
    在创建时一次性从 NEAT 配置对象提取所有参数。
    """
    # 属性在 .pxd 文件中声明

    def __init__(self):
        """初始化默认值"""
        self.elitism = 0
        self.survival_threshold = 0.2
        self.min_species_size = 1

    @staticmethod
    def from_neat_config(reproduction_config):
        """
        从 NEAT 繁殖配置对象提取所有参数

        Args:
            reproduction_config: DefaultReproduction 的配置对象

        Returns:
            ReproductionConfig 实例
        """
        cdef ReproductionConfig rc = ReproductionConfig()

        rc.elitism = getattr(reproduction_config, 'elitism', 0)
        rc.survival_threshold = getattr(reproduction_config, 'survival_threshold', 0.2)
        rc.min_species_size = getattr(reproduction_config, 'min_species_size', 1)

        return rc


# ============================================================================
# 向量化 spawn 数量计算
# ============================================================================

cpdef list compute_spawn_fast(
    np.ndarray adjusted_fitness,
    np.ndarray previous_sizes,
    int pop_size,
    int min_species_size
):
    """
    向量化计算每个 species 的 spawn 数量

    实现与原始 compute_spawn 相同的逻辑：
    1. 根据适应度比例分配后代数量
    2. 使用渐进式调整（50% 靠近目标）
    3. 归一化到目标种群大小

    Args:
        adjusted_fitness: 调整后的适应度数组
        previous_sizes: 上一代各 species 的大小数组
        pop_size: 目标种群大小
        min_species_size: 每个 species 的最小大小

    Returns:
        spawn 数量列表
    """
    cdef int n = adjusted_fitness.shape[0]
    if n == 0:
        return []

    cdef double af_sum = 0.0
    cdef int i
    cdef double af, s, d
    cdef int c, ps
    cdef np.ndarray spawn_raw = np.empty(n, dtype=DTYPE)
    cdef np.ndarray spawn_amounts = np.empty(n, dtype=np.int32)

    # 计算适应度总和
    for i in range(n):
        af_sum += adjusted_fitness[i]

    # 计算每个 species 的 spawn 数量
    for i in range(n):
        af = adjusted_fitness[i]
        ps = previous_sizes[i]

        if af_sum > 0:
            s = af / af_sum * pop_size
            if s < min_species_size:
                s = <double>min_species_size
        else:
            s = <double>min_species_size

        # 渐进式调整：向目标移动 50%
        d = (s - <double>ps) * 0.5
        c = <int>(d + 0.5)  # 四舍五入

        spawn_raw[i] = <double>ps
        if c > 0:
            spawn_raw[i] += <double>c
        elif c < 0:
            spawn_raw[i] += <double>c
        elif d > 0:
            spawn_raw[i] += 1.0
        elif d < 0:
            spawn_raw[i] -= 1.0

    # 归一化到目标种群大小
    cdef double total_spawn = 0.0
    for i in range(n):
        total_spawn += spawn_raw[i]

    cdef double norm = <double>pop_size / total_spawn

    for i in range(n):
        spawn_amounts[i] = <int>(spawn_raw[i] * norm + 0.5)
        if spawn_amounts[i] < min_species_size:
            spawn_amounts[i] = min_species_size

    return list(spawn_amounts)


cpdef list adjust_spawn_exact_fast(
    list spawn_amounts,
    int pop_size,
    int min_species_size
):
    """
    调整 spawn 数量使总和精确等于 pop_size

    保持每个 species 至少有 min_species_size 个个体。

    Args:
        spawn_amounts: spawn 数量列表（会被复制，不修改原始数据）
        pop_size: 目标种群大小
        min_species_size: 每个 species 的最小大小

    Returns:
        调整后的 spawn 数量列表
    """
    cdef list result = list(spawn_amounts)
    cdef int num_species = len(result)

    if num_species == 0:
        return result

    cdef int total_spawn = sum(result)
    if total_spawn == pop_size:
        return result

    cdef int min_total = num_species * min_species_size
    if min_total > pop_size:
        raise RuntimeError(
            f"Configuration conflict: population size {pop_size} is less than "
            f"num_species * min_species_size {min_total} ({num_species} * {min_species_size}). "
            "Cannot satisfy per-species minima."
        )

    cdef int diff = pop_size - total_spawn
    cdef list indexed = [(i, result[i]) for i in range(num_species)]
    cdef int i, idx, val, remaining

    if diff > 0:
        # 太少：给较小的 species 增加
        indexed.sort(key=_get_spawn_value)
        i = 0
        while diff > 0:
            idx, val = indexed[i]
            val += 1
            result[idx] = val
            indexed[i] = (idx, val)
            diff -= 1
            i = (i + 1) % len(indexed)
    else:
        # 太多：从较大的 species 减少
        remaining = -diff
        indexed.sort(key=_get_spawn_value, reverse=True)
        i = 0
        while remaining > 0 and indexed:
            idx, val = indexed[i]
            if val > min_species_size:
                val -= 1
                result[idx] = val
                indexed[i] = (idx, val)
                remaining -= 1
            i = (i + 1) % len(indexed)
            if i == 0 and remaining > 0:
                break

    return result


# ============================================================================
# 快速适应度计算
# ============================================================================

cpdef tuple compute_adjusted_fitness_fast(
    list species_list,
    list all_fitnesses
):
    """
    快速计算调整后的适应度

    使用向量化操作计算每个 species 的调整后适应度。

    Args:
        species_list: species 对象列表
        all_fitnesses: 所有成员的适应度列表

    Returns:
        (adjusted_fitnesses 数组, avg_adjusted_fitness)
    """
    if not all_fitnesses:
        return np.array([], dtype=DTYPE), 0.0

    cdef double min_fitness = min(all_fitnesses)
    cdef double max_fitness = max(all_fitnesses)
    cdef double fitness_range = max_fitness - min_fitness
    if fitness_range < 1.0:
        fitness_range = 1.0

    cdef int n = len(species_list)
    cdef np.ndarray adjusted_fitnesses = np.empty(n, dtype=DTYPE)
    cdef double msf, af
    cdef int i
    cdef object species_obj, members

    for i in range(n):
        # 计算 species 成员的平均适应度
        species_obj = species_list[i]
        members = species_obj.members
        msf = _sum_member_fitness(members) / len(members)
        # 归一化到 [0, 1]
        af = (msf - min_fitness) / fitness_range
        adjusted_fitnesses[i] = af
        species_obj.adjusted_fitness = af

    cdef double avg_adjusted_fitness = np.mean(adjusted_fitnesses)

    return adjusted_fitnesses, avg_adjusted_fitness


# ============================================================================
# 单 Species 繁殖核心函数
# ============================================================================

cpdef dict reproduce_species_fast(
    object species,
    int spawn_count,
    int elitism,
    double survival_threshold,
    object genome_indexer,
    object config,
    object mutation_config,
    dict ancestors
):
    """
    快速繁殖单个 species

    使用 Cython 优化的交叉和变异操作。

    Args:
        species: species 对象
        spawn_count: 需要产生的后代数量
        elitism: 精英保留数量
        survival_threshold: 存活阈值
        genome_indexer: 基因组 ID 生成器
        config: NEAT 配置
        mutation_config: Cython 变异配置
        ancestors: 祖先记录字典

    Returns:
        新种群字典 {genome_id: genome}
    """
    cdef dict new_population = {}
    cdef list old_members
    cdef int i, repro_cutoff
    cdef object parent1, parent2, child
    cdef int parent1_id, parent2_id, gid

    # 确保至少产生 elitism 数量的后代
    if spawn_count < elitism:
        spawn_count = elitism

    if spawn_count <= 0:
        return new_population

    # 按适应度降序排序成员
    old_members = list(species.members.items())
    old_members.sort(reverse=True, key=_get_fitness_key)

    # 精英保留
    if elitism > 0:
        for i in range(min(elitism, len(old_members))):
            member_id, member = old_members[i]
            new_population[member_id] = member
            spawn_count -= 1

    if spawn_count <= 0:
        return new_population

    # 计算繁殖截止线
    repro_cutoff = <int>ceil(survival_threshold * len(old_members))
    if repro_cutoff < 2:
        repro_cutoff = 2
    old_members = old_members[:repro_cutoff]

    # 随机选择父代产生后代
    while spawn_count > 0:
        spawn_count -= 1

        parent1_id, parent1 = random.choice(old_members)
        parent2_id, parent2 = random.choice(old_members)

        gid = next(genome_indexer)
        child = config.genome_type(gid)

        # 使用 Cython 优化的交叉
        if parent1.fitness > parent2.fitness:
            _fast_crossover(child, parent1, parent2, config.genome_config)
        else:
            _fast_crossover(child, parent2, parent1, config.genome_config)

        # 使用 Cython 优化的变异
        _fast_mutate(child, config.genome_config, mutation_config)

        new_population[gid] = child
        ancestors[gid] = (parent1_id, parent2_id)

    return new_population


cdef void _fast_crossover(
    object child,
    object parent1,
    object parent2,
    object genome_config
):
    """
    使用 fast_genome 的交叉函数
    """
    fast_configure_crossover(
        child.nodes,
        child.connections,
        parent1.nodes,
        parent1.connections,
        parent2.nodes,
        parent2.connections,
        genome_config.node_gene_type,
        genome_config.connection_gene_type,
        genome_config.activation_options,
        genome_config.aggregation_options
    )


cdef void _fast_mutate(
    object genome,
    object genome_config,
    object mutation_config
):
    """
    使用 fast_genome 的变异函数

    包括结构变异和属性变异。
    """
    # 结构变异（仍使用原始实现，因为涉及复杂的拓扑操作）
    cdef double div, r
    if genome_config.single_structural_mutation:
        div = max(1, (genome_config.node_add_prob + genome_config.node_delete_prob +
                      genome_config.conn_add_prob + genome_config.conn_delete_prob))
        r = random.random()
        if r < (genome_config.node_add_prob / div):
            genome.mutate_add_node(genome_config)
        elif r < ((genome_config.node_add_prob + genome_config.node_delete_prob) / div):
            genome.mutate_delete_node(genome_config)
        elif r < ((genome_config.node_add_prob + genome_config.node_delete_prob +
                   genome_config.conn_add_prob) / div):
            genome.mutate_add_connection(genome_config)
        elif r < ((genome_config.node_add_prob + genome_config.node_delete_prob +
                   genome_config.conn_add_prob + genome_config.conn_delete_prob) / div):
            genome.mutate_delete_connection()
    else:
        if random.random() < genome_config.node_add_prob:
            genome.mutate_add_node(genome_config)
        if random.random() < genome_config.node_delete_prob:
            genome.mutate_delete_node(genome_config)
        if random.random() < genome_config.conn_add_prob:
            genome.mutate_add_connection(genome_config)
        if random.random() < genome_config.conn_delete_prob:
            genome.mutate_delete_connection()

    # 属性变异（使用 Cython 优化）
    if mutation_config is not None:
        fast_mutate_genome(genome.nodes, genome.connections, mutation_config)
    else:
        # 回退到原始实现
        for cg in genome.connections.values():
            cg.mutate(genome_config)
        for ng in genome.nodes.values():
            ng.mutate(genome_config)


# ============================================================================
# 主繁殖函数
# ============================================================================

cpdef dict reproduce_fast(
    object reproduction,
    object config,
    object species_set,
    int pop_size,
    int generation
):
    """
    Cython 优化的繁殖函数

    完整实现 DefaultReproduction.reproduce() 的功能，
    使用向量化操作和 Cython 加速。

    Args:
        reproduction: DefaultReproduction 实例
        config: NEAT 配置
        species_set: species 集合
        pop_size: 目标种群大小
        generation: 当前代数

    Returns:
        新种群字典 {genome_id: genome}
    """
    # 设置 innovation tracker
    config.genome_config.innovation_tracker = reproduction.innovation_tracker
    reproduction.innovation_tracker.reset_generation()

    # 停滞检查
    cdef list all_fitnesses = []
    cdef list remaining_species = []
    cdef object stag_sid, stag_s
    cdef bint stagnant

    for stag_sid, stag_s, stagnant in reproduction.stagnation.update(species_set, generation):
        if stagnant:
            reproduction.reporters.species_stagnant(stag_sid, stag_s)
        else:
            _extend_fitnesses(all_fitnesses, stag_s.members)
            remaining_species.append(stag_s)

    # 没有剩余 species
    if not remaining_species:
        species_set.species = {}
        return {}

    # 计算调整后适应度
    cdef np.ndarray adjusted_fitnesses
    cdef double avg_adjusted_fitness
    adjusted_fitnesses, avg_adjusted_fitness = compute_adjusted_fitness_fast(
        remaining_species, all_fitnesses
    )
    reproduction.reporters.info(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

    # 计算 spawn 数量
    cdef int n_species = len(remaining_species)
    cdef np.ndarray previous_sizes = np.array(
        _get_member_sizes(remaining_species), dtype=np.int32
    )
    cdef int min_species_size = reproduction.reproduction_config.min_species_size
    cdef int elitism_val = reproduction.reproduction_config.elitism
    min_species_size = max(min_species_size, elitism_val)

    cdef list spawn_amounts = compute_spawn_fast(
        adjusted_fitnesses, previous_sizes, pop_size, min_species_size
    )
    spawn_amounts = adjust_spawn_exact_fast(spawn_amounts, pop_size, min_species_size)

    # 获取或创建 MutationConfig
    cdef object mutation_config = None
    if hasattr(config.genome_config, '_mutation_config'):
        mutation_config = config.genome_config._mutation_config

    # 繁殖
    cdef dict new_population = {}
    cdef double survival_threshold = reproduction.reproduction_config.survival_threshold
    cdef object s
    cdef int spawn
    cdef dict species_offspring
    cdef list old_members
    cdef int i

    species_set.species = {}

    for i in range(n_species):
        s = remaining_species[i]
        spawn = spawn_amounts[i]

        # 确保至少产生 elitism 数量的后代
        if spawn < elitism_val:
            spawn = elitism_val

        if spawn <= 0:
            continue

        # 先繁殖（需要读取 s.members）
        species_offspring = reproduce_species_fast(
            s,
            spawn,
            elitism_val,
            survival_threshold,
            reproduction.genome_indexer,
            config,
            mutation_config,
            reproduction.ancestors
        )
        new_population.update(species_offspring)

        # 繁殖完成后，清空并更新 species
        s.members = {}
        species_set.species[s.key] = s

    return new_population


# ============================================================================
# 批量繁殖（用于多种群并行进化）
# ============================================================================

def reproduce_populations_batch(
    list reproductions,
    list configs,
    list species_sets,
    list pop_sizes,
    int generation
):
    """
    批量繁殖多个种群

    由于 NEAT 进化涉及大量 Python 对象操作，
    完全的 nogil 并行不可行。
    这个函数提供串行但优化的批量繁殖。

    Args:
        reproductions: DefaultReproduction 实例列表
        configs: NEAT 配置列表
        species_sets: species 集合列表
        pop_sizes: 目标种群大小列表
        generation: 当前代数

    Returns:
        新种群字典列表
    """
    cdef int n = len(reproductions)
    cdef list results = []
    cdef int i

    for i in range(n):
        result = reproduce_fast(
            reproductions[i],
            configs[i],
            species_sets[i],
            pop_sizes[i],
            generation
        )
        results.append(result)

    return results


# ============================================================================
# 工具函数：批量创建基因组
# ============================================================================

cpdef dict create_new_genomes_fast(
    object genome_type,
    object genome_config,
    object genome_indexer,
    int num_genomes,
    dict ancestors
):
    """
    快速创建新基因组（用于初始化种群）

    Args:
        genome_type: 基因组类型
        genome_config: 基因组配置
        genome_indexer: ID 生成器
        num_genomes: 要创建的数量
        ancestors: 祖先记录字典

    Returns:
        新基因组字典
    """
    cdef dict new_genomes = {}
    cdef int i
    cdef int key
    cdef object g

    for i in range(num_genomes):
        key = next(genome_indexer)
        g = genome_type(key)
        g.configure_new(genome_config)
        new_genomes[key] = g
        ancestors[key] = ()

    return new_genomes

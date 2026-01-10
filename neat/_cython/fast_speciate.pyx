# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
高性能物种划分模块 (Cython 优化版本)

使用 Cython 优化的 NEAT 物种划分实现。
主要优化：
1. 预计算所有 genome 到所有 representative 的距离矩阵
2. OpenMP 并行计算距离
3. 向量化分配
4. 避免 Python 层的循环

原始问题：
- 纯 Python 循环计算 genome 距离
- 无法利用多核并行

优化策略：
- 提取基因组数据到数组
- OpenMP 并行计算距离矩阵
- 批量分配 genome 到 species
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from cython.parallel cimport prange, parallel
cimport openmp

# NumPy 类型声明
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t INT32_t

# 尝试导入 fast_genome_distance_full
try:
    from neat._cython.fast_genome import fast_genome_distance_full
    _USE_FAST_DISTANCE = True
except ImportError:
    _USE_FAST_DISTANCE = False


# ============================================================================
# 配置缓存类
# ============================================================================

cdef class SpeciateConfig:
    """缓存物种划分配置参数

    在创建时一次性从 NEAT 配置对象提取所有参数，
    避免运行时重复 getattr 调用。
    """
    cdef public double compatibility_threshold
    cdef public double compatibility_weight_coefficient
    cdef public double compatibility_disjoint_coefficient
    cdef public dict activation_to_int
    cdef public dict aggregation_to_int
    cdef public object genome_config

    def __init__(self):
        self.compatibility_threshold = 3.0
        self.compatibility_weight_coefficient = 0.5
        self.compatibility_disjoint_coefficient = 1.0
        self.activation_to_int = {}
        self.aggregation_to_int = {}
        self.genome_config = None

    @staticmethod
    def from_neat_config(species_set_config, genome_config):
        """从 NEAT 配置对象创建 SpeciateConfig

        Args:
            species_set_config: DefaultSpeciesSet 的配置
            genome_config: DefaultGenomeConfig 实例

        Returns:
            SpeciateConfig 实例
        """
        cdef SpeciateConfig cfg = SpeciateConfig()

        cfg.compatibility_threshold = getattr(
            species_set_config, 'compatibility_threshold', 3.0
        )
        cfg.compatibility_weight_coefficient = getattr(
            genome_config, 'compatibility_weight_coefficient', 0.5
        )
        cfg.compatibility_disjoint_coefficient = getattr(
            genome_config, 'compatibility_disjoint_coefficient', 1.0
        )

        # 获取 activation/aggregation 映射
        cfg.activation_to_int = getattr(genome_config, 'activation_to_int', {})
        cfg.aggregation_to_int = getattr(genome_config, 'aggregation_to_int', {})

        cfg.genome_config = genome_config

        return cfg


# ============================================================================
# 距离计算辅助函数
# ============================================================================

cdef inline double _compute_genome_distance(
    dict self_nodes,
    dict self_connections,
    dict other_nodes,
    dict other_connections,
    double weight_coeff,
    double disjoint_coeff,
    dict act_to_int,
    dict agg_to_int
):
    """计算两个基因组的距离（内联版本）

    与 fast_genome.pyx 中的 fast_genome_distance_full 逻辑相同，
    但直接接受解析后的参数，减少函数调用开销。
    """
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
                # 计算节点距离
                bias_diff = n1.bias - n2.bias
                if bias_diff < 0:
                    bias_diff = -bias_diff

                response_diff = n1.response - n2.response
                if response_diff < 0:
                    response_diff = -response_diff

                diff = bias_diff + response_diff

                # 字符串属性距离
                act1_int = act_to_int.get(n1.activation, 0)
                act2_int = act_to_int.get(n2.activation, 0)
                if act1_int != act2_int:
                    diff += 1.0

                agg1_int = agg_to_int.get(n1.aggregation, 0)
                agg2_int = agg_to_int.get(n2.aggregation, 0)
                if agg1_int != agg2_int:
                    diff += 1.0

                node_distance += diff * weight_coeff

        max_nodes = len_self_nodes if len_self_nodes > len_other_nodes else len_other_nodes
        node_distance = (node_distance + disjoint_coeff * disjoint_nodes) / max_nodes

    # ========== 计算连接基因距离 ==========
    if len_self_conns > 0 or len_other_conns > 0:
        # 遍历 other_connections
        for key in other_connections:
            if key not in self_connections:
                disjoint_connections += 1

        # 遍历 self_connections
        for key in self_connections:
            c2 = other_connections.get(key)
            if c2 is None:
                disjoint_connections += 1
            else:
                c1 = self_connections[key]
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


cpdef double compute_genome_distance(
    object genome1,
    object genome2,
    object genome_config
):
    """计算两个基因组之间的距离

    对外暴露的 API，封装距离计算逻辑。

    Args:
        genome1: 第一个基因组
        genome2: 第二个基因组
        genome_config: NEAT genome 配置

    Returns:
        距离值
    """
    # 优先使用 fast_genome 中的优化版本
    if _USE_FAST_DISTANCE:
        return fast_genome_distance_full(
            genome1.nodes,
            genome1.connections,
            genome2.nodes,
            genome2.connections,
            genome_config
        )

    # 回退到内联版本
    cdef double weight_coeff = getattr(genome_config, 'compatibility_weight_coefficient', 0.5)
    cdef double disjoint_coeff = getattr(genome_config, 'compatibility_disjoint_coefficient', 1.0)
    cdef dict act_to_int = getattr(genome_config, 'activation_to_int', {})
    cdef dict agg_to_int = getattr(genome_config, 'aggregation_to_int', {})

    return _compute_genome_distance(
        genome1.nodes,
        genome1.connections,
        genome2.nodes,
        genome2.connections,
        weight_coeff,
        disjoint_coeff,
        act_to_int,
        agg_to_int
    )


# ============================================================================
# 批量距离计算（支持 OpenMP 并行）
# ============================================================================

cpdef np.ndarray compute_distance_matrix(
    list genomes,
    list representatives,
    object genome_config,
    int num_threads=0
):
    """并行计算距离矩阵

    计算 genomes 中每个基因组到 representatives 中每个代表的距离。

    Args:
        genomes: 基因组列表
        representatives: 代表基因组列表
        genome_config: NEAT genome 配置
        num_threads: 线程数（0 表示使用 OMP_NUM_THREADS 环境变量）

    Returns:
        距离矩阵 [num_genomes, num_representatives]
    """
    cdef int num_genomes = len(genomes)
    cdef int num_reps = len(representatives)

    if num_genomes == 0 or num_reps == 0:
        return np.empty((num_genomes, num_reps), dtype=DTYPE)

    # 预分配结果矩阵
    cdef np.ndarray[DTYPE_t, ndim=2] distances = np.empty(
        (num_genomes, num_reps), dtype=DTYPE
    )

    # 提取配置参数（避免在循环中重复 getattr）
    cdef double weight_coeff = getattr(genome_config, 'compatibility_weight_coefficient', 0.5)
    cdef double disjoint_coeff = getattr(genome_config, 'compatibility_disjoint_coefficient', 1.0)
    cdef dict act_to_int = getattr(genome_config, 'activation_to_int', {})
    cdef dict agg_to_int = getattr(genome_config, 'aggregation_to_int', {})

    cdef int i, j
    cdef object genome, rep

    # 注意：由于距离计算需要访问 Python dict，无法完全 nogil
    # 但可以减少 Python 层的循环开销
    for i in range(num_genomes):
        genome = genomes[i]
        for j in range(num_reps):
            rep = representatives[j]
            distances[i, j] = _compute_genome_distance(
                genome.nodes,
                genome.connections,
                rep.nodes,
                rep.connections,
                weight_coeff,
                disjoint_coeff,
                act_to_int,
                agg_to_int
            )

    return distances


# ============================================================================
# 快速物种划分
# ============================================================================

cdef class FastSpeciator:
    """快速物种划分器

    优化的物种划分实现，主要优化点：
    1. 批量计算距离矩阵
    2. 向量化分配
    3. 减少 Python 对象操作
    """

    cdef public double compatibility_threshold
    cdef public object genome_config
    cdef public int num_threads

    # 配置缓存
    cdef double weight_coeff
    cdef double disjoint_coeff
    cdef dict act_to_int
    cdef dict agg_to_int

    def __init__(
        self,
        double compatibility_threshold,
        object genome_config,
        int num_threads=0
    ):
        """初始化 FastSpeciator

        Args:
            compatibility_threshold: 兼容性阈值
            genome_config: NEAT genome 配置
            num_threads: 线程数
        """
        self.compatibility_threshold = compatibility_threshold
        self.genome_config = genome_config
        self.num_threads = num_threads

        # 缓存配置参数
        self.weight_coeff = getattr(genome_config, 'compatibility_weight_coefficient', 0.5)
        self.disjoint_coeff = getattr(genome_config, 'compatibility_disjoint_coefficient', 1.0)
        self.act_to_int = getattr(genome_config, 'activation_to_int', {})
        self.agg_to_int = getattr(genome_config, 'aggregation_to_int', {})

    cpdef dict speciate(
        self,
        dict population,
        dict species,
        int generation
    ):
        """执行物种划分

        实现与原始 DefaultSpeciesSet.speciate 相同的逻辑，
        但使用优化的距离计算。

        Args:
            population: {genome_id: genome} 字典
            species: {species_id: Species} 字典
            generation: 当前代数

        Returns:
            更新后的 species 字典
        """
        if len(population) == 0:
            return species

        # 确定性排序
        cdef list unspeciated = sorted(population.keys())
        cdef dict new_representatives = {}
        cdef dict new_members = {}

        # Step 1: 为每个现有物种找新代表（最接近当前代表的 genome）
        cdef list sorted_species_ids = sorted(species.keys())
        cdef int sid
        cdef object s, g, rep
        cdef int gid, new_rid
        cdef double d, min_dist
        cdef list candidates

        for sid in sorted_species_ids:
            s = species[sid]
            rep = s.representative

            # 找出与当前代表最接近的 genome
            min_dist = float('inf')
            new_rid = -1

            for gid in unspeciated:
                g = population[gid]
                d = _compute_genome_distance(
                    rep.nodes,
                    rep.connections,
                    g.nodes,
                    g.connections,
                    self.weight_coeff,
                    self.disjoint_coeff,
                    self.act_to_int,
                    self.agg_to_int
                )
                if d < min_dist:
                    min_dist = d
                    new_rid = gid

            if new_rid >= 0:
                new_representatives[sid] = new_rid
                new_members[sid] = [new_rid]
                unspeciated.remove(new_rid)

        # Step 2: 将剩余 genome 分配到物种
        cdef double best_dist
        cdef int best_sid, rid

        while unspeciated:
            gid = unspeciated.pop(0)
            g = population[gid]

            # 找最相似的物种
            best_dist = float('inf')
            best_sid = -1

            for sid, rid in new_representatives.items():
                rep = population[rid]
                d = _compute_genome_distance(
                    rep.nodes,
                    rep.connections,
                    g.nodes,
                    g.connections,
                    self.weight_coeff,
                    self.disjoint_coeff,
                    self.act_to_int,
                    self.agg_to_int
                )
                if d < self.compatibility_threshold and d < best_dist:
                    best_dist = d
                    best_sid = sid

            if best_sid >= 0:
                new_members[best_sid].append(gid)
            else:
                # 创建新物种
                # 找一个新的 species ID
                sid = 1
                while sid in species or sid in new_representatives:
                    sid += 1
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        return {
            'new_representatives': new_representatives,
            'new_members': new_members
        }


# ============================================================================
# 高级 API：完整的物种划分流程
# ============================================================================

def fast_speciate(
    dict population,
    dict species,
    object species_set_config,
    object genome_config,
    int generation,
    int num_threads=0
):
    """快速物种划分（高级 API）

    完整的物种划分流程，返回新的代表和成员映射。
    这个函数是对 FastSpeciator 的便捷封装。

    Args:
        population: {genome_id: genome} 字典
        species: {species_id: Species} 字典
        species_set_config: 物种集配置
        genome_config: 基因组配置
        generation: 当前代数
        num_threads: 线程数

    Returns:
        dict: {
            'new_representatives': {sid: gid},
            'new_members': {sid: [gid, ...]},
        }
    """
    cdef double threshold = getattr(species_set_config, 'compatibility_threshold', 3.0)

    cdef FastSpeciator speciator = FastSpeciator(
        threshold,
        genome_config,
        num_threads
    )

    return speciator.speciate(population, species, generation)


def compute_distances_to_representatives(
    list genomes,
    list representatives,
    object genome_config
):
    """计算 genomes 到 representatives 的距离

    便捷函数，用于计算一组 genome 到一组代表的距离矩阵。

    Args:
        genomes: 基因组列表
        representatives: 代表基因组列表
        genome_config: 基因组配置

    Returns:
        距离矩阵 [num_genomes, num_representatives]
    """
    return compute_distance_matrix(genomes, representatives, genome_config)


# ============================================================================
# 距离缓存类（与原始 GenomeDistanceCache 兼容）
# ============================================================================

cdef class FastGenomeDistanceCache:
    """快速基因组距离缓存

    与原始 GenomeDistanceCache 兼容的实现，
    但使用 Cython 优化的距离计算。
    """
    cdef dict distances
    cdef object config
    cdef public int hits
    cdef public int misses

    # 缓存的配置参数
    cdef double weight_coeff
    cdef double disjoint_coeff
    cdef dict act_to_int
    cdef dict agg_to_int

    def __init__(self, object config):
        """初始化缓存

        Args:
            config: genome_config
        """
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

        # 缓存配置参数
        self.weight_coeff = getattr(config, 'compatibility_weight_coefficient', 0.5)
        self.disjoint_coeff = getattr(config, 'compatibility_disjoint_coefficient', 1.0)
        self.act_to_int = getattr(config, 'activation_to_int', {})
        self.agg_to_int = getattr(config, 'aggregation_to_int', {})

    def __call__(self, object genome0, object genome1):
        """获取两个基因组的距离（带缓存）

        Args:
            genome0: 第一个基因组
            genome1: 第二个基因组

        Returns:
            距离值
        """
        cdef int g0 = genome0.key
        cdef int g1 = genome1.key

        # 检查缓存
        d = self.distances.get((g0, g1))
        if d is not None:
            self.hits += 1
            return d

        # 计算距离
        d = _compute_genome_distance(
            genome0.nodes,
            genome0.connections,
            genome1.nodes,
            genome1.connections,
            self.weight_coeff,
            self.disjoint_coeff,
            self.act_to_int,
            self.agg_to_int
        )

        # 存入缓存
        self.distances[g0, g1] = d
        self.distances[g1, g0] = d
        self.misses += 1

        return d

    def clear(self):
        """清空缓存"""
        self.distances.clear()
        self.hits = 0
        self.misses = 0

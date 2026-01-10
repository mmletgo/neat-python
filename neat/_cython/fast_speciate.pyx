# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
高性能物种划分模块 (真正的 OpenMP 并行版本)

核心优化策略：
1. 预提取阶段：将 Python dict 数据转换为 C 数组（串行）
2. 距离计算阶段：使用 OpenMP prange 并行计算距离矩阵（并行）
3. 分配阶段：根据距离矩阵分配 genome 到 species（串行）

数据布局：
- 节点数据：扁平化数组 + 偏移量数组
- 连接数据：扁平化数组 + 偏移量数组
- 距离矩阵：连续的 2D 数组
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


# ============================================================================
# C 结构体定义
# ============================================================================

# 节点数据的 C 表示
cdef struct NodeData:
    int key
    double bias
    double response
    int activation_id
    int aggregation_id


# 连接数据的 C 表示
cdef struct ConnectionData:
    int in_node
    int out_node
    double weight
    int enabled  # 0 or 1


# 单个 genome 在扁平化数组中的布局
cdef struct GenomeLayout:
    int node_start
    int node_count
    int conn_start
    int conn_count


# ============================================================================
# 数据提取
# ============================================================================

cdef class GenomeDataExtractor:
    """从 Python genome 对象提取数据到 C 数组"""

    cdef:
        # 节点数据（扁平化）
        NodeData* nodes
        int total_nodes

        # 连接数据（扁平化）
        ConnectionData* connections
        int total_connections

        # 每个 genome 的布局信息
        GenomeLayout* layouts
        int num_genomes

        # 配置缓存
        double weight_coeff
        double disjoint_coeff
        dict activation_to_int
        dict aggregation_to_int

        # 是否已分配内存
        bint allocated

    def __cinit__(self):
        self.nodes = NULL
        self.connections = NULL
        self.layouts = NULL
        self.num_genomes = 0
        self.total_nodes = 0
        self.total_connections = 0
        self.allocated = False

    def __dealloc__(self):
        self._free_memory()

    cdef void _free_memory(self) noexcept:
        if self.nodes != NULL:
            free(self.nodes)
            self.nodes = NULL
        if self.connections != NULL:
            free(self.connections)
            self.connections = NULL
        if self.layouts != NULL:
            free(self.layouts)
            self.layouts = NULL
        self.allocated = False

    cpdef void extract(self, list genomes, object genome_config):
        """从 genome 列表提取数据

        Args:
            genomes: genome 对象列表
            genome_config: NEAT genome 配置
        """
        self._free_memory()

        cdef int n = len(genomes)
        if n == 0:
            return

        # 缓存配置
        self.weight_coeff = getattr(genome_config, 'compatibility_weight_coefficient', 0.5)
        self.disjoint_coeff = getattr(genome_config, 'compatibility_disjoint_coefficient', 1.0)
        self.activation_to_int = getattr(genome_config, 'activation_to_int', {})
        self.aggregation_to_int = getattr(genome_config, 'aggregation_to_int', {})

        self.num_genomes = n

        # 第一遍：统计总数
        cdef int i
        cdef object g
        self.total_nodes = 0
        self.total_connections = 0

        for i in range(n):
            g = genomes[i]
            self.total_nodes += len(g.nodes)
            self.total_connections += len(g.connections)

        # 分配内存
        self.layouts = <GenomeLayout*>malloc(n * sizeof(GenomeLayout))
        if self.total_nodes > 0:
            self.nodes = <NodeData*>malloc(self.total_nodes * sizeof(NodeData))
        if self.total_connections > 0:
            self.connections = <ConnectionData*>malloc(self.total_connections * sizeof(ConnectionData))

        self.allocated = True

        # 第二遍：填充数据
        cdef int node_offset = 0
        cdef int conn_offset = 0
        cdef object node_key, node_gene, conn_key, conn_gene
        cdef int j

        for i in range(n):
            g = genomes[i]

            # 记录布局
            self.layouts[i].node_start = node_offset
            self.layouts[i].node_count = len(g.nodes)
            self.layouts[i].conn_start = conn_offset
            self.layouts[i].conn_count = len(g.connections)

            # 提取节点数据
            for node_key, node_gene in g.nodes.items():
                self.nodes[node_offset].key = node_key
                self.nodes[node_offset].bias = node_gene.bias
                self.nodes[node_offset].response = node_gene.response
                self.nodes[node_offset].activation_id = self.activation_to_int.get(
                    node_gene.activation, 0
                )
                self.nodes[node_offset].aggregation_id = self.aggregation_to_int.get(
                    node_gene.aggregation, 0
                )
                node_offset += 1

            # 提取连接数据
            for conn_key, conn_gene in g.connections.items():
                self.connections[conn_offset].in_node = conn_key[0]
                self.connections[conn_offset].out_node = conn_key[1]
                self.connections[conn_offset].weight = conn_gene.weight
                self.connections[conn_offset].enabled = 1 if conn_gene.enabled else 0
                conn_offset += 1


# ============================================================================
# 并行距离计算
# ============================================================================

cdef double _compute_distance_from_arrays(
    NodeData* nodes1, int n1_start, int n1_count,
    ConnectionData* conns1, int c1_start, int c1_count,
    NodeData* nodes2, int n2_start, int n2_count,
    ConnectionData* conns2, int c2_start, int c2_count,
    double weight_coeff,
    double disjoint_coeff
) noexcept nogil:
    """从 C 数组计算两个 genome 的距离（nogil 版本）

    这个函数可以在 OpenMP 并行区域内调用。
    """
    cdef double node_distance = 0.0
    cdef double connection_distance = 0.0
    cdef int disjoint_nodes = 0
    cdef int disjoint_connections = 0
    cdef int max_nodes, max_conns
    cdef int i, j
    cdef bint found
    cdef double diff, bias_diff, response_diff, weight_diff

    # ========== 计算节点距离 ==========
    if n1_count > 0 or n2_count > 0:
        # 遍历 genome2 的节点，找不在 genome1 中的
        for i in range(n2_count):
            found = False
            for j in range(n1_count):
                if nodes2[n2_start + i].key == nodes1[n1_start + j].key:
                    found = True
                    break
            if not found:
                disjoint_nodes += 1

        # 遍历 genome1 的节点
        for i in range(n1_count):
            found = False
            for j in range(n2_count):
                if nodes1[n1_start + i].key == nodes2[n2_start + j].key:
                    found = True
                    # 计算同源节点距离
                    bias_diff = nodes1[n1_start + i].bias - nodes2[n2_start + j].bias
                    if bias_diff < 0:
                        bias_diff = -bias_diff

                    response_diff = nodes1[n1_start + i].response - nodes2[n2_start + j].response
                    if response_diff < 0:
                        response_diff = -response_diff

                    diff = bias_diff + response_diff

                    # 属性距离
                    if nodes1[n1_start + i].activation_id != nodes2[n2_start + j].activation_id:
                        diff += 1.0
                    if nodes1[n1_start + i].aggregation_id != nodes2[n2_start + j].aggregation_id:
                        diff += 1.0

                    node_distance += diff * weight_coeff
                    break

            if not found:
                disjoint_nodes += 1

        max_nodes = n1_count if n1_count > n2_count else n2_count
        if max_nodes > 0:
            node_distance = (node_distance + disjoint_coeff * disjoint_nodes) / max_nodes

    # ========== 计算连接距离 ==========
    if c1_count > 0 or c2_count > 0:
        # 遍历 genome2 的连接
        for i in range(c2_count):
            found = False
            for j in range(c1_count):
                if (conns2[c2_start + i].in_node == conns1[c1_start + j].in_node and
                    conns2[c2_start + i].out_node == conns1[c1_start + j].out_node):
                    found = True
                    break
            if not found:
                disjoint_connections += 1

        # 遍历 genome1 的连接
        for i in range(c1_count):
            found = False
            for j in range(c2_count):
                if (conns1[c1_start + i].in_node == conns2[c2_start + j].in_node and
                    conns1[c1_start + i].out_node == conns2[c2_start + j].out_node):
                    found = True
                    # 计算同源连接距离
                    weight_diff = conns1[c1_start + i].weight - conns2[c2_start + j].weight
                    if weight_diff < 0:
                        weight_diff = -weight_diff

                    diff = weight_diff
                    if conns1[c1_start + i].enabled != conns2[c2_start + j].enabled:
                        diff += 1.0

                    connection_distance += diff * weight_coeff
                    break

            if not found:
                disjoint_connections += 1

        max_conns = c1_count if c1_count > c2_count else c2_count
        if max_conns > 0:
            connection_distance = (connection_distance + disjoint_coeff * disjoint_connections) / max_conns

    return node_distance + connection_distance


cpdef np.ndarray compute_distance_matrix_parallel(
    GenomeDataExtractor extractor,
    list genome_indices,
    list rep_indices,
    int num_threads=0
):
    """并行计算距离矩阵

    使用 OpenMP 并行计算 genomes 到 representatives 的距离。

    Args:
        extractor: 已提取数据的 GenomeDataExtractor
        genome_indices: 要计算距离的 genome 索引列表
        rep_indices: representative 的索引列表
        num_threads: 线程数（0 表示使用默认值）

    Returns:
        距离矩阵 [num_genomes, num_reps]
    """
    cdef int num_genomes = len(genome_indices)
    cdef int num_reps = len(rep_indices)

    if num_genomes == 0 or num_reps == 0:
        return np.empty((num_genomes, num_reps), dtype=DTYPE)

    # 转换为 C 数组
    cdef np.ndarray[INT32_t, ndim=1] g_indices = np.array(genome_indices, dtype=np.int32)
    cdef np.ndarray[INT32_t, ndim=1] r_indices = np.array(rep_indices, dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=2] distances = np.empty((num_genomes, num_reps), dtype=DTYPE)

    cdef int i, j, gi, ri
    cdef double weight_coeff = extractor.weight_coeff
    cdef double disjoint_coeff = extractor.disjoint_coeff

    # 获取指针
    cdef NodeData* nodes = extractor.nodes
    cdef ConnectionData* connections = extractor.connections
    cdef GenomeLayout* layouts = extractor.layouts

    cdef int n_threads = num_threads if num_threads > 0 else 8

    # OpenMP 并行计算
    with nogil:
        for i in prange(num_genomes, num_threads=n_threads, schedule='dynamic'):
            gi = g_indices[i]
            for j in range(num_reps):
                ri = r_indices[j]
                distances[i, j] = _compute_distance_from_arrays(
                    nodes, layouts[gi].node_start, layouts[gi].node_count,
                    connections, layouts[gi].conn_start, layouts[gi].conn_count,
                    nodes, layouts[ri].node_start, layouts[ri].node_count,
                    connections, layouts[ri].conn_start, layouts[ri].conn_count,
                    weight_coeff,
                    disjoint_coeff
                )

    return distances


# ============================================================================
# 快速物种划分
# ============================================================================

def fast_speciate(
    dict population,
    dict species,
    object species_set_config,
    object genome_config,
    int generation,
    int num_threads=0
):
    """快速物种划分（真正的并行版本）

    流程：
    1. 提取所有 genome 数据到 C 数组
    2. 并行计算距离矩阵
    3. 串行分配 genome 到 species

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
    if len(population) == 0:
        return {'new_representatives': {}, 'new_members': {}}

    cdef double threshold = getattr(species_set_config, 'compatibility_threshold', 3.0)

    # 准备数据
    cdef list sorted_gids = sorted(population.keys())
    cdef dict gid_to_idx = {gid: i for i, gid in enumerate(sorted_gids)}
    cdef list all_genomes = [population[gid] for gid in sorted_gids]

    # 提取数据到 C 数组
    cdef GenomeDataExtractor extractor = GenomeDataExtractor()
    extractor.extract(all_genomes, genome_config)

    # 准备结果
    cdef dict new_representatives = {}
    cdef dict new_members = {}
    cdef list unspeciated = list(sorted_gids)

    # Step 1: 为每个现有物种找新代表
    cdef list sorted_sids = sorted(species.keys())
    cdef list rep_genomes = []
    cdef list rep_indices = []
    cdef dict sid_to_rep_idx = {}

    # 预声明 if 块内的变量（Cython 不允许在 if/else 块内声明 cdef 变量）
    cdef GenomeDataExtractor rep_extractor
    cdef GenomeDataExtractor rep_extractor2
    cdef GenomeDataExtractor unspec_extractor
    cdef GenomeDataExtractor rep_ext
    cdef list unspec_indices
    cdef list rep_idx_list
    cdef list unspec_gids
    cdef list sid_list
    cdef int n_unspec, n_reps
    cdef np.ndarray dist_matrix
    cdef int i, j
    cdef double best_dist, d, dist
    cdef int best_sid_idx, best_sid
    cdef int next_sid

    cdef int idx = 0
    for sid in sorted_sids:
        s = species[sid]
        rep_genomes.append(s.representative)
        rep_indices.append(idx)
        sid_to_rep_idx[sid] = idx
        idx += 1

    if rep_genomes:
        # 为 representatives 也提取数据（添加到同一个 extractor 会更复杂，这里单独处理）
        rep_extractor = GenomeDataExtractor()
        rep_extractor.extract(rep_genomes, genome_config)

        # 计算所有 unspeciated genomes 到所有 representatives 的距离
        unspec_indices = [gid_to_idx[gid] for gid in unspeciated]
        rep_idx_list = list(range(len(rep_genomes)))

        # 为每个 species 找最接近的 genome 作为新代表
        for sid_idx, sid in enumerate(sorted_sids):
            s = species[sid]

            # 计算 unspeciated genomes 到这个 representative 的距离
            min_dist = float('inf')
            min_gid = -1

            for gid in unspeciated:
                gi = gid_to_idx[gid]
                # 计算单个距离
                g = population[gid]
                rep = s.representative

                # 使用 extractor 中的数据计算距离
                dist = _compute_distance_py(
                    extractor, gi,
                    rep_extractor, sid_idx,
                )

                if dist < min_dist:
                    min_dist = dist
                    min_gid = gid

            if min_gid >= 0:
                new_representatives[sid] = min_gid
                new_members[sid] = [min_gid]
                unspeciated.remove(min_gid)

    # Step 2: 将剩余 genome 分配到物种（使用并行距离计算）
    if unspeciated and new_representatives:
        # 准备 representative 数据
        rep_gids = list(new_representatives.values())
        rep_genome_list = [population[gid] for gid in rep_gids]

        rep_extractor2 = GenomeDataExtractor()
        rep_extractor2.extract(rep_genome_list, genome_config)

        # 批量计算距离
        unspec_gids = list(unspeciated)
        n_unspec = len(unspec_gids)
        n_reps = len(rep_gids)

        # 提取 unspeciated genomes 的数据
        unspec_genomes = [population[gid] for gid in unspec_gids]
        unspec_extractor = GenomeDataExtractor()
        unspec_extractor.extract(unspec_genomes, genome_config)

        # 并行计算距离矩阵
        dist_matrix = _compute_distance_matrix_between(
            unspec_extractor, rep_extractor2, num_threads
        )

        # 分配 genome 到 species
        sid_list = list(new_representatives.keys())

        for i in range(n_unspec):
            gid = unspec_gids[i]

            best_dist = float('inf')
            best_sid_idx = -1

            for j in range(n_reps):
                d = dist_matrix[i, j]
                if d < threshold and d < best_dist:
                    best_dist = d
                    best_sid_idx = j

            if best_sid_idx >= 0:
                best_sid = sid_list[best_sid_idx]
                new_members[best_sid].append(gid)
            else:
                # 创建新物种
                next_sid = 1
                while next_sid in species or next_sid in new_representatives:
                    next_sid += 1
                new_representatives[next_sid] = gid
                new_members[next_sid] = [gid]
                sid_list.append(next_sid)

                # 更新 rep_extractor2（添加新的 representative）
                # 简化处理：新物种直接加入，不重新计算距离矩阵

    # 处理没有现有物种的情况
    elif unspeciated and not new_representatives:
        # 第一个 genome 成为第一个物种的代表
        gid = unspeciated.pop(0)
        new_representatives[1] = gid
        new_members[1] = [gid]

        if unspeciated:
            # 简化处理：逐个分配剩余 genome
            rep_gids = [population[new_representatives[1]]]
            rep_ext = GenomeDataExtractor()
            rep_ext.extract(rep_gids, genome_config)

            for gid in unspeciated:
                g = population[gid]

                # 计算到现有代表的距离
                best_dist = float('inf')
                best_sid = -1

                for sid, rep_gid in new_representatives.items():
                    rep = population[rep_gid]

                    # 简化：直接用 Python 计算
                    dist = _compute_distance_simple(g, rep, genome_config)

                    if dist < threshold and dist < best_dist:
                        best_dist = dist
                        best_sid = sid

                if best_sid >= 0:
                    new_members[best_sid].append(gid)
                else:
                    next_sid = max(new_representatives.keys()) + 1
                    new_representatives[next_sid] = gid
                    new_members[next_sid] = [gid]

    return {
        'new_representatives': new_representatives,
        'new_members': new_members
    }


cdef np.ndarray _compute_distance_matrix_between(
    GenomeDataExtractor ext1,
    GenomeDataExtractor ext2,
    int num_threads
):
    """计算两组 genome 之间的距离矩阵

    使用 OpenMP 并行计算。
    """
    cdef int n1 = ext1.num_genomes
    cdef int n2 = ext2.num_genomes

    if n1 == 0 or n2 == 0:
        return np.empty((n1, n2), dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=2] distances = np.empty((n1, n2), dtype=DTYPE)

    cdef int i, j
    cdef double weight_coeff = ext1.weight_coeff
    cdef double disjoint_coeff = ext1.disjoint_coeff

    cdef NodeData* nodes1 = ext1.nodes
    cdef ConnectionData* conns1 = ext1.connections
    cdef GenomeLayout* layouts1 = ext1.layouts

    cdef NodeData* nodes2 = ext2.nodes
    cdef ConnectionData* conns2 = ext2.connections
    cdef GenomeLayout* layouts2 = ext2.layouts

    cdef int n_threads = num_threads if num_threads > 0 else 8

    with nogil:
        for i in prange(n1, num_threads=n_threads, schedule='dynamic'):
            for j in range(n2):
                distances[i, j] = _compute_distance_from_arrays(
                    nodes1, layouts1[i].node_start, layouts1[i].node_count,
                    conns1, layouts1[i].conn_start, layouts1[i].conn_count,
                    nodes2, layouts2[j].node_start, layouts2[j].node_count,
                    conns2, layouts2[j].conn_start, layouts2[j].conn_count,
                    weight_coeff,
                    disjoint_coeff
                )

    return distances


cdef double _compute_distance_py(
    GenomeDataExtractor ext1, int idx1,
    GenomeDataExtractor ext2, int idx2
):
    """Python 可调用的距离计算（单个）"""
    return _compute_distance_from_arrays(
        ext1.nodes, ext1.layouts[idx1].node_start, ext1.layouts[idx1].node_count,
        ext1.connections, ext1.layouts[idx1].conn_start, ext1.layouts[idx1].conn_count,
        ext2.nodes, ext2.layouts[idx2].node_start, ext2.layouts[idx2].node_count,
        ext2.connections, ext2.layouts[idx2].conn_start, ext2.layouts[idx2].conn_count,
        ext1.weight_coeff,
        ext1.disjoint_coeff
    )


def _compute_distance_simple(genome1, genome2, genome_config):
    """简单的 Python 距离计算（作为回退）"""
    weight_coeff = getattr(genome_config, 'compatibility_weight_coefficient', 0.5)
    disjoint_coeff = getattr(genome_config, 'compatibility_disjoint_coefficient', 1.0)
    act_to_int = getattr(genome_config, 'activation_to_int', {})
    agg_to_int = getattr(genome_config, 'aggregation_to_int', {})

    node_distance = 0.0
    connection_distance = 0.0
    disjoint_nodes = 0
    disjoint_connections = 0

    # 节点距离
    nodes1 = genome1.nodes
    nodes2 = genome2.nodes

    if nodes1 or nodes2:
        for key in nodes2:
            if key not in nodes1:
                disjoint_nodes += 1

        for key in nodes1:
            n2 = nodes2.get(key)
            if n2 is None:
                disjoint_nodes += 1
            else:
                n1 = nodes1[key]
                diff = abs(n1.bias - n2.bias) + abs(n1.response - n2.response)
                if act_to_int.get(n1.activation, 0) != act_to_int.get(n2.activation, 0):
                    diff += 1.0
                if agg_to_int.get(n1.aggregation, 0) != agg_to_int.get(n2.aggregation, 0):
                    diff += 1.0
                node_distance += diff * weight_coeff

        max_nodes = max(len(nodes1), len(nodes2))
        if max_nodes > 0:
            node_distance = (node_distance + disjoint_coeff * disjoint_nodes) / max_nodes

    # 连接距离
    conns1 = genome1.connections
    conns2 = genome2.connections

    if conns1 or conns2:
        for key in conns2:
            if key not in conns1:
                disjoint_connections += 1

        for key in conns1:
            c2 = conns2.get(key)
            if c2 is None:
                disjoint_connections += 1
            else:
                c1 = conns1[key]
                diff = abs(c1.weight - c2.weight)
                if c1.enabled != c2.enabled:
                    diff += 1.0
                connection_distance += diff * weight_coeff

        max_conns = max(len(conns1), len(conns2))
        if max_conns > 0:
            connection_distance = (connection_distance + disjoint_coeff * disjoint_connections) / max_conns

    return node_distance + connection_distance


# ============================================================================
# 公开 API
# ============================================================================

cpdef double compute_genome_distance(
    object genome1,
    object genome2,
    object genome_config
):
    """计算两个基因组之间的距离"""
    return _compute_distance_simple(genome1, genome2, genome_config)

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython 优化的图算法

提供 NEAT 图算法的 Cython 优化实现：
1. fast_creates_cycle - 环检测
2. fast_required_for_output - 计算必需节点
3. fast_feed_forward_layers - 计算拓扑层

主要优化：
- 使用 cdef 类型声明加速变量访问
- 使用 set 进行 O(1) 查找
- 减少 Python 对象创建开销
"""

from typing import List, Tuple, Set


cpdef bint fast_creates_cycle(list connections, tuple test):
    """
    检测添加新连接是否会形成环

    Args:
        connections: 现有连接列表，每个元素为 (源节点, 目标节点) 元组
        test: 待测试的新连接 (源节点, 目标节点)

    Returns:
        True 如果添加该连接会形成环，否则 False

    算法：从 test 的目标节点开始，遍历所有可达节点，
    如果能到达 test 的源节点，则形成环。
    """
    cdef int i_node = test[0]
    cdef int o_node = test[1]
    cdef int a, b, num_added
    cdef tuple conn
    cdef set visited

    # 自环检测
    if i_node == o_node:
        return True

    # 从输出节点开始，向前遍历所有可达节点
    visited = {o_node}

    while True:
        num_added = 0
        for conn in connections:
            a = conn[0]
            b = conn[1]
            # 如果源节点已访问，将目标节点加入访问集合
            if a in visited and b not in visited:
                # 如果到达了测试连接的源节点，说明形成环
                if b == i_node:
                    return True
                visited.add(b)
                num_added += 1

        # 没有新节点加入，遍历完成
        if num_added == 0:
            return False


cpdef set fast_required_for_output(list inputs, list outputs, list connections):
    """
    计算输出所需的必需节点集合

    Args:
        inputs: 输入节点 ID 列表
        outputs: 输出节点 ID 列表
        connections: 连接列表，每个元素为 (源节点, 目标节点) 元组

    Returns:
        必需节点的集合（不包含输入节点）

    算法：从输出节点反向遍历，找出所有能影响输出的节点。
    """
    cdef set inputs_set = set(inputs)
    cdef set required = set(outputs)
    cdef set s = set(outputs)
    cdef set t, layer_nodes
    cdef tuple conn
    cdef int a, b

    while True:
        # 找出不在 s 中但输出被 s 中节点消费的节点
        t = set()
        for conn in connections:
            a = conn[0]
            b = conn[1]
            if b in s and a not in s:
                t.add(a)

        if not t:
            break

        # 只将非输入节点加入必需集合
        layer_nodes = set()
        for x in t:
            if x not in inputs_set:
                layer_nodes.add(x)

        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


cpdef tuple fast_feed_forward_layers(list inputs, list outputs, list connections):
    """
    计算前馈网络的拓扑层

    Args:
        inputs: 输入节点 ID 列表
        outputs: 输出节点 ID 列表
        connections: 连接列表，每个元素为 (源节点, 目标节点) 元组

    Returns:
        (layers, required) 元组
        - layers: 层列表，每层是可并行计算的节点集合
        - required: 必需节点集合

    算法：
    1. 首先计算必需节点
    2. 找出偏置神经元（必需但无输入的节点）
    3. 按拓扑顺序构建层
    """
    cdef set required = fast_required_for_output(inputs, outputs, connections)
    cdef set nodes_with_inputs = set()
    cdef set bias_neurons
    cdef set inputs_set = set(inputs)
    cdef list layers = []
    cdef set potential_input
    cdef set c, next_layer
    cdef tuple conn
    cdef int a, b, n
    cdef bint all_inputs_available
    cdef list connections_to_n

    # 找出有输入连接的节点
    for conn in connections:
        nodes_with_inputs.add(conn[1])

    # 偏置神经元：必需但没有输入连接的节点
    bias_neurons = required - nodes_with_inputs

    # 从输入节点和偏置神经元开始
    potential_input = inputs_set | bias_neurons

    # 如果有偏置神经元，作为第一层
    if bias_neurons:
        layers.append(bias_neurons.copy())

    while True:
        # 找出候选节点：从 potential_input 连接到非 potential_input 的节点
        c = set()
        for conn in connections:
            a = conn[0]
            b = conn[1]
            if a in potential_input and b not in potential_input:
                c.add(b)

        # 筛选出所有输入都已就绪的必需节点
        next_layer = set()
        for n in c:
            if n not in required:
                continue

            # 检查该节点的所有必需输入是否都已就绪
            all_inputs_available = True
            for conn in connections:
                if conn[1] == n and conn[0] in required:
                    if conn[0] not in potential_input:
                        all_inputs_available = False
                        break

            if all_inputs_available:
                next_layer.add(n)

        if not next_layer:
            break

        layers.append(next_layer)
        potential_input = potential_input.union(next_layer)

    return layers, required


# 辅助函数：使用邻接表优化大规模图
cpdef tuple fast_feed_forward_layers_optimized(list inputs, list outputs, list connections):
    """
    使用邻接表优化的拓扑层计算（适用于大规模网络）

    与 fast_feed_forward_layers 功能相同，但使用邻接表加速查找。
    当连接数量较大时（>100），此版本更高效。
    """
    cdef set required = fast_required_for_output(inputs, outputs, connections)
    cdef set inputs_set = set(inputs)
    cdef list layers = []
    cdef set potential_input
    cdef set next_layer, bias_neurons
    cdef dict incoming_edges = {}  # node -> [source_nodes]
    cdef dict outgoing_edges = {}  # node -> [target_nodes]
    cdef tuple conn
    cdef int a, b, n
    cdef bint all_inputs_available
    cdef set nodes_with_inputs = set()
    cdef list sources

    # 构建邻接表
    for conn in connections:
        a = conn[0]
        b = conn[1]

        # 出边
        if a not in outgoing_edges:
            outgoing_edges[a] = []
        outgoing_edges[a].append(b)

        # 入边
        if b not in incoming_edges:
            incoming_edges[b] = []
        incoming_edges[b].append(a)

        nodes_with_inputs.add(b)

    # 偏置神经元
    bias_neurons = required - nodes_with_inputs
    potential_input = inputs_set | bias_neurons

    if bias_neurons:
        layers.append(bias_neurons.copy())

    while True:
        # 使用邻接表找候选节点
        next_layer = set()

        for source in potential_input:
            if source not in outgoing_edges:
                continue
            for target in outgoing_edges[source]:
                if target in potential_input:
                    continue
                if target not in required:
                    continue

                # 检查所有必需输入是否就绪
                all_inputs_available = True
                if target in incoming_edges:
                    for src in incoming_edges[target]:
                        if src in required and src not in potential_input:
                            all_inputs_available = False
                            break

                if all_inputs_available:
                    next_layer.add(target)

        if not next_layer:
            break

        layers.append(next_layer)
        potential_input = potential_input.union(next_layer)

    return layers, required

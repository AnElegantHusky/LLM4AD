from __future__ import annotations

import math
from threading import Lock
from typing import List

import deprecated
import numpy as np
from collections import defaultdict

import overrides

from .base.code import Function
from .base.print_utils import print_error


class TreeNode():
    def __init__(self, sample_order, func, level):
        self._ID = func.ID
        self._fitness_vector = func.fitness_vector
        self._sample_order = sample_order
        self._func: Function = func

        self._prompt_type = func.prompt_type
        self._parents = func.parents
        self._children = []

        self._level = level

        # if self._parents in [None, [], [None]]:
        #     self._level = 0
        # else:
        #     self._level = sum([p.level for p in self._parents]) // len(
        #         self._parents)  # TODO: if there are multiple parents, what's the level?

        # self._search_history = defaultdict(int)

    def is_leaf(self):
        return len(self._children) == 0

    def is_root(self):
        return self._parents is None

    def __repr__(self):
        return f"TreeNode(ID={self._ID}, level={self._level}, children={len(self._children)})"


class TreePopulation:
    """
    算子树管理器。

    这个类维护树的完整结构和所有访问索引。
    所有对树的修改（如添加节点）都应通过这个类来完成，
    以确保所有索引始终保持同步。
    """

    def __init__(self):
        self._ID_set = {None}
        self._lock = Lock()
        self._roots = []

        # 1. 根据 算子ID 访问
        self._nodes_by_id: dict[str, TreeNode] = {}

        # 2. 根据 算子Index 访问
        self._nodes_by_index: dict[int, TreeNode] = {}

        # 3. 根据 树的层次 访问
        self._nodes_by_level = defaultdict(list)

        # 4. 访问 所有叶子节点
        self._leaf_nodes: set[TreeNode] = set()

        self._sample_count = 0

        self._tabu_dict = {
            'E1': {},
            'E2': {},
            'M1': {},
            'M2': {}
        }

        # --- 新增: 记录正在处理中的父代组合 (防止并发重复采样) ---
        # 存储格式: tuple(sorted([ID1, ID2, ...]))
        self._processing_parents: set[tuple] = set()

    def __len__(self):
        return len(self._nodes_by_index)

    def __getitem__(self, item) -> TreeNode:
        return self._nodes_by_index[item]

    @property
    def population(self):
        return [node._func for node in self._nodes_by_index.values()]

    def if_ID_duplicate(self, ID):
        return ID in self._nodes_by_id

    def register_function(self, func: Function):
        # Note: unlike EoH, we only accept valid functions
        if func.score is None:
            return

        # if the ID is duplicated, discard      # Step 1.4: 用ID去重；否则添加ID
        if self.if_ID_duplicate(func.ID):  # Note: this line should have no effect
            return

        try:
            with self._lock:
                self.add_node(func)
        except Exception as e:
            print_error(f'TreePopulation.register_function: {type(e).__name__}: {e}')
            return
        finally:
            if self._lock.locked():
                self._lock.release()

    def add_node(self, func: Function) -> TreeNode:
        """
        向树中添加一个新节点。
        这是唯一应该用于添加节点的方法，以确保索引一致。
        """
        sample_order = self._sample_count
        self._sample_count += 1

        if func.parents in [None, [], [None]]:
            parents = None
            level = 0
        else:
            parents = [self.get_node_by_id(pid) for pid in func.parents]
            level = sum([p._level for p in parents]) // len(parents)

        new_node = TreeNode(
            sample_order=sample_order,
            func=func,
            level=level
        )

        if parents in [None, [], [None]]:
            self._roots.append(new_node)
        else:
            for parent in parents:
                parent._children.append(new_node)
                if parent in self._leaf_nodes:
                    self._leaf_nodes.remove(parent)

        ID = func.ID

        # --- 4. 维护所有索引 (关键步骤!) ---

        # 维护 ID 索引
        self._nodes_by_id[ID] = new_node

        # 维护 Index 索引
        self._nodes_by_index[sample_order] = new_node

        # 维护 Level 索引
        self._nodes_by_level[level].append(new_node)

        # 维护 Leaf 索引
        # 新节点总是叶子节点
        self._leaf_nodes.add(new_node)

        return new_node

    # --- 灵活的访问方法 (全部是 O(1) 复杂度) ---

    def get_node_by_id(self, operator_id):
        """根据 算子ID 获取节点"""
        return self._nodes_by_id.get(operator_id, None)

    def get_node_by_index(self, index):
        """根据 算子Index 获取节点"""
        return self._nodes_by_index.get(index, None)

    def get_nodes_by_level(self, level):
        """获取某一层的所有节点"""
        # 返回列表的副本，防止外部修改
        return list(self._nodes_by_level.get(level, []))

    def get_leaf_nodes(self):
        """获取所有叶子节点"""
        # 返回集合的副本
        return set(self._leaf_nodes)

    def select(self, n, prompt_type) -> list[Function]:
        """
        选择 n 个父代节点。
        包含重试机制，防止多线程同时选中完全相同的父代组合。
        """
        max_retries = 10  # 最大重试次数，防止死循环
        retry_count = 0

        with self._lock:
            while retry_count < max_retries:
                try:
                    selected_nodes = self._select_logic(n, prompt_type)

                    # 如果没有选出节点，直接返回空
                    if not selected_nodes:
                        return []

                    # 生成唯一Key: 对ID进行排序，保证 (A, B) 和 (B, A) 视为同一个组合
                    # 只有当 n > 0 时才检查
                    key = tuple(sorted([node._ID for node in selected_nodes]))

                    # 检查是否正在被处理
                    if key in self._processing_parents:
                        # 冲突：当前组合正在被另一个线程使用，重试
                        retry_count += 1
                        continue
                    else:
                        # 未冲突：锁定该组合
                        self._processing_parents.add(key)
                        return [node._func for node in selected_nodes]

                except Exception as e:
                    print_error(f'TreePopulation.select: {type(e).__name__}: {e}')
                    return []

            # 如果重试次数耗尽，仍然只采样到了被锁定的节点
            # 策略：允许重复（兜底），避免程序挂起或返回空
            # 这种情况通常发生在种群很小且高性能节点很少时
            return [node._func for node in selected_nodes]

    def _select_logic(self, n, prompt_type) -> list[TreeNode]:
        """
        内部选择逻辑（不包含锁检查），原 select 的核心逻辑。
        """
        tabu_dict = self._tabu_dict[prompt_type]

        # --- Step 1: 准备三个互斥的候选池 ---
        # 获取所有非Tabu的节点
        non_tabu_nodes = [
            node for node in self._nodes_by_index.values()
            if node._ID not in tabu_dict
        ]

        # Tier 1: Leaf & Non-Tabu (最高优先级)
        # 假设 leaf_nodes 里的节点一定也在 nodes_by_index 里
        tier1_leaves = [node for node in self._leaf_nodes if node._ID not in tabu_dict]
        tier1_ids = {node._ID for node in tier1_leaves}

        # Tier 2: Non-Leaf & Non-Tabu (次优先级)
        tier2_internal = [node for node in non_tabu_nodes if node._ID not in tier1_ids]

        # Tier 3: Tabu Nodes (最低优先级，兜底)
        tier3_tabu = [
            node for node in self._nodes_by_index.values()
            if node._ID in tabu_dict
        ]

        # 放入列表，按优先级排序
        candidate_tiers = [tier1_leaves, tier2_internal, tier3_tabu]

        selected_nodes = []

        # --- Step 2: 级联选择逻辑 ---
        for pool in candidate_tiers:
            needed = n - len(selected_nodes)
            if needed <= 0:
                break  # 已经选够了

            if not pool:
                continue  # 当前层级为空，跳过

            if len(pool) <= needed:
                # 情况 A: 当前层级不够或刚好填满需求 -> 全选
                selected_nodes.extend(pool)
            else:
                # 情况 B: 当前层级充裕 -> 基于 Score 进行加权随机采样 (填满剩余 needed)
                selected_subset = self._weighted_sample(pool, needed)
                selected_nodes.extend(selected_subset)
                break  # 选够了，结束

        return selected_nodes

    def release_parents(self, parents: list[Function]):
        """
        释放被锁定的父代组合。
        必须在 TreeEoH 中调用 (最好在 finally 块中)。
        """
        if not parents:
            return

        with self._lock:
            # 重建 Key
            key = tuple(sorted([func.ID for func in parents]))
            if key in self._processing_parents:
                self._processing_parents.remove(key)

    def _weighted_sample(self, node_list, k):
        """
        辅助函数：基于排名的加权无放回采样 (Linear Rank-Based Selection)。
        解决了原始 Softmax 在分数差异巨大（如 -5000 vs -10）时，
        导致低分个体选中概率为 0 的问题。
        """
        if k == 0:
            return []

        # 如果需要的数量大于等于列表长度，直接全选
        if k >= len(node_list):
            return node_list

        scores = np.array([node._func.score for node in node_list])

        # --- 修改开始: 使用排名代替原始分数 ---

        # 1. 获取排名 (从小到大，0 表示分数最低，N-1 表示分数最高)
        # argsort 调用两次可以得到每个元素的排名索引
        ranks = np.argsort(np.argsort(scores))

        # 2. 计算权重：使用线性排名
        # 最差的个体权重为 1，最好的个体权重为 N + Alpha
        # 这种方式保证了最差个体也有 1/Sum 的概率被选中
        # 你可以通过调整 base_weight 来调节“贫富差距”，base 越大，选择越均匀
        base_weight = 1.0
        weights = ranks + base_weight

        # 3. 归一化为概率
        probs = weights / np.sum(weights)

        # --- 修改结束 ---

        # 使用 numpy 进行加权无放回采样
        selected = np.random.choice(node_list, size=k, replace=False, p=probs)
        return list(selected)

    def feedback(self, parents: List[str], prompt_type: str):
        if parents in [None, [], [None]]:
            return

        with self._lock:
            for parent_id in parents:
                parent_node = self._nodes_by_id[parent_id]
                if parent_node._ID not in self._tabu_dict[prompt_type]:
                    self._tabu_dict[prompt_type][parent_node._ID] = 0
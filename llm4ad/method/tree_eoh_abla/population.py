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
        return f"TreeNode(level={self._level}, children={len(self._children)})"


class TreePopulation:
    """
    算子树管理器。

    这个类维护树的完整结构和所有访问索引。
    所有对树的修改（如添加节点）都应通过这个类来完成，
    以确保所有索引始终保持同步。
    """

    def __init__(self):
        self._lock = Lock()
        self._roots = []

        # 2. 根据 算子Index 访问
        self._nodes_by_index: dict[int, TreeNode] = {}

        # 3. 根据 树的层次 访问
        self._nodes_by_level = defaultdict(list)

        # 4. 访问 所有叶子节点
        self._leaf_nodes: set[TreeNode] = set()

        self._sample_order = 0

        self._tabu_dict = {
            'E1': {},
            'E2': {},
            'M1': {},
            'M2': {}
        }

    def __len__(self):
        return len(self._nodes_by_index)

    def __getitem__(self, item) -> TreeNode:
        return self._nodes_by_index[item]

    @property
    def population(self):
        return [node._func for node in self._nodes_by_index.values()]

    def register_function(self, func: Function):
        # Note: unlike EoH, we only accept valid functions
        if func.score is None:
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
        sample_order = self._sample_order
        self._sample_order += 1

        if func.parents in [None, [], [None]]:
            parents = None
            level = 0
        else:
            parents = [self.get_node_by_index(pidx) for pidx in func.parents]
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

        # --- 4. 维护所有索引 (关键步骤!) ---

        # 维护 Index 索引
        self._nodes_by_index[sample_order] = new_node

        # 维护 Level 索引
        self._nodes_by_level[level].append(new_node)

        # 维护 Leaf 索引
        # 新节点总是叶子节点
        self._leaf_nodes.add(new_node)

        return new_node

    # --- 灵活的访问方法 (全部是 O(1) 复杂度) ---

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

    def select(self, n, prompt_type) -> Function | list[Function]:
        with self._lock:
            # step 1: if there are leaf nodes not in tabu, select the ones with minimum level
            tabu_dict = self._tabu_dict[prompt_type]
            leaf_list = [node for node in self._leaf_nodes if node._ID not in tabu_dict]

            if len(leaf_list) >= n:
                min_level_nodes = sorted(leaf_list, key=lambda x: x._level)[:n]
                return [x._func for x in min_level_nodes]

            # step 2: if there are non-leaf nodes not in tabu, select from all nodes not in tabu
            node_list = [node for node in self._nodes_by_index.values() if node._ID not in tabu_dict]
            if len(node_list) >= n:
                min_nodes = sorted(node_list, key=lambda x: x._level)[:n]
                return [x._func for x in min_nodes]

            # step 3: otherwise, select the best available node
            else:
                func = sorted(node_list, key=lambda x: x._func.score, reverse=True)
                p = [1 / (r + len(func)) for r in range(len(func))]
                p = np.array(p)
                p = p / np.sum(p)
                return np.random.choice(func, p=p)

    def feedback(self, parents: List[str], prompt_type: str):
        if parents in [None, [], [None]]:
            return

        with self._lock:
            for parent_id in parents:
                parent_node = self._nodes_by_id[parent_id]
                if parent_node._ID not in self._tabu_dict[prompt_type]:
                    self._tabu_dict[prompt_type][parent_node._ID] = 0

# class SelectionPriorityQueue:
#     def __init__(self, priority_func):
#         self._heap = []
#         self._counter = itertools.count()
#         self.priority_func = priority_func
#         self._lock = threading.Lock()
#
#     def push(self, item):
#         with self._lock:
#             priority = self.priority_func(item)
#             count = next(self._counter)
#             heapq.heappush(self._heap, (priority, count, item))
#
#     def pop(self):
#         if not self._heap:
#             raise IndexError("pop from an empty priority queue")
#
#         with self._lock:
#             priority, count, item = heapq.heappop(self._heap)
#         return item
#
#     def peek(self):
#         if not self._heap:
#             raise None
#
#         with self._lock:
#             priority, count, item = self._heap[0]
#         return item
#
#     def __len__(self):
#         return len(self._heap)
#
#     def is_empty(self):
#         return len(self._heap) == 0

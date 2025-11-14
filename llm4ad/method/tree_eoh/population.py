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
import queue
import heapq
import itertools
import threading


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._ID_set = set([func._ID for func in self._population])  # Step 1.3 记录所有ID

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._generation = generation

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    def survival(self):
        pop = self._population + self._next_gen_pop
        pop = sorted(pop, key=lambda f: f.score, reverse=True)
        self._population = pop[:self._pop_size]
        self._next_gen_pop = []
        self._generation += 1

    def if_ID_duplicate(self, ID):
        return ID in self._ID_set

    def register_function(self, func: Function):
        # in population initialization, we only accept valid functions
        if self._generation == 0 and func.score is None:
            return

        # if the ID is duplicated, discard      # Step 1.4: 用ID去重；否则添加ID
        if self.if_ID_duplicate(func.ID):  # Note: this line should have no effect
            return

        # else, record the new ID
        self._ID_set.add(func.ID)

        # if the score is None, we still put it into the population,
        # we set the score to '-inf'
        if func.score is None:
            func.score = float('-inf')
        try:
            self._lock.acquire()
            if self.has_duplicate_function(func):
                func.score = float('-inf')
            # register to next_gen
            self._next_gen_pop.append(func)
            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size:
                self.survival()
        except Exception as e:
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        for f in self._population:
            if str(f) == str(func) or func.score == f.score:
                return True
        for f in self._next_gen_pop:
            if str(f) == str(func) or func.score == f.score:
                return True
        return False

    def selection(self) -> Function:
        funcs = [f for f in self._population if not math.isinf(f.score)]
        func = sorted(funcs, key=lambda f: f.score, reverse=True)
        p = [1 / (r + len(func)) for r in range(len(func))]
        p = np.array(p)
        p = p / np.sum(p)
        return np.random.choice(func, p=p)


class TreeNode():
    def __init__(self, sample_order, func):
        self._ID = func.ID
        self._sample_order = sample_order
        self._func = func

        self._prompt_type = func.prompt_type
        self._parents = func.parents
        self._children = []
        self._level = sum([p.level for p in self._parents]) // len(
            self._parents)  # TODO: if there are multiple parents, what's the level?

        self._search_history = defaultdict(int)

    def is_leaf(self):
        return len(self._children) == 0

    def is_root(self):
        return self._parents is None

    def __repr__(self):
        return f"TreeNode(ID={self._ID}, level={self._level}, children={len(self._children)})"


class SelectionPriorityQueue:
    def __init__(self, priority_func):
        self._heap = []
        self._counter = itertools.count()
        self.priority_func = priority_func
        self._lock = threading.Lock()

    def push(self, item):
        with self._lock:
            priority = self.priority_func(item)
            count = next(self._counter)
            heapq.heappush(self._heap, (priority, count, item))

    def pop(self):
        if not self._heap:
            raise IndexError("pop from an empty priority queue")

        with self._lock:
            priority, count, item = heapq.heappop(self._heap)
        return item

    def peek(self):
        if not self._heap:
            raise None

        with self._lock:
            priority, count, item = self._heap[0]
        return item

    def __len__(self):
        return len(self._heap)

    def is_empty(self):
        return len(self._heap) == 0


class TreePopulation(Population):
    """
    算子树管理器。

    这个类维护树的完整结构和所有访问索引。
    所有对树的修改（如添加节点）都应通过这个类来完成，
    以确保所有索引始终保持同步。
    """

    def __init__(self):
        super().__init__(0, 0)

        self._roots = []

        # 1. 根据 算子ID 访问
        self._nodes_by_id = {}

        # 2. 根据 算子Index 访问
        self._nodes_by_index = {}

        # 3. 根据 树的层次 访问
        self._nodes_by_level = defaultdict(list)

        # 4. 访问 所有叶子节点
        self._leaf_nodes = set()

        self._sample_order = 0

        # Queues for different prompt selection
        # Tabu for prompt enginering
        self._E1_queue = SelectionPriorityQueue(self.E1_priority)
        self._E2_queue = SelectionPriorityQueue(self.E2_priority)
        self._M1_queue = SelectionPriorityQueue(self.M1_priority)
        self._M2_queue = SelectionPriorityQueue(self.M2_priority)
        self._tabu_set = set()


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
            return
        finally:
            self._lock.release()

    def add_node(self, func: Function) -> TreeNode:
        """
        向树中添加一个新节点。
        这是唯一应该用于添加节点的方法，以确保索引一致。
        """
        sample_order = self._sample_order
        self._sample_order += 1

        parents = [self.get_node_by_id(pid) for pid in func.parents]
        new_node = TreeNode(
            sample_order=sample_order,
            func=func,
        )

        if parents in [None, [], [None]]:
            self._roots.append(new_node)
        else:
            for parent in parents:
                parent._children.append(new_node)
                if parent in self._leaf_nodes:
                    self._leaf_nodes.remove(parent)

        ID = func.ID
        level = new_node._level

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

    def survival(self):
        raise NotImplementedError("To be implemented.")

    def E1_priority(self, node: TreeNode) -> int:
        raise NotImplementedError("To be implemented.")

    def E2_priority(self, node: TreeNode) -> int:
        raise NotImplementedError("To be implemented.")

    def M1_priority(self, node: TreeNode) -> int:
        raise NotImplementedError("To be implemented.")

    def M2_priority(self, node: TreeNode) -> int:
        raise NotImplementedError("To be implemented.")

    @overrides.override()
    def selection(self, prompt_type) -> Function:
        raise NotImplementedError("To be implemented.")

    def feedback(self, parents: List[str], prompt_type: str):
        with self._lock:
            for parent_id in parents:
                parent_node = self._nodes_by_id[parent_id]
                parent_node._search_history[prompt_type] += 1

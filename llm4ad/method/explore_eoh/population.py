from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np

from .base.code import Function


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._ID_set = set([func.ID for func in self._population])  # Step 1.3 记录所有ID

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
        if self.if_ID_duplicate(func.ID):
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


# TODO: TreePopulation: record the relationship between functions.
# TODO:     nodes:
# TODO:     edges:
# TODO:     visit:
# TODO:     select:
class TreePopulation():
    pass

class TreeNode():
    pass
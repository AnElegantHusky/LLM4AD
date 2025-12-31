# Module Name: EoH
# Last Revision: 2025/2/16
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Fei Liu, Tong Xialiang, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang.
#       "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model."
#       In Forty-first International Conference on Machine Learning (ICML). 2024.
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import concurrent.futures
import time
import traceback
from threading import Thread, Lock
from typing import Optional, Literal

from .population import Population
from .profiler import AtomicProfiler
from .prompt import ExplorePrompt
from .sampler import ExploreSampler
from .base.code import Function, Program, TextFunctionProgramConverter
from .base.print_utils import print_error, print_success, print_warning
from .evaluate import Evaluation, SecureEvaluator
from ...base import LLM
# from ...base import (
#     Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
# )
from .profiler import AtomicProfiler


class Atomic:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: AtomicProfiler = None,
                 atomic_algo_dict: dict = None,
                 # num_samplers: int = 1,
                 num_evaluators: int = 1,
                 *,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Evolutionary of Heuristics.
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
            max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            selection_num   : number of selected individuals while crossover.
            use_e2_operator : if use e2 operator.
            use_m1_operator : if use m1 operator.
            use_m2_operator : if use m2 operator.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description

        # samplers and evaluators
        # self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        # population, sampler, and evaluator
        self._population = Population(pop_size=20)
        # self._sampler = ExploreSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler
        self._duplicate_lock = Lock()

        # statistics
        self.atomic_algo_dict = atomic_algo_dict

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)  # ZL: necessary


    # TODO   1.2 修改 _sample_evaluate_register
    #            score: 可能是一个元组；ID: 需要额外记录
    def _sample_evaluate_register(self, algo_name):
        """Perform following steps:
        1. Sample an algorithm using the given prompt.
        2. Evaluate it by submitting to the process/thread pool, and get the results.
        3. Add the function to the population and register it to the profiler.
        """
        sample_start = time.time()
        func = TextFunctionProgramConverter.text_to_function(self.atomic_algo_dict[algo_name])

        thought = func.docstring
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return
        # convert to Program instance
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return

        # program = self._evaluator._modify_program_code(program)

        # obtain and check ID   # Step 1.1 获取并检查ID
        ID = self._evaluation_executor.submit(
            self._evaluator.evaluate_ID,
            program
        ).result()
        func.ID = ID  # Step 1.2: 记录ID

        # evaluate
        res, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()

        # register to profiler
        func.score = res
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program))
            if isinstance(self._profiler, AtomicProfiler):
                self._profiler.register_population(self._population)

        # register to the population
        self._population.register_function(func)

    def _iteratively_use_eoh_operator(self):
        for algo_name in self.atomic_algo_dict.keys():
            self._sample_evaluate_register(algo_name)

    # def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
    #     """Execute `fn` using multithreading.
    #     In EoH, `fn` can be `self._iteratively_init_population` or `self._iteratively_use_eoh_operator`.
    #     """
    #     # threads for sampling
    #     sampler_threads = [
    #         Thread(target=fn, args=args, kwargs=kwargs)
    #         for _ in range(self._num_samplers)
    #     ]
    #     for t in sampler_threads:
    #         t.start()
    #     for t in sampler_threads:
    #         t.join()

    def run(self):
        # evolutionary search
        # self._multi_threaded_sampling(self._iteratively_use_eoh_operator)

        self._iteratively_use_eoh_operator()

        # finish
        if self._profiler is not None:
            self._profiler.finish()




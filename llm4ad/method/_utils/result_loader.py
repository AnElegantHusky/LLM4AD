import os
import json
from pathlib import Path


class ResultLoader:
    def __init__(self, log_dir: str):
        self._log_dir = log_dir

        self.search_result = []
        self.best_result = []

        self._load_results()

    def _load_results(self):
        samples_dir = Path(self._log_dir) / "samples"
        if not samples_dir.exists():
            raise FileNotFoundError(f"No 'samples' directory in {samples_dir}. Skipping.")

        for sample_file in samples_dir.glob("*.json"):
            basename = os.path.basename(sample_file)
            with open(sample_file, 'r', encoding='utf-8') as f:
                sample_list = json.load(f)
            if '~' in basename:
                self.search_result.extend(sample_list)
            elif 'best' in basename:
                self.best_result.extend(sample_list)

    def get_best_result(self, idx):
        result_dict = self.best_result[idx]
        print(result_dict['function'])
        return result_dict

    def get_search_result(self, idx):
        result_dict = self.search_result[idx]
        print(result_dict['program'])
        return result_dict

    def get_runable_best_result(self, function_name):
        import numpy as np
        import random

        print("Loading best function... ======================")
        function_str = self.get_search_result(-1)['function']

        global_namespace = {
            'np': np,
            'random': random,
        }

        exec(function_str, global_namespace)
        return global_namespace[function_name]



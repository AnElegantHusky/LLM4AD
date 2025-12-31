import os
import json
from pathlib import Path


class ResultLoader:
    def __init__(self, log_dir: str):
        self._log_dir = log_dir

        self.search_result = []
        self.best_result = []
        self.filtered_search_result = []

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
                self.filtered_search_result.extend(
                    [sample for sample in sample_list if sample['score'] is not -float('inf')])
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

    def get_filtered_search_result(self, idx):
        result_dict = self.filtered_search_result[idx]
        print(result_dict['program'])
        return result_dict

    def get_ID_list(self):
        return [x['ID'] for x in self.search_result]

    def get_fitness_list(self):
        return [x['score'] for x in self.search_result]

    def print_search_result(self, idx):
        result_dict = self.search_result[idx]
        self._print_idx(result_dict)

    def print_best_result(self, idx):
        result_dict = self.best_result[idx]
        self._print_idx(result_dict)

    def print_filtered_search_result(self, idx):
        result_dict = self.filtered_search_result[idx]
        self._print_idx(result_dict)

    def _print_idx(self, result_dict):
        print(f"Sample Order: {result_dict['sample_order']}")
        print(f"Score: {result_dict['score']}")
        if 'ID' in result_dict:
            print(f"ID: {result_dict['ID']}")
        print('\n')
        if 'thought' in result_dict:
            print(f"Thought: \n{result_dict['thought']}\n")
        elif 'algorithm' in result_dict:
            print(f"Algorithm: \n{result_dict['algorithm']}\n")
        print(f"Function: \n{result_dict['function']}")


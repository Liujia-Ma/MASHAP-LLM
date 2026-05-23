import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from datetime import datetime
import numpy as np
from mashap_llm.mas import MAS

class BaseEvaluator(ABC):

    def __init__(
            self, 
            mas: MAS, 
            data_path: str | os.PathLike = None, 
            output_dir: str | os.PathLike = None,
            metrics_filename: str = None,
            metrics_timestamp: bool = False,
            response_filename: str = None,
            **kwargs: Dict[str, Any]
        ):
        self.mas = mas
        self.responses = []
        self.results = []
        self.metrics = {}
        self.eval_seed = kwargs.get("eval_seed", None)
        self.dataset = self.load_data(data_path, seed=self.eval_seed)
        self.output_dir = output_dir or "."
        self.metrics_filename = metrics_filename
        self.metrics_timestamp = metrics_timestamp
        self.response_filename = response_filename
        self.args = kwargs

    @staticmethod
    def _normalize_base_filename(filename: str | None, default_base: str) -> str:
        if filename is None or str(filename).strip() == "":
            return default_base
        return os.path.splitext(os.path.basename(str(filename)))[0]

    @staticmethod
    def _build_timestamped_json_name(base_name: str, add_timestamp: bool) -> str:
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            return f"{base_name}_{timestamp}.json"
        return f"{base_name}.json"

    def load_data(self, data_path: str | os.PathLike, *, seed: int | None):
        if str(data_path).endswith(".jsonl"):
            data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        # Always shuffle for evaluation; if seed is None, use non-deterministic shuffle.
        if len(data) > 1:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(data))
            data = [data[i] for i in indices]
            if seed is None:
                print(f"Loaded {len(data)} entries from {data_path} (shuffled with random seed)")
            else:
                print(f"Loaded {len(data)} entries from {data_path} (shuffled with seed={seed})")
        else:
            print(f"Loaded {len(data)} entries from {data_path}")
        return data

    @abstractmethod
    def evaluate(self):
        """evaluate logics"""
        pass

    def _save_metrics(self):
        if not self.metrics:
            print("⚠️ No metrics to save.")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        metrics_base = self._normalize_base_filename(self.metrics_filename, "metrics")
        # If user explicitly provides --metrics_filename, always append timestamp.
        add_timestamp = self.metrics_timestamp or (self.metrics_filename is not None)
        filename = self._build_timestamped_json_name(metrics_base, add_timestamp)
        output_path = os.path.join(self.output_dir, filename)
        self.metrics.setdefault("metrics", metrics_base)
        sanitized_metrics = {
            k: float(v) if isinstance(v, (torch.Tensor, np.generic)) else v 
            for k, v in self.metrics.items()
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sanitized_metrics, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"✅ Evaluation metrics saved to {output_path}.")

    def _save_responses(self):
        if not self.responses:
            print("⚠️ No responses to save.")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        response_base = self._normalize_base_filename(self.response_filename, "responses")
        # If user explicitly provides --response_filename, always append timestamp.
        add_timestamp = self.response_filename is not None
        response_name = self._build_timestamped_json_name(response_base, add_timestamp)
        output_file = os.path.join(self.output_dir, response_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.responses, f, indent=2, ensure_ascii=False, default=lambda x: str(x))
        print(f"\n✅ Successfully saved {len(self.responses)} responses to {output_file}.")

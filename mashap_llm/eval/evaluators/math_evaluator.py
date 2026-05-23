import numpy as np
from tqdm import tqdm
from mashap_llm.eval.evaluators.base_evaluator import BaseEvaluator
from mashap_llm.eval.utils.grader import math_equal
from mashap_llm.eval.utils.parse_utils_qwen import extract_answer


class MathEvaluator(BaseEvaluator):
    @staticmethod
    def add_args(parser):
        """添加数学评估专用参数"""
        parser.add_argument('--math_strict_mode', action='store_true', help='数学严格模式')
        return parser
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _extract_gt(entry: dict) -> str:
        # Support multiple benchmark schemas.
        if "final_answer" in entry:
            return entry["final_answer"]
        if "answer" in entry:
            return entry["answer"]
        raise KeyError(
            "Ground-truth answer field not found. Expected one of: "
            "['final_answer', 'answer']."
        )
    
    def evaluate(self):
        correct = 0.0
        self.metrics["accuracy"] = 0.
        self.metrics["correct"] = 0.0
        self.metrics["total"] = len(self.dataset)
        self.metrics["metrics"] = self._normalize_base_filename(self.metrics_filename, "metrics")
        with tqdm(total=len(self.dataset), desc="Evaluating...") as pbar:
            for idx, entry in enumerate(tqdm(self.dataset), 1):
                response = {}
                to_check = []
                problem = entry["problem"]
                gt = self._extract_gt(entry)
                response["problem"] = problem
                response["gt"] = gt
                problem = "<|im_start|>problem: " + problem + " <|im_end|>\n"
                problem = np.array([problem for _ in range(self.mas.num_agents)], dtype=np.object_)
                problem = np.expand_dims(problem, axis=0)
                _, actions, _ = self.mas.get_actions_sequential(problem)
                actions = np.squeeze(actions, axis=0)
                for agent_idx, profile in enumerate(self.mas.profiles):
                    response[profile["role"]] = actions[agent_idx]
                    if profile["with_answer"]:
                        to_check.append(actions[agent_idx])
                result = self.check_response(to_check, gt)
                response["result"] = result
                sample_correct = (sum(result) / len(result)) if len(result) > 0 else 0.0
                correct += sample_correct
                current_acc = correct / idx if idx > 0 else 0.0
                pbar.set_postfix({'acc': f'{current_acc:.2%}',})
                pbar.update(1)
                self.responses.append(response)
        self.metrics["correct"] = float(correct)
        self.metrics["accuracy"] = (correct / len(self.dataset)) if len(self.dataset) > 0 else 0.0
        self._save_responses()
        self._save_metrics()
        return self.metrics
    
    def check_response(self, to_check: list, gt: str) -> list:
        result = []
        extracted_gt = extract_answer(gt, data_name='math')
        for response in to_check:
            extracted_response = extract_answer(response, data_name='math')
            result.append(math_equal(extracted_response, extracted_gt))
        return result
    

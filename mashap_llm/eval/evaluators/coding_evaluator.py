import numpy as np
from tqdm import tqdm

from mashap_llm.envs.coding import prime_code
from mashap_llm.eval.evaluators.base_evaluator import BaseEvaluator


class CodingEvaluator(BaseEvaluator):
    @staticmethod
    def add_args(parser):
        return parser

    @staticmethod
    def _extract_problem(entry: dict) -> str:
        if "prompt" in entry and isinstance(entry["prompt"], list) and len(entry["prompt"]) > 0:
            first_msg = entry["prompt"][0]
            if isinstance(first_msg, dict) and "content" in first_msg:
                return first_msg["content"]
        if "problem" in entry:
            return entry["problem"]
        raise KeyError("Coding prompt not found. Expected one of: ['prompt[0].content', 'problem'].")

    @staticmethod
    def _extract_test_cases(entry: dict):
        reward_model = entry.get("reward_model", {})
        if isinstance(reward_model, dict) and "ground_truth" in reward_model:
            return reward_model["ground_truth"]
        if "ground_truth" in entry:
            return entry["ground_truth"]
        raise KeyError("Coding ground truth not found. Expected one of: ['reward_model.ground_truth', 'ground_truth'].")

    @staticmethod
    def _score_action(action: str, test_cases) -> float:
        res = prime_code.compute_score(action, test_cases, continuous=True)
        if isinstance(res, dict):
            return float(res)
        if isinstance(res, (int, float, bool, np.number)):
            return float(res)
        if isinstance(res, (list, tuple)) and len(res) > 0:
            return float(res[0])
        return float(res)

    def evaluate(self):
        total_score = 0.0
        self.metrics["accuracy"] = 0.0
        self.metrics["correct"] = 0.0
        self.metrics["total"] = len(self.dataset)
        self.metrics["metrics"] = self._normalize_base_filename(self.metrics_filename, "metrics")

        with tqdm(total=len(self.dataset), desc="Evaluating...") as pbar:
            for idx, entry in enumerate(self.dataset, 1):
                response = {}
                problem_text = self._extract_problem(entry)
                test_cases = self._extract_test_cases(entry)
                response["problem"] = problem_text
                response["gt"] = test_cases

                prompt = "<|im_start|>problem: " + problem_text + " <|im_end|>\n"
                prompt = np.array([prompt for _ in range(self.mas.num_agents)], dtype=np.object_)
                prompt = np.expand_dims(prompt, axis=0)
                _, actions, _ = self.mas.get_actions_sequential(prompt)
                actions = np.squeeze(actions, axis=0)

                answer_scores = []
                for agent_idx, profile in enumerate(self.mas.profiles):
                    action = actions[agent_idx]
                    response[profile["role"]] = action
                    if profile.get("with_answer", False):
                        score = self._score_action(action, test_cases)
                        answer_scores.append(score)
                        response[f"{profile['role']}_score"] = score

                sample_score = (sum(answer_scores) / len(answer_scores)) if len(answer_scores) > 0 else 0.0
                response["result"] = sample_score
                total_score += sample_score

                current_acc = total_score / idx if idx > 0 else 0.0
                pbar.set_postfix({"acc": f"{current_acc:.2%}"})
                pbar.update(1)
                self.responses.append(response)

        self.metrics["correct"] = float(total_score)
        self.metrics["accuracy"] = (total_score / len(self.dataset)) if len(self.dataset) > 0 else 0.0
        self._save_responses()
        self._save_metrics()
        return self.metrics

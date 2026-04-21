import numpy as np
import json
import random
import re
from typing import Optional
from . import prime_code

DEFAULT_ABSENCE_MESSAGE_TEMPLATE = "System: The {role} did not participate in this round."

# training data with mode="train" and testing data with mode="test"
def load_dataset(dataset_path, mode):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset

def load_profiles(path):
    with open(path, 'r') as file:
        profiles = json.load(file)
    return profiles

class CodingEnv:

    def __init__(
        self,
        rank,
        model_name,
        num_agents,
        profile_path,
        dataset_path,
        horizon,
        mode,
        seed=None,
        experiment_mode="baseline",
        debug_print_state=False,
    ):
        
        self.rank = rank
        self.mode = mode
        if seed is not None:
            random.seed(seed)
        self.model_name = model_name
        self.dataset = load_dataset(dataset_path=dataset_path, mode=mode)
        self.profiles = load_profiles(profile_path)
        self.n_agents = num_agents
        assert self.n_agents == len(self.profiles), "Number of agents must match the number of profiles."
        self.max_steps = horizon
        self.step_count = 0
        # Reward allocation config:
        # - terminal: original baseline behavior
        # - shapley: allocate reward by marginal contribution
        self.experiment_mode = experiment_mode
        self.debug_print_state = debug_print_state
        self.llmshap_rollout_allocator_fn = None
        
        self.problem = None
        self.label = None
        self.current_state = None
        if rank == 0:
            print(f"The {mode} mode environment has {len(self.dataset)} entries in total.")

    def reset(self):
        # Keep sampling until a valid label is found
        max_trials = max(1, len(self.dataset) * 2)
        for _ in range(max_trials):
            problem_answer_pair = random.choice(self.dataset)
            # Try to get the final answer label
            label = problem_answer_pair['reward_model']['ground_truth']
            # If label is still None, skip this sample
            if label is None:
                continue
            # Valid sample found
            self.problem = problem_answer_pair["prompt"][0]['content']
            self.label = label
            break
        else:
            raise RuntimeError(
                "CodingEnv.reset failed to sample a valid label after multiple trials. "
                "Please check dataset quality (missing reward_model.ground_truth)."
            )

        self.current_state = '<|im_start|>problem: ' + self.problem + "<|im_end|>\n"
        self.history = []
        obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        self.step_count = 0
        return obs
    
    def set_llmshap_rollout_allocator_fn(self, allocator_fn):
        """
        Inject Q-critic rollout allocator callback.

        allocator_fn signature:
            allocator_fn(actions: np.ndarray[str], base_state: str, global_score: float, original_problem: str, gt) -> tuple[list[float], dict]
        """
        self.llmshap_rollout_allocator_fn = allocator_fn

    def _score_terminal(self, actions) -> float:
        """
        Baseline terminal score (readability-oriented explicit path).
        This preserves existing behavior by evaluating answer-capable agents only.
        """
        actions_to_check = [actions[i] for i in range(self.n_agents) if self.profiles[i]["with_answer"]]
        if len(actions_to_check) == 0:
            return 0.0
        score = 0.0
        for action in actions_to_check:
            score += self.compute_reward(action, self.label)
        score /= len(actions_to_check)
        return float(score)

    def step(self, actions):
        self.step_count += 1
        base_state = self.current_state
        self.state_transition(actions)
        # Global score should stay consistent across all experiment_mode modes.
        score = self._score_terminal(actions)
        
        if score > 0.0 or self.step_count >= self.max_steps:
            dones = np.ones((self.n_agents), dtype=bool)
            # score -= self.step_count # penalize for more steps
        else:
            dones = np.zeros((self.n_agents), dtype=bool)
            
        if score == 0.0:
            self.current_state = self.current_state + "judge: The answer is incorrect.\n"
        else:
            self.current_state = self.current_state + "judge: The answer is correct.\n"
        if self.debug_print_state and self.rank == 0:
            print(
                f"[debug-state][coding][mode={self.mode}] step={self.step_count} "
                f"chars={len(self.current_state)} score={float(score):.4f}\n{self.current_state}"
            )

        next_obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        if self.experiment_mode in ("llmshap", "pureshap"):
            if self.llmshap_rollout_allocator_fn is None:
                raise RuntimeError(
                    f"{self.experiment_mode} mode requires llmshap_rollout_allocator_fn. "
                    "Please ensure runner injects it before training."
                )
            alloc_out = self.llmshap_rollout_allocator_fn(actions, base_state, float(score), self.problem, self.label)
            if isinstance(alloc_out, tuple) and len(alloc_out) == 2:
                rewards, shapley_debug = alloc_out
            else:
                rewards, shapley_debug = alloc_out, {}
            if not isinstance(shapley_debug, dict):
                shapley_debug = {}
            shapley_debug.setdefault("allocation_mode", self.experiment_mode)
            shapley_debug.setdefault("total_score", float(score))
            shapley_debug.setdefault("counterfactual_mode", "rollout")
            shapley_debug.setdefault("absence_message_template", DEFAULT_ABSENCE_MESSAGE_TEMPLATE)
        elif self.experiment_mode == "baseline":
            rewards = [0 if idx != self.n_agents - 1 else score for idx in range(self.n_agents)]
            shapley_debug = {"allocation_mode": "baseline", "total_score": float(score)}
        else:
            raise ValueError(
                f"Unknown experiment_mode={self.experiment_mode}. "
                "Supported modes: baseline, llmshap, pureshap."
            )
        infos = {
            "state": self.current_state,
            "gt": self.label,
            "episodic_return": score,
            "experiment_mode": self.experiment_mode,
            "reward_vector": rewards,
            **shapley_debug,
        }
        return next_obs, rewards, dones, infos

    def state_transition(self, actions):
        for i, action in enumerate(actions):
            self.current_state = self.current_state + self.profiles[i]["role"] + ": " + action + "\n"

    def compute_reward(self, solution_str, test_cases):
        res = prime_code.compute_score(solution_str, test_cases, continuous=True)

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])

    def seed(self, seed):
        np.random.seed(seed)

    def get_env_info(self):
        env_info = {"n_agents": self.n_agents}
        return env_info
    
    def close(self):
        pass 

import numpy as np
import json
import random
import re
from typing import Optional
from . import math
from . import math_verify

DEFAULT_ABSENCE_MESSAGE_TEMPLATE = "System: The {role} did not participate in this round."

# training data with mode="train" and testing data with mode="test"
def load_dataset(dataset_path, mode):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset

def load_profiles(path):
    with open(path, 'r') as file:
        profiles = json.load(file)
    return profiles

def extract_boxed_value(text):
    """
    Extracts the first LaTeX \\boxed{...} expression from the string, supporting nested braces.

    Parameters:
        text (str): The input string containing LaTeX.

    Returns:
        str or None: The content inside the first \\boxed{...}, or None if not found.
    """
    start = text.find(r'\boxed{')
    if start == -1:
        return None

    i = start + len(r'\boxed{')
    brace_count = 1
    content = []

    while i < len(text):
        char = text[i]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

        if brace_count == 0:
            break
        content.append(char)
        i += 1

    return ''.join(content) if brace_count == 0 else None

class MathEnv:

    def __init__(
        self,
        rank,
        model_name,
        num_agents,
        profile_path,
        dataset_path,
        horizon,
        mode,
        reward_allocation="terminal",
        clip_value=-1.0,
        debug_print_state=False,
    ):
        
        self.rank = rank
        self.mode = mode
        self.model_name = model_name
        self.dataset = load_dataset(dataset_path=dataset_path, mode=mode)
        self.profiles = load_profiles(profile_path)
        self.n_agents = num_agents
        assert self.n_agents == len(self.profiles), "Number of agents must match the number of profiles."
        self.max_steps = horizon
        self.step_count = 0
        # Reward allocation config:
        # - terminal: baseline behavior ([0, ..., score] to last agent)
        # - shapley: fair per-agent allocation by marginal contribution
        self.reward_allocation = reward_allocation
        self.clip_value = clip_value
        self.debug_print_state = debug_print_state
        self.qcritic_masked_allocator_fn = None
        self.qcritic_rollout_allocator_fn = None
        
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
            label = problem_answer_pair.get("final_answer")
            if not label:
                label = extract_boxed_value(problem_answer_pair.get("solution", ""))
            
            # If label is still None, skip this sample
            if label is None:
                continue
            
            # Valid sample found
            self.problem = problem_answer_pair["problem"]
            self.label = label
            break
        else:
            raise RuntimeError(
                "MathEnv.reset failed to sample a valid label after multiple trials. "
                "Please check dataset quality (missing final_answer/boxed solution)."
            )

        self.current_state = '<|im_start|>problem: ' + self.problem + "<|im_end|>\n"
        self.history = []
        obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        self.step_count = 0
        return obs
    
    def set_qcritic_masked_allocator_fn(self, allocator_fn):
        """
        Inject Q-critic masked allocator callback.

        allocator_fn signature:
            allocator_fn(actions: np.ndarray[str], global_score: float, original_problem: str, agent_states: str) -> list[float]
        """
        self.qcritic_masked_allocator_fn = allocator_fn

    def set_qcritic_rollout_allocator_fn(self, allocator_fn):
        """
        Inject Q-critic rollout allocator callback.

        allocator_fn signature:
            allocator_fn(actions: np.ndarray[str], base_state: str, global_score: float, original_problem: str) -> tuple[list[float], dict]
        """
        self.qcritic_rollout_allocator_fn = allocator_fn

    def step(self, actions):
        self.step_count += 1
        base_state = self.current_state
        self.state_transition(actions)
        actions_to_check = [actions[i] for i in range(self.n_agents) if self.profiles[i]["with_answer"]]
        if len(actions_to_check) == 0:
            score = 0.0
        else:
            score = 0.0
            for action in actions_to_check:
                score += self.compute_reward(action, self.label)
            score /= len(actions_to_check)
            score = float(score)
        
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
                f"[debug-state][math][mode={self.mode}] step={self.step_count} "
                f"chars={len(self.current_state)} score={float(score):.4f}\n{self.current_state}"
            )

        next_obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        
        if self.reward_allocation == "qcritic_rollout":
            if self.qcritic_rollout_allocator_fn is None:
                raise RuntimeError(
                    "qcritic_rollout mode requires qcritic_rollout_allocator_fn. "
                    "Please ensure runner injects it before training."
                )
            alloc_out = self.qcritic_rollout_allocator_fn(actions, base_state, float(score), self.problem)
            if isinstance(alloc_out, tuple) and len(alloc_out) == 2:
                rewards, shapley_debug = alloc_out
            else:
                rewards, shapley_debug = alloc_out, {}
            if not isinstance(shapley_debug, dict):
                shapley_debug = {}
            shapley_debug.setdefault("allocation_mode", "qcritic_rollout")
            shapley_debug.setdefault("total_score", float(score))
            shapley_debug.setdefault("counterfactual_mode", "rollout")
            shapley_debug.setdefault("absence_message_template", DEFAULT_ABSENCE_MESSAGE_TEMPLATE)
        elif self.reward_allocation == "qcritic_masked":
            if self.qcritic_masked_allocator_fn is None:
                raise RuntimeError(
                    "qcritic_masked mode requires qcritic_masked_allocator_fn. "
                    "Please ensure runner injects it before training."
                )
            alloc_out = self.qcritic_masked_allocator_fn(actions, float(score), self.problem, base_state)
            if isinstance(alloc_out, tuple) and len(alloc_out) == 2:
                rewards, sr_stats = alloc_out
            else:
                rewards, sr_stats = alloc_out, {}
            shapley_debug = {"allocation_mode": "qcritic_masked", "total_score": float(score)}
            if isinstance(sr_stats, dict):
                if sr_stats.get("loss", None) is not None:
                    shapley_debug["qcritic_loss"] = float(sr_stats["loss"])
                if sr_stats.get("grad_norm", None) is not None:
                    shapley_debug["qcritic_grad_norm"] = float(sr_stats["grad_norm"])
        else:
            rewards = [0 if idx != self.n_agents - 1 else score for idx in range(self.n_agents)]
            shapley_debug = {"allocation_mode": "terminal", "total_score": float(score)}
        
        infos = {
            "state": self.current_state,
            "gt": self.label,
            "episodic_return": score,
            "reward_allocation": self.reward_allocation,
            "reward_vector": rewards,
            **shapley_debug,
        }
        return next_obs, rewards, dones, infos

    def state_transition(self, actions):
        for i, action in enumerate(actions):
            self.current_state = self.current_state + self.profiles[i]["role"] + ": " + action + "\n"

    def compute_reward(self, solution_str, gt):

        # res = math.compute_score(solution_str, gt)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:
        res = math_verify.compute_score(solution_str, gt)

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

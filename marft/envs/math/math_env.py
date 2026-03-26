import numpy as np
import json
import random
import re
from typing import Optional
from . import math
from . import math_verify

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

    def __init__(self, rank, model_name, num_agents, profile_path, dataset_path, horizon, mode):
        
        self.rank = rank
        self.mode = mode
        self.model_name = model_name
        self.dataset = load_dataset(dataset_path=dataset_path, mode=mode)
        self.profiles = load_profiles(profile_path)
        self.n_agents = num_agents
        assert self.n_agents == len(self.profiles), "Number of agents must match the number of profiles."
        self.max_steps = horizon
        self.step_count = 0
        
        self.problem = None
        self.label = None
        self.current_state = None
        if rank == 0:
            print(f"The {mode} mode environment has {len(self.dataset)} entries in total.")

    def reset(self):
        # Keep sampling until a valid label is found
        while True:
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

        self.current_state = '<|im_start|>problem: ' + self.problem + "<|im_end|>\n"
        self.history = []
        obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        self.step_count = 0
        return obs
    
    def step(self, actions):
        self.step_count += 1
        actions_to_check = []
        self.state_transition(actions)

        for i in range(self.n_agents):
            if self.profiles[i]["with_answer"]:
                actions_to_check.append(actions[i])

        score = 0.0
        for action in actions_to_check:
            # if self._is_correct(action): 
            #     score += 1.0
            score += self.compute_reward(action, self.label)
        score /= len(actions_to_check) # normalize
        
        if score > 0.0 or self.step_count >= self.max_steps:
            dones = np.ones((self.n_agents), dtype=bool)
            # score -= self.step_count # penalize for more steps
        else:
            dones = np.zeros((self.n_agents), dtype=bool)
            
        if score == 0.0:
            self.current_state = self.current_state + "judge: The answer is incorrect.\n"
        else:
            self.current_state = self.current_state + "judge: The answer is correct.\n"

        next_obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        rewards = [0 if idx != self.n_agents - 1 else score for idx in range(self.n_agents)]
        infos = {"state": self.current_state, "gt": self.label, "episodic_return": score}
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
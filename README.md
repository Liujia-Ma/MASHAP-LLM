# MASHAP-LLM: Multi-Agent Fine-Tuning with Context-aware Shapley Value Reward Allocation
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

MASHAP-LLM turns sparse system-level verifier feedback into per-agent rewards using Shapley value allocation, enabling stable multi-agent LLM fine-tuning with action-level APPO. It supports exact Shapley (PureSHAP) and learned coalition-value estimation (LLM-SHAP).

## Table of Contents
- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
- [Reward Allocation Modes](#reward-allocation-modes)
- [Environment Extension](#environment-extension)
- [Multi-Adapter](#multi-adapter)
- [Agent-by-Agent Training](#agent-by-agent-training)
- [Resume Training](#resume-training)
- [License](#license)
- [Citation](#citation)

## About
MASHAP-LLM formulates multi-agent LLM fine-tuning as a coalition-value game. Each environment step yields a system-level score from a verifier, which is then allocated to agents via Shapley values to form role-aware learning signals. In LLM-SHAP mode, a contextual coalition-value estimator is trained from counterfactual rollouts and then frozen to provide fast coalition value predictions; in PureSHAP, coalition values are computed directly by rollouts/evaluation.

## Features
- Shapley-based reward allocation from system-level scores to agent-level rewards
- LLM-SHAP estimator training and inference for efficient coalition valuation
- PureSHAP exact coalition evaluation for reference experiments
- Action-level APPO updates with multi-adapter policies
- Extensible environments, runners, and reward allocators

## Getting Started

### Installation
1. Create a virtual environment:
   ```bash
   conda create -n mashap_llm
   conda activate mashap_llm
   ```

2. Clone the repository and install dependencies:
   ```bash
   git clone <repo_url>
   cd MASHAP-LLM
   pip install -r requirements.txt
   ```

**Note**: You may need to adjust package versions to match your CUDA version.

## Reward Allocation Modes
The framework supports three experiment modes:

- `baseline`: original behavior, only the last agent receives the task score.
- `llmshap`: allocate rewards by coalition-specific counterfactual rollouts, then value coalitions with LLM-SHAP and distribute rewards by Shapley.
- `pureshap`: bypass LLM-SHAP prediction, enumerate coalition scores directly from environment rollout/evaluation, and compute exact Shapley values.

Related arguments:

- `--experiment_mode {baseline,llmshap,pureshap}`
- `--llmshap_lr` (used in `llmshap`)

## Environment Extension
To create a custom environment for your specific agentic task:
1. Navigate to `mashap_llm/envs` and create a folder for your environment.
2. Create a Python file (e.g., `env_name.py`) and implement the necessary environment components:
   - `__init__`: Initialize the environment.
   - `reset`: Reset the environment state.
   - `step`: Define the agent's action step.
   - `transition`: Define state transitions.
3. Create a corresponding `runner` and `train` entry in `runner/shared` and `scripts` respectively.

**Example**:
```python
class CustomEnv:
    def __init__(self):
        # Initialize your environment
        pass

    def reset(self):
        # Reset the environment state
        pass

    def step(self, action):
        # Define how the environment responds to actions
        pass

    def transition(self, state):
        # Define state transitions
        pass
```

## Multi-Adapter
The framework supports a multi-agent system (MAS) where each agent shares the same base model but uses different **LoRA (Low-Rank Adaptation)** adapters. This allows agents to specialize in different tasks while maintaining a shared foundation. Checkpoint loading is also supported for seamless model resumption.

## Agent-by-Agent Training
The repository supports **agent-by-agent training**, where a single agent is trained while others are frozen. This is controlled by the `--agent_iteration_interval` argument, which defines the training interval for each agent.

## Resume Training
Training can be resumed from a checkpoint via `--load_path`. Under the path, there should be multiple folders containing LoRA adapter parameters and configurations. A critic model `critic.pth` can also be included and will be auto-loaded.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Citation
If you find this repository helpful, please consider citing our paper:

```bibtex
@misc{mashapllm,
  title={MASHAP-LLM: Multi-Agent Fine-Tuning with Context-aware Shapley Value Reward Allocation in Large Language Models},
  author={Anonymous},
  note={EMNLP submission}
}
```

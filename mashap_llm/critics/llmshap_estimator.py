from typing import Iterable

import torch
import torch.nn as nn

from .llmshap_network import LLMShapCriticNetwork


class CounterfactualEstimator(nn.Module):
    """
    Counterfactual coalition estimator for llmshap mode.

    Responsibilities:
    1) Train critic network on coalition/global rewards.
    2) Build masked counterfactual coalition tokens.
    3) Estimate coalition value V(S) from critic network.
    """

    def __init__(
        self,
        model_path: str,
        device: str | torch.device,
        *,
        critic_lr: float = 1e-5,
        bf16: bool = True,
        llmshap_layers: int = 1,
    ):
        super().__init__()
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.critic_network = LLMShapCriticNetwork(
            model_path=model_path,
            device=self.device,
            critic_lr=critic_lr,
            bf16=bf16,
            llmshap_layers=llmshap_layers,
        )
        self.tokenizer = self.critic_network.tokenizer
        self.pad_token_id = self.critic_network.pad_token_id
        self.last_update_stats = {"loss": None, "grad_norm": None}

    def update_critic(self, joint_tokens: torch.Tensor, global_reward: torch.Tensor | Iterable[float]) -> float:
        loss_item = self.critic_network.update(joint_tokens, global_reward)
        self.last_update_stats = dict(self.critic_network.last_update_stats)
        return loss_item

    def _prepare_baseline_tokens(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        return torch.full_like(joint_tokens, self.pad_token_id, device=joint_tokens.device)

    def _mask_counterfactual(self, joint_tokens: torch.Tensor, coalition: set[int]) -> torch.Tensor:
        masked = joint_tokens.clone()
        baseline_tokens = self._prepare_baseline_tokens(joint_tokens)
        _, num_agents, _ = masked.shape
        for i in range(num_agents):
            if i not in coalition:
                masked[:, i, :] = baseline_tokens[:, i, :]
        return masked

    @torch.no_grad()
    def estimate_coalition_value(self, joint_tokens: torch.Tensor, coalition_indices: Iterable[int]) -> torch.Tensor:
        joint_tokens = joint_tokens.to(self.device).long()
        masked = self._mask_counterfactual(joint_tokens, set(coalition_indices))
        return self.critic_network.predict(masked).detach()

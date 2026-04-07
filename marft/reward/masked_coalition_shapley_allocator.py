import itertools
import math
from typing import Iterable

import torch
import torch.nn as nn
from .qcritic_model import CentralizedQCritic


class MaskedCoalitionShapleyAllocator(nn.Module):
    """
    SR-MARFT style reward allocator:
    1) Train a centralized Q-critic on global rewards.
    2) Use Shapley value over counterfactual masked coalitions.
    3) Return per-agent intrinsic rewards.

    Expected token shape:
        joint_tokens: [batch_size, num_agents, segment_len]
    Each agent segment should be tokenized independently from a consistent prompt
    template, so segment i semantically corresponds to agent i. Counterfactual
    masking then replaces excluded agent segments with baseline tokens.
    """

    def __init__(
        self,
        model_path: str,
        device: str | torch.device,
        *,
        critic_lr: float = 1e-4,
        mask_token_type: str = "pad-only",
        agent_roles: list[str] | None = None,
        absence_message_template: str = "System: The {role} did not participate in this round.",
        clip_value: float = -1.0,
        bf16: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.mask_token_type = mask_token_type
        self.agent_roles = agent_roles
        self.absence_message_template = absence_message_template
        self.clip_value = clip_value
        self.qcritic = CentralizedQCritic(
            model_path=model_path,
            device=self.device,
            critic_lr=critic_lr,
            bf16=bf16,
        )
        self.tokenizer = self.qcritic.tokenizer
        self.pad_token_id = self.qcritic.pad_token_id
        self.last_update_stats = {"loss": None, "grad_norm": None}

    def update_critic(self, joint_tokens: torch.Tensor, global_reward: torch.Tensor | Iterable[float]) -> float:
        loss_item = self.qcritic.update(joint_tokens, global_reward)
        self.last_update_stats = dict(self.qcritic.last_update_stats)
        return loss_item

    def _prepare_baseline_tokens(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        B, N, T = joint_tokens.shape
        if self.mask_token_type == "pad-only":
            return torch.full_like(joint_tokens, self.pad_token_id, device=joint_tokens.device)
        if self.mask_token_type == "zero":
            return torch.zeros_like(joint_tokens, device=joint_tokens.device)
        if self.mask_token_type == "random":
            high_val = max(2, self.pad_token_id)
            return torch.randint_like(joint_tokens, low=1, high=high_val, device=joint_tokens.device)
        if self.mask_token_type == "mean":
            return joint_tokens.mean(dim=(0, 2), keepdim=True).expand_as(joint_tokens).long()
        if self.mask_token_type == "semantic-placeholder":
            roles = self.agent_roles if self.agent_roles is not None else [f"agent_{i}" for i in range(N)]
            token_rows = []
            for i in range(N):
                msg = self.absence_message_template.format(role=roles[i])
                ids = self.tokenizer.encode(msg, add_special_tokens=False)[:T]
                ids = ids + [self.pad_token_id] * (T - len(ids))
                token_rows.append(ids)
            per_agent = torch.tensor(token_rows, device=joint_tokens.device, dtype=torch.long).unsqueeze(0)  # [1,N,T]
            return per_agent.expand(B, N, T).contiguous()
        raise ValueError(f"Unknown mask_token_type: {self.mask_token_type}")

    def _mask_counterfactual(self, joint_tokens: torch.Tensor, coalition: set[int]) -> torch.Tensor:
        """
        Build C by masking excluded agents with pad tokens.
        """
        masked = joint_tokens.clone()
        baseline_tokens = self._prepare_baseline_tokens(joint_tokens)
        _, num_agents, _ = masked.shape
        for i in range(num_agents):
            if i not in coalition:
                masked[:, i, :] = baseline_tokens[:, i, :]
        return masked

    def compute_shapley_values(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            shapley_values: [B, N]
        """
        joint_tokens = joint_tokens.to(self.device).long()
        batch_size, num_agents, _ = joint_tokens.shape
        phi = torch.zeros(batch_size, num_agents, device=self.device, dtype=torch.float32)
        all_agents = tuple(range(num_agents))
        cache: dict[tuple[int, ...], torch.Tensor] = {}

        def coalition_value(coalition_tuple: tuple[int, ...]) -> torch.Tensor:
            key = tuple(sorted(coalition_tuple))
            if key not in cache:
                masked = self._mask_counterfactual(joint_tokens, set(key))
                cache[key] = self.qcritic.predict(masked).detach()
            return cache[key]

        factorial_n = math.factorial(num_agents)
        for i in range(num_agents):
            others = [a for a in all_agents if a != i]
            for r in range(len(others) + 1):
                for subset in itertools.combinations(others, r):
                    subset = tuple(sorted(subset))
                    subset_with_i = tuple(sorted(subset + (i,)))
                    weight = (math.factorial(len(subset)) * math.factorial(num_agents - len(subset) - 1)) / factorial_n
                    marginal = coalition_value(subset_with_i) - coalition_value(subset)
                    phi[:, i] += weight * marginal

        if self.clip_value is not None and self.clip_value > 0:
            phi = torch.clamp(phi, -self.clip_value, self.clip_value)
        return phi

    @torch.no_grad()
    def estimate_coalition_value(self, joint_tokens: torch.Tensor, coalition_indices: Iterable[int]) -> torch.Tensor:
        """
        Estimate coalition value V(S) from the current Q-critic without updating parameters.
        Returns:
            values: [B]
        """
        joint_tokens = joint_tokens.to(self.device).long()
        masked = self._mask_counterfactual(joint_tokens, set(coalition_indices))
        return self.qcritic.predict(masked).detach()

    def allocate_rewards(self, joint_tokens: torch.Tensor, global_reward: torch.Tensor | Iterable[float]) -> torch.Tensor:
        """
        Main API:
        1) Update critic with global reward.
        2) Compute Shapley values.
        3) Return per-agent reward tensor [B, N].
        """
        target = torch.as_tensor(global_reward, device=self.device, dtype=torch.float32).view(-1)
        self.update_critic(joint_tokens, target)
        rewards = self.compute_shapley_values(joint_tokens)
        sums = rewards.sum(dim=1, keepdim=True)
        safe = torch.where(torch.abs(sums) < 1e-8, torch.ones_like(sums), sums)
        rewards = rewards * (target.view(-1, 1) / safe)
        return rewards

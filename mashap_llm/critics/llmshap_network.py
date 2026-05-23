import math
import os
from typing import Iterable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMShapCriticNetwork(nn.Module):
    """
    Frozen LLM backbone + trainable value head for scalar coalition value prediction.
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
        self.llmshap_layers = llmshap_layers

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
        ).to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, "n_embd", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, "word_embed_proj_dim", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden size from backbone config.")

        if self.llmshap_layers == 3:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, 1024, bias=False),
                nn.ReLU(),
                nn.Linear(1024, 512, bias=False),
                nn.ReLU(),
                nn.Linear(512, 1, bias=False),
            ).to(self.device)
        else:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, 1, bias=False),
            ).to(self.device)

        self.optimizer = torch.optim.Adam(self.value_head.parameters(), lr=critic_lr, eps=1e-5)
        self.mse = nn.MSELoss()
        self.last_update_stats = {"loss": None, "grad_norm": None}

    def _flatten_joint_tokens(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        if joint_tokens.dim() != 3:
            raise ValueError(f"joint_tokens must be [B,N,T], got shape={tuple(joint_tokens.shape)}")
        return joint_tokens.reshape(joint_tokens.shape[0], -1)

    def predict(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        input_ids = self._flatten_joint_tokens(joint_tokens).to(self.device).long()
        attention_mask = (input_ids != self.pad_token_id).long()
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            last_hidden = outputs.hidden_states[-1][:, -1, :].float()
        return self.value_head(last_hidden).squeeze(-1)

    def update(self, joint_tokens: torch.Tensor, global_reward: torch.Tensor | Iterable[float]) -> float:
        target = torch.as_tensor(global_reward, device=self.device, dtype=torch.float32).view(-1)
        pred = self.predict(joint_tokens)
        loss = self.mse(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        grad_sq_sum = 0.0
        for p in self.value_head.parameters():
            if p.grad is not None:
                grad_sq_sum += float(torch.sum(p.grad.detach().float() ** 2).item())
        grad_norm = math.sqrt(grad_sq_sum)
        self.optimizer.step()
        loss_item = float(loss.item())
        self.last_update_stats = {"loss": loss_item, "grad_norm": float(grad_norm)}
        return loss_item

    def save_value_head(self, ckpt_path: str) -> None:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(self.value_head.state_dict(), ckpt_path)

    def load_value_head(self, ckpt_path: str, map_location: str | torch.device = "cpu") -> None:
        state_dict = torch.load(ckpt_path, map_location=map_location)
        self.value_head.load_state_dict(state_dict, strict=True)

    def save_optimizer(self, ckpt_path: str) -> None:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(self.optimizer.state_dict(), ckpt_path)

    def load_optimizer(self, ckpt_path: str, map_location: str | torch.device = "cpu") -> None:
        state_dict = torch.load(ckpt_path, map_location=map_location)
        self.optimizer.load_state_dict(state_dict)

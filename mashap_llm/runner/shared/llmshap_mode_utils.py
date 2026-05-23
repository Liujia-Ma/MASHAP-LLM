from __future__ import annotations

import os
from typing import Iterable, Sequence

import torch

from mashap_llm.critics import CounterfactualEstimator
from mashap_llm.reward import LLMShapAllocator, PureShapleyAllocator

DEFAULT_ABSENCE_MESSAGE_TEMPLATE = "System: The {role} did not participate in this round."


def build_llmshap_estimator(all_args, mas) -> CounterfactualEstimator:
    llmshap_device = mas.get_llmshap_device()
    return CounterfactualEstimator(
        model_path=all_args.model_name_or_path,
        device=llmshap_device,
        critic_lr=all_args.llmshap_lr,
        llmshap_layers=all_args.llmshap_layers,
    )


def build_llmshap_allocator(all_args) -> LLMShapAllocator:
    return LLMShapAllocator(tau_floor=float(getattr(all_args, "llmshap_tau_floor", 0.05)))


def build_pureshap_allocator() -> PureShapleyAllocator:
    return PureShapleyAllocator()


def register_llmshap_estimator_on_mas(mas, estimator: CounterfactualEstimator) -> None:
    """
    Expose llmshap estimator on MAS so trainer-level optimizer checkpointing can
    include llmshap optimizer state in optimizers.pt.
    """
    mas.llmshap_estimator = estimator
    # If trainer loaded optimizers before allocator existed, apply deferred llmshap state now.
    pending_llmshap_opt_state = getattr(mas, "_pending_llmshap_opt_state", None)
    if pending_llmshap_opt_state is not None:
        try:
            estimator.critic_network.optimizer.load_state_dict(pending_llmshap_opt_state)
            print("[LLMShap] Applied deferred optimizer state from optimizers.pt")
            mas._pending_llmshap_opt_state = None
        except Exception as e:
            print(f"[LLMShap] warning: failed to apply deferred optimizer state: {e}")


def save_llmshap_value_head(estimator: CounterfactualEstimator, checkpoint_dir: str) -> str:
    """
    Save llmshap value-head weights to a standalone file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    llmshap_value_head_path = os.path.join(checkpoint_dir, "llmshap_value_head.pth")
    estimator.critic_network.save_value_head(llmshap_value_head_path)
    return llmshap_value_head_path


def load_llmshap_checkpoint(
    estimator: CounterfactualEstimator,
    checkpoint_dir: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[bool, bool]:
    """
    Load llmshap state from a checkpoint directory.

    Returns:
        (loaded_value_head, loaded_optimizer)
    """
    llmshap = estimator.critic_network
    value_head_path = os.path.join(checkpoint_dir, "llmshap_value_head.pth")
    optimizers_path = os.path.join(checkpoint_dir, "optimizers.pt")
    legacy_optimizer_path = os.path.join(checkpoint_dir, "llmshap_optimizer.pt")

    loaded_value_head = False
    loaded_optimizer = False

    if os.path.exists(value_head_path):
        llmshap.load_value_head(value_head_path, map_location=map_location)
        loaded_value_head = True

    if os.path.exists(optimizers_path):
        ckpt = torch.load(optimizers_path, map_location=map_location)
        llmshap_opt_state = ckpt.get("llmshap_opt_state", None)
        if llmshap_opt_state is not None:
            llmshap.optimizer.load_state_dict(llmshap_opt_state)
            loaded_optimizer = True

    # Backward compatibility for older runs that saved standalone optimizer file.
    if (not loaded_optimizer) and os.path.exists(legacy_optimizer_path):
        llmshap.load_optimizer(legacy_optimizer_path, map_location=map_location)
        loaded_optimizer = True

    return loaded_value_head, loaded_optimizer


def resolve_llmshap_checkpoint_dir(load_path: str) -> str:
    """
    Resolve the actual checkpoint directory that contains llmshap_value_head.pth.
    Supports:
    1) direct step dir: .../steps_xxxx
    2) run dir: .../run_xxx (auto-pick latest steps_xxxx with llmshap checkpoint)
    """
    if os.path.exists(os.path.join(load_path, "llmshap_value_head.pth")):
        return load_path

    if not os.path.isdir(load_path):
        return load_path

    candidates = []
    for entry in os.listdir(load_path):
        entry_path = os.path.join(load_path, entry)
        if (
            os.path.isdir(entry_path)
            and entry.startswith("steps_")
            and os.path.exists(os.path.join(entry_path, "llmshap_value_head.pth"))
        ):
            candidates.append(entry_path)

    if len(candidates) == 0:
        return load_path

    return sorted(candidates)[-1]


def build_llmshap_joint_tokens(
    *,
    tokenizer,
    device: torch.device,
    profiles: list[dict],
    actions,
    coalition_indices: Sequence[int] | None,
    original_problem: str,
    agent_states: str | None,
    include_state: bool = True,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    Build [1, N, T] critic input where segment i corresponds to agent i.
    """
    coalition_set = set(range(len(profiles))) if coalition_indices is None else set(int(i) for i in coalition_indices)
    participating_roles = [profiles[i]["role"] for i in range(len(profiles)) if i in coalition_set]
    absent_roles = [profiles[i]["role"] for i in range(len(profiles)) if i not in coalition_set]
    present_text = ", ".join(participating_roles) if participating_roles else "none"
    absent_text = ", ".join(absent_roles) if absent_roles else "none"
    coalition_description = (
        f"[Coalition Membership]: current coalition includes {present_text}; "
        f"{absent_text} do(es) not participate in this round."
    )

    segment_blocks = []
    for i, profile in enumerate(profiles):
        role = profile["role"]
        # In inference mode we can drop verbose state text to reduce noise and
        # force Q-critic to focus on coalition membership + coalition actions.
        state_block = f"[Agent States/Context]: {agent_states}\n" if include_state and agent_states is not None else ""
        prompt_i = (
            f"PlaintextInstruction: {original_problem}\n"
            f"{state_block}"
            f"{coalition_description}\n"
            f"[Coalition Actions]: {role}: {str(actions[i])}\n"
            "[Evaluation Standard]: As a reward model, rate the contribution and logic of "
            "the above Actions to solve the Instruction. Output a scalar score."
        )
        encoded_i = tokenizer(
            [prompt_i],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_new_tokens,
        )["input_ids"].to(device).long()  # [1, T]
        segment_blocks.append(encoded_i)
    return torch.stack(segment_blocks, dim=1)  # [1, N, T]

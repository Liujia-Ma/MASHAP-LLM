from __future__ import annotations

from typing import Iterable, Sequence

import torch

from marft.reward import MaskedCoalitionShapleyAllocator, RealCoalitionShapleyAllocator

DEFAULT_ABSENCE_MESSAGE_TEMPLATE = "System: The {role} did not participate in this round."


def build_masked_coalition_allocator(all_args, mas, *, for_rollout: bool) -> MaskedCoalitionShapleyAllocator:
    return MaskedCoalitionShapleyAllocator(
        model_path=all_args.model_name_or_path,
        device=mas.qcritic_device,
        critic_lr=all_args.qcritic_lr,
        mask_token_type=all_args.mask_token_type,
        agent_roles=[p["role"] for p in mas.profiles],
        absence_message_template=DEFAULT_ABSENCE_MESSAGE_TEMPLATE,
        clip_value=all_args.clip_value,
    )


def build_real_coalition_allocator(all_args) -> RealCoalitionShapleyAllocator:
    return RealCoalitionShapleyAllocator(
        clip_value=all_args.clip_value,
    )


def build_qcritic_joint_tokens(
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

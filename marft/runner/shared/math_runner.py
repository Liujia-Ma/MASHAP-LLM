import os
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from marft.mas import MAS
from marft.envs.math import math_verify
from .qcritic_mode_utils import (
    DEFAULT_ABSENCE_MESSAGE_TEMPLATE,
    build_qcritic_joint_tokens,
    build_masked_coalition_allocator,
    build_rollout_qcritic_allocator,
    build_real_coalition_allocator,
    load_qcritic_checkpoint,
    register_qcritic_allocator_on_mas,
    save_qcritic_value_head,
)

class MathRunner:
    """Runner class to perform training, evaluation. and data collection. See parent class for details."""

    def __init__(self, config):
        self.num_agents = config["num_agents"]
        self.all_args = config["all_args"]
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval
        self.eval_interval = self.all_args.eval_interval
        self.algo = self.all_args.algorithm_name

        # 所有的step都要加上resume_steps偏移，以保证日志和模型保存的step数是连续的
        self.resume_steps = int(getattr(self.all_args, "resume_steps", 0) or 0)
        self.resume_training_updates = self.resume_steps // max(1, self.episode_length * self.n_rollout_threads)
        
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]

        self.mas = MAS(
            model_path=self.all_args.model_name_or_path, 
            context_window=self.all_args.context_window,
            max_new_tokens=self.all_args.max_new_tokens, 
            num_agents=self.num_agents,
            profile_path=self.all_args.profile_path,
            algo=self.algo,
            normalization_mode=self.all_args.normalization_mode,
            load_path=self.all_args.load_path,
        )

        if self.algo == "APPO":
            from marft.algorithms import APPOTrainer
            from marft.buffers.action_level_buffer import ActionBuffer
            self.trainer = APPOTrainer(self.all_args, self.mas)
            self.buffer = ActionBuffer(self.all_args, self.num_agents)
        elif self.algo == "TPPO":
            from marft.algorithms import TPPOTrainer
            from marft.buffers.token_level_buffer import TokenBuffer
            self.trainer = TPPOTrainer(self.all_args, self.mas)
            self.buffer = TokenBuffer(self.all_args, self.num_agents, self.mas.tokenizer.pad_token_id)
        else:
            raise NotImplementedError
        
        if self.all_args.reward_allocation == "real_coalition":
            self.qcritic_allocator = build_rollout_qcritic_allocator(self.all_args, self.mas)
            register_qcritic_allocator_on_mas(self.mas, self.qcritic_allocator)
            self.real_coalition_allocator = build_real_coalition_allocator(self.all_args)
            for env in self.envs.envs:
                env.set_qcritic_rollout_allocator_fn(self._allocate_qcritic_rollout_rewards)
        elif self.all_args.reward_allocation == "masked_coalition":
            self.qcritic_allocator = build_masked_coalition_allocator(self.all_args, self.mas)
            register_qcritic_allocator_on_mas(self.mas, self.qcritic_allocator)
            for env in self.envs.envs:
                env.set_qcritic_masked_allocator_fn(self._allocate_qcritic_masked_rewards)

        self.run_dir = config["run_dir"]
        self._make_log_dir()
        self.writter = SummaryWriter(self.log_dir)


    def run(self):
        training_steps = self.resume_training_updates
        next_obs = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = next_obs.copy()

        steps_per_episode = self.episode_length * self.n_rollout_threads
        remaining_steps = max(0, int(self.num_env_steps) - int(self.resume_steps))
        episodes = remaining_steps // max(1, steps_per_episode)

        progress_bar = tqdm(total=episodes, desc=f"Start running...", position=0, leave=True)

        for episode in range(episodes):

            # if eval
            if self.all_args.use_eval and episode % self.all_args.eval_interval == 0:
                torch.cuda.empty_cache()
                self.eval(training_steps)

            total_num_steps = self.resume_steps + (episode + 1) * self.episode_length * self.n_rollout_threads
            episode_global_scores = []
            qcritic_losses = []
            qcritic_grad_norms = []
            for step in range(self.episode_length):
                torch.cuda.empty_cache()
                rollout_obs, actions, action_tokens, values, log_probs = self.mas.infer_for_rollout(self.buffer.obs[self.buffer.cur_batch_index, step])
                next_obs, rewards, dones, infos = self.envs.step(actions)

                # insert data into buffer
                data = next_obs, rollout_obs, rewards, dones, values, actions, action_tokens, log_probs
                self.insert(data)

                for i in range(self.n_rollout_threads):
                    global_step = total_num_steps + step * self.n_rollout_threads + i
                    episode_global_scores.append(float(infos[i].get("total_score", infos[i].get("episodic_return", 0.0))))
                    
                    # Log Q-critic stats if applicable. For baseline reward allocation, these stats are not relevant and thus skipped.
                    if self.all_args.reward_allocation != "baseline":
                        qcritic_loss = infos[i].get("qcritic_loss", None)
                        qcritic_grad_norm = infos[i].get("qcritic_grad_norm", None)
                        if qcritic_loss is not None:
                            qcritic_losses.append(float(qcritic_loss))
                        if qcritic_grad_norm is not None:
                            qcritic_grad_norms.append(float(qcritic_grad_norm))
                    if dones[i, 0]:
                        episodic_return = infos[i]['episodic_return']
                        self.writter.add_scalar("episodic return", episodic_return, global_step)

            self.before_update()
            train_infos = self.trainer.train(self.buffer, total_num_steps)
            training_steps += 1

            self.buffer.after_update()

            # post process
            # save model
            if (episode == episodes - 1) or ((episode + 1) % self.all_args.save_interval == 0):
                self.save(total_num_steps)

            # log info
            if episode % self.log_interval == 0:
                if self.all_args.reward_allocation == "baseline":
                    avg_step_reward = np.mean(self.buffer.rewards[self.buffer.pre_batch_index, :, :, -1])
                else:
                    avg_step_reward = float(np.mean(episode_global_scores)) if len(episode_global_scores) > 0 else 0.0
                # Log per-agent reward, advantage for masked or rollout Q-critic allocation.
                # For terminal allocation, these per-agent stats are not meaningful and thus skipped.
                per_agent_reward = None
                if self.all_args.reward_allocation != "baseline":
                    per_agent_reward = np.mean(self.buffer.rewards[self.buffer.pre_batch_index], axis=(0, 1))
                if hasattr(self.buffer, "action_level_advantages"):
                    per_agent_advantage = np.mean(self.buffer.action_level_advantages[self.buffer.pre_batch_index], axis=(0, 1))
                elif hasattr(self.buffer, "tppo_advantages"):
                    per_agent_advantage = np.mean(self.buffer.tppo_advantages[self.buffer.pre_batch_index], axis=(0, 1, 3))
                else:
                    per_agent_advantage = None
                progress_bar.set_description(
                    f"Episode {episode}/{episodes}"
                    f"(total step num: {total_num_steps} | average step reward: {avg_step_reward:.4f})",
                )
                train_infos["average_step_rewards"] = avg_step_reward
                
                # Log per-agent rewards and advantages for each mode if applicable.
                if per_agent_reward is not None:
                    for agent_id, agent_reward in enumerate(per_agent_reward):
                        train_infos[f"reward/agent_{agent_id}"] = float(agent_reward)
                if per_agent_advantage is not None:
                    for agent_id, agent_adv in enumerate(per_agent_advantage):
                        train_infos[f"advantage/agent_{agent_id}"] = float(agent_adv)
                
                # Log Q-critic stats if applicable.
                if self.all_args.reward_allocation != "baseline":
                    if len(qcritic_losses) > 0:
                        train_infos["qcritic/loss"] = float(np.mean(qcritic_losses))
                    if len(qcritic_grad_norms) > 0:
                        train_infos["qcritic/grad_norm"] = float(np.mean(qcritic_grad_norms))
                self.log_train(train_infos, total_num_steps)
                self.writter.add_scalar('average reward', avg_step_reward, training_steps)
            progress_bar.update(1)

    def insert(self, data):
        next_obs, rollout_obs, rewards, dones, values, actions, action_tokens, log_probs = data
        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents), dtype=np.float32)
        self.buffer.insert(next_obs, actions, rollout_obs, values, rewards, masks, action_tokens, log_probs)

    def _allocate_qcritic_masked_rewards(
        self,
        actions: np.ndarray,
        global_score: float,
        original_problem: str,
        agent_states: str,
    ):
        """
        Build contextual prompt input for Q-critic, then allocate rewards.
        """
        joint_tokens = build_qcritic_joint_tokens(
            tokenizer=self.qcritic_allocator.tokenizer,
            device=self.mas.qcritic_device,
            profiles=self.mas.profiles,
            actions=actions,
            coalition_indices=tuple(range(self.num_agents)),
            original_problem=original_problem,
            agent_states=None,
            include_state=False,
            max_new_tokens=self.all_args.max_new_tokens,
        )
        self.qcritic_allocator.update_critic(
            joint_tokens,
            torch.tensor([global_score], device=self.mas.qcritic_device),
        )
        rewards = self.qcritic_allocator.compute_shapley_values(joint_tokens)
        target = torch.tensor([global_score], device=self.mas.qcritic_device, dtype=torch.float32)
        sums = rewards.sum(dim=1, keepdim=True)
        safe = torch.where(torch.abs(sums) < 1e-8, torch.ones_like(sums), sums)
        rewards = rewards * (target.view(-1, 1) / safe)
        stats = getattr(self.qcritic_allocator, "last_update_stats", {})
        return rewards[0].detach().float().cpu().tolist(), stats

    def _get_rollout_full_coalition_indices(self, actions: np.ndarray) -> tuple[int, ...]:
        """
        In rollout mode, use the answer-capable currently participating agents
        as the full coalition for qcritic training input.
        """
        coalition = tuple(
            i for i, profile in enumerate(self.mas.profiles)
            if profile.get("with_answer", False) and str(actions[i]).strip() != ""
        )
        return coalition if len(coalition) > 0 else tuple(range(self.num_agents))

    def _allocate_qcritic_rollout_rewards(
        self,
        actions: np.ndarray,
        base_state: str,
        global_score: float,
        original_problem: str,
        gt: str,
    ):
        all_agents = tuple(range(self.num_agents))
        coalition_actions_cache: dict[tuple[int, ...], np.ndarray] = {
            all_agents: np.array(actions, dtype=np.object_),
        }

        def rollout_fn(coalition_indices: tuple[int, ...]):
            coalition_indices = tuple(sorted(coalition_indices))
            if coalition_indices in coalition_actions_cache:
                return coalition_actions_cache[coalition_indices]
            return self.mas.get_actions_sequential_counterfactual(
                np.array([[base_state for _ in range(self.num_agents)]], dtype=np.object_),
                coalition_indices=coalition_indices,
                absence_message_template=DEFAULT_ABSENCE_MESSAGE_TEMPLATE,
            )[0]

        def coalition_value_fn(effective_actions: np.ndarray, coalition_indices: tuple[int, ...]) -> float:
            joint_tokens = build_qcritic_joint_tokens(
                tokenizer=self.qcritic_allocator.tokenizer,
                device=self.mas.qcritic_device,
                profiles=self.mas.profiles,
                actions=effective_actions,
                coalition_indices=coalition_indices,
                original_problem=original_problem,
                agent_states=None,
                include_state=False,
                max_new_tokens=self.all_args.max_new_tokens,
            )
            value = self.qcritic_allocator.estimate_coalition_value(joint_tokens, coalition_indices)
            return float(value[0].detach().item())

        def coalition_has_answer_fn(coalition_indices: tuple[int, ...]) -> bool:
            coalition_set = set(coalition_indices)
            return any(i in coalition_set and self.mas.profiles[i]["with_answer"] for i in range(self.num_agents))

        def coalition_env_score(effective_actions: np.ndarray, coalition_indices: tuple[int, ...]) -> float:
            if not coalition_has_answer_fn(coalition_indices):
                return 0.0
            answers = [
                effective_actions[i]
                for i in coalition_indices
                if self.mas.profiles[i].get("with_answer", False)
            ]
            if len(answers) == 0:
                return 0.0
            score = 0.0
            for ans in answers:
                score += float(math_verify.compute_score(ans, gt))
            return float(score / len(answers))

        # Train qcritic on all coalition rollouts to reduce train/infer distribution shift.
        # Include empty/no-answer coalitions with target=0.0 as requested.
        coalition_losses = []
        for mask in range(1 << self.num_agents):
            coalition_indices = tuple(i for i in range(self.num_agents) if (mask >> i) & 1)
            if len(coalition_indices) == 0:
                effective_actions = np.array(
                    [DEFAULT_ABSENCE_MESSAGE_TEMPLATE.format(role=self.mas.profiles[i]["role"]) for i in range(self.num_agents)],
                    dtype=np.object_,
                )
                coalition_target = 0.0
            else:
                effective_actions = rollout_fn(coalition_indices)
                coalition_actions_cache[tuple(sorted(coalition_indices))] = np.array(effective_actions, dtype=np.object_)
                coalition_target = coalition_env_score(effective_actions, coalition_indices)

            train_tokens = build_qcritic_joint_tokens(
                tokenizer=self.qcritic_allocator.tokenizer,
                device=self.mas.qcritic_device,
                profiles=self.mas.profiles,
                actions=effective_actions,
                coalition_indices=coalition_indices,
                original_problem=original_problem,
                agent_states=None,
                include_state=False,
                max_new_tokens=self.all_args.max_new_tokens,
            )
            loss_item = self.qcritic_allocator.update_critic(
                train_tokens,
                torch.tensor([coalition_target], device=self.mas.qcritic_device),
            )
            coalition_losses.append(float(loss_item))

        rewards, debug = self.real_coalition_allocator.allocate(
            actions=actions,
            rollout_fn=rollout_fn,
            coalition_value_fn=coalition_value_fn,
            coalition_has_answer_fn=coalition_has_answer_fn,
            total_score_precomputed=float(global_score),
        )
        # Surface qcritic training stats to env infos so runner logging can record them
        qcritic_stats = getattr(self.qcritic_allocator, "last_update_stats", {})
        if isinstance(qcritic_stats, dict):
            if qcritic_stats.get("loss", None) is not None:
                debug["qcritic_loss"] = float(qcritic_stats["loss"])
            if qcritic_stats.get("grad_norm", None) is not None:
                debug["qcritic_grad_norm"] = float(qcritic_stats["grad_norm"])
        if len(coalition_losses) > 0:
            debug["qcritic_rollout_train_updates"] = int(len(coalition_losses))
            debug["qcritic_rollout_train_loss_mean"] = float(np.mean(coalition_losses))
        debug["counterfactual_mode"] = "rollout"
        debug["absence_message_template"] = DEFAULT_ABSENCE_MESSAGE_TEMPLATE
        return rewards.tolist(), debug

    @torch.no_grad()
    def before_update(self):
        """Calculate returns for the collected data."""
        values = self.mas.get_next_values(self.buffer.obs[self.buffer.cur_batch_index, -1])
        self.buffer.compute_gae_and_returns(values)

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, training_steps):
        print(f"start evaluating......")
        eval_episode = 0
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()
        _, eval_actions, _, _, _ = self.mas.infer_for_rollout(eval_obs, evaluating=True)
        eval_next_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
        eval_episode_rewards = eval_rewards[:, -1]
        eval_env_infos = {"eval_episode_rewards": eval_episode_rewards}
        print("eval reward is {}.".format(np.mean(eval_episode_rewards)))
        self.log_eval(eval_env_infos, training_steps)

        # eval_dones_env = np.all(eval_dones, axis=1)

        # for eval_i in range(self.n_eval_rollout_threads):
        #     if eval_dones_env[eval_i]:
        #         eval_episode += 1
        #         eval_episode_rewards.append(eval_rewards[eval_i])

        # if eval_episode >= self.all_args.eval_episodes:
        #     eval_episode_rewards = np.array(eval_episode_rewards)
        #     eval_env_infos = {"eval_average_episode_rewards": eval_episode_rewards}
        #     print("total_num_steps: ", total_num_steps)
        #     print("eval reward is {}.".format(np.mean(eval_episode_rewards)))
        #     self.log_eval(eval_env_infos, total_num_steps)
        #     break

    def _make_log_dir(self):
        self.log_dir = str(self.run_dir / "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.save_dir = str(self.run_dir / "checkpoints/")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def log_eval(self, eval_infos, total_num_steps):
        for k, v in eval_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def save(self, steps):
        """Save the MAS policies, critic networks, and qcritic value-head (if present)."""
        self.mas.save(self.save_dir, steps)
        # Trainer.save_optimizers will include qcritic optimizer state under optimizers.pt when applicable
        self.trainer.save_optimizers(self.save_dir, steps)

        # Save qcritic value_head in standalone file.
        if hasattr(self, "qcritic_allocator") and self.qcritic_allocator is not None:
            try:
                exp_path = os.path.join(self.save_dir, "steps_{:04d}".format(steps))
                qcritic_vh_path = save_qcritic_value_head(self.qcritic_allocator, exp_path)
                print(f"[MathRunner] qcritic checkpoints saved -> {exp_path}")
            except Exception as e:
                print(f"[MathRunner] warning: failed to save qcritic state: {e}")

    def restore(self, model_dir):
        """Restore policy's networks from a saved model and load qcritic if present."""
        # try MAS restore (if implemented)
        try:
            self.mas.restore(model_dir)
        except Exception:
            pass

        # Attempt to load qcritic value head and optimizer if qcritic allocator exists
        if hasattr(self, "qcritic_allocator") and self.qcritic_allocator is not None:
            checkpoint_dir = model_dir
            # If model_dir is run root, locate the first steps_* dir with qcritic checkpoint.
            if os.path.isdir(model_dir) and not os.path.exists(os.path.join(model_dir, "qcritic_value_head.pth")):
                for entry in os.listdir(model_dir):
                    entry_path = os.path.join(model_dir, entry)
                    if os.path.isdir(entry_path) and entry.startswith("steps_") and os.path.exists(os.path.join(entry_path, "qcritic_value_head.pth")):
                        checkpoint_dir = entry_path
                        break

            try:
                loaded_value_head, loaded_optimizer = load_qcritic_checkpoint(
                    self.qcritic_allocator,
                    checkpoint_dir,
                    map_location="cpu",
                )
                if loaded_value_head:
                    print(f"[MathRunner] Loaded qcritic value head from {checkpoint_dir}")
                if loaded_optimizer:
                    print(f"[MathRunner] Loaded qcritic optimizer from {checkpoint_dir}/optimizers.pt")
                if not loaded_value_head:
                    print(f"[MathRunner] no qcritic checkpoint found at {checkpoint_dir}")
                else:
                    pass
            except Exception as e:
                print(f"[MathRunner] warning: failed to load qcritic state: {e}")

import os
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from marft.mas import MAS
from marft.envs.math import math_verify
from .llmshap_mode_utils import (
    DEFAULT_ABSENCE_MESSAGE_TEMPLATE,
    build_llmshap_joint_tokens,
    build_llmshap_estimator,
    build_pureshap_allocator,
    build_llmshap_allocator,
    load_llmshap_checkpoint,
    register_llmshap_estimator_on_mas,
    save_llmshap_value_head,
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
            experiment_mode=self.all_args.experiment_mode,
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
        
        if self.all_args.experiment_mode == "llmshap":
            self.llmshap_estimator = build_llmshap_estimator(self.all_args, self.mas)
            register_llmshap_estimator_on_mas(self.mas, self.llmshap_estimator)
            self.llmshap_allocator = build_llmshap_allocator()
            for env in self.envs.envs:
                env.set_llmshap_rollout_allocator_fn(self._allocate_llmshap_rollout_rewards)
        elif self.all_args.experiment_mode == "pureshap":
            self.pureshap_allocator = build_pureshap_allocator()
            for env in self.envs.envs:
                env.set_llmshap_rollout_allocator_fn(self._allocate_pureshap_rewards)

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
            llmshap_stats_enabled = self.all_args.experiment_mode == "llmshap"

            # if eval
            if self.all_args.use_eval and episode % self.all_args.eval_interval == 0:
                torch.cuda.empty_cache()
                self.eval(training_steps)

            total_num_steps = self.resume_steps + (episode + 1) * self.episode_length * self.n_rollout_threads
            episode_global_scores = []
            llmshap_losses = []
            llmshap_grad_norms = []
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
                    
                    # Log llmshap stats only in llmshap mode.
                    if llmshap_stats_enabled:
                        llmshap_loss = infos[i].get("llmshap_loss", None)
                        llmshap_grad_norm = infos[i].get("llmshap_grad_norm", None)
                        if llmshap_loss is not None:
                            llmshap_losses.append(float(llmshap_loss))
                        if llmshap_grad_norm is not None:
                            llmshap_grad_norms.append(float(llmshap_grad_norm))
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
                if self.all_args.experiment_mode == "baseline":
                    avg_step_reward = np.mean(self.buffer.rewards[self.buffer.pre_batch_index, :, :, -1])
                else:
                    avg_step_reward = float(np.mean(episode_global_scores)) if len(episode_global_scores) > 0 else 0.0
                # Log per-agent reward/advantage for all modes.
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
                
                # Log llmshap stats only in llmshap mode.
                if llmshap_stats_enabled:
                    if len(llmshap_losses) > 0:
                        train_infos["llmshap/loss"] = float(np.mean(llmshap_losses))
                    if len(llmshap_grad_norms) > 0:
                        train_infos["llmshap/grad_norm"] = float(np.mean(llmshap_grad_norms))
                self.log_train(train_infos, total_num_steps)
                self.writter.add_scalar('average reward', avg_step_reward, training_steps)
            progress_bar.update(1)

    def insert(self, data):
        next_obs, rollout_obs, rewards, dones, values, actions, action_tokens, log_probs = data
        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents), dtype=np.float32)
        self.buffer.insert(next_obs, actions, rollout_obs, values, rewards, masks, action_tokens, log_probs)

    def _get_rollout_full_coalition_indices(self, actions: np.ndarray) -> tuple[int, ...]:
        """
        In rollout mode, use the answer-capable currently participating agents
        as the full coalition for llmshap training input.
        """
        coalition = tuple(
            i for i, profile in enumerate(self.mas.profiles)
            if profile.get("with_answer", False) and str(actions[i]).strip() != ""
        )
        return coalition if len(coalition) > 0 else tuple(range(self.num_agents))

    def _allocate_llmshap_rollout_rewards(
        self,
        actions: np.ndarray,
        base_state: str,
        global_score: float,
        original_problem: str,
        gt: str,
    ):
        llmshap_device = self.mas.get_llmshap_device()
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
            joint_tokens = build_llmshap_joint_tokens(
                tokenizer=self.llmshap_estimator.tokenizer,
                device=llmshap_device,
                profiles=self.mas.profiles,
                actions=effective_actions,
                coalition_indices=coalition_indices,
                original_problem=original_problem,
                agent_states=None,
                include_state=False,
                max_new_tokens=self.all_args.max_new_tokens,
            )
            value = self.llmshap_estimator.estimate_coalition_value(joint_tokens, coalition_indices)
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

        # Train llmshap on all coalition rollouts to reduce train/infer distribution shift.
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

            train_tokens = build_llmshap_joint_tokens(
                tokenizer=self.llmshap_estimator.tokenizer,
                device=llmshap_device,
                profiles=self.mas.profiles,
                actions=effective_actions,
                coalition_indices=coalition_indices,
                original_problem=original_problem,
                agent_states=None,
                include_state=False,
                max_new_tokens=self.all_args.max_new_tokens,
            )
            loss_item = self.llmshap_estimator.update_critic(
                train_tokens,
                torch.tensor([coalition_target], device=llmshap_device),
            )
            coalition_losses.append(float(loss_item))

        rewards, debug = self.llmshap_allocator.allocate(
            actions=actions,
            rollout_fn=rollout_fn,
            coalition_value_fn=coalition_value_fn,
            coalition_has_answer_fn=coalition_has_answer_fn,
            total_score_precomputed=float(global_score),
        )
        # Surface llmshap training stats to env infos so runner logging can record them
        llmshap_stats = getattr(self.llmshap_estimator, "last_update_stats", {})
        if isinstance(llmshap_stats, dict):
            if llmshap_stats.get("loss", None) is not None:
                debug["llmshap_loss"] = float(llmshap_stats["loss"])
            if llmshap_stats.get("grad_norm", None) is not None:
                debug["llmshap_grad_norm"] = float(llmshap_stats["grad_norm"])
        if len(coalition_losses) > 0:
            debug["llmshap_rollout_train_updates"] = int(len(coalition_losses))
            debug["llmshap_rollout_train_loss_mean"] = float(np.mean(coalition_losses))
        debug["counterfactual_mode"] = "rollout"
        debug["absence_message_template"] = DEFAULT_ABSENCE_MESSAGE_TEMPLATE
        return rewards.tolist(), debug

    def _allocate_pureshap_rewards(
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
        coalition_score_cache: dict[tuple[int, ...], float] = {
            all_agents: float(global_score),
        }

        def coalition_has_answer_fn(coalition_indices: tuple[int, ...]) -> bool:
            coalition_set = set(coalition_indices)
            return any(
                i in coalition_set and self.mas.profiles[i].get("with_answer", False)
                for i in range(self.num_agents)
            )

        def rollout_fn(coalition_indices: tuple[int, ...]) -> np.ndarray:
            coalition_indices = tuple(sorted(coalition_indices))
            if coalition_indices in coalition_actions_cache:
                return coalition_actions_cache[coalition_indices]
            effective_actions = self.mas.get_actions_sequential_counterfactual(
                np.array([[base_state for _ in range(self.num_agents)]], dtype=np.object_),
                coalition_indices=coalition_indices,
                absence_message_template=DEFAULT_ABSENCE_MESSAGE_TEMPLATE,
            )[0]
            coalition_actions_cache[coalition_indices] = np.array(effective_actions, dtype=np.object_)
            return coalition_actions_cache[coalition_indices]

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

        def coalition_score_fn(coalition_indices: tuple[int, ...]) -> float:
            coalition_indices = tuple(sorted(coalition_indices))
            if coalition_indices in coalition_score_cache:
                return float(coalition_score_cache[coalition_indices])
            if len(coalition_indices) == 0 or not coalition_has_answer_fn(coalition_indices):
                coalition_score_cache[coalition_indices] = 0.0
                return 0.0
            effective_actions = rollout_fn(coalition_indices)
            coalition_score_cache[coalition_indices] = float(
                coalition_env_score(effective_actions, coalition_indices)
            )
            return float(coalition_score_cache[coalition_indices])

        rewards, debug = self.pureshap_allocator.allocate(
            num_agents=self.num_agents,
            coalition_score_fn=coalition_score_fn,
            coalition_has_actor_fn=coalition_has_answer_fn,
            full_coalition_score=float(global_score),
        )
        debug["counterfactual_mode"] = "rollout"
        debug["absence_message_template"] = DEFAULT_ABSENCE_MESSAGE_TEMPLATE
        debug["full_coalition_cached_score"] = float(global_score)
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
        """Save the MAS policies, critic networks, and llmshap value-head (if present)."""
        self.mas.save(self.save_dir, steps)
        # Trainer.save_optimizers will include llmshap optimizer state under optimizers.pt when applicable
        self.trainer.save_optimizers(self.save_dir, steps)

        # Save llmshap value_head in standalone file.
        if hasattr(self, "llmshap_estimator") and self.llmshap_estimator is not None:
            try:
                exp_path = os.path.join(self.save_dir, "steps_{:04d}".format(steps))
                llmshap_vh_path = save_llmshap_value_head(self.llmshap_estimator, exp_path)
                print(f"[MathRunner] llmshap checkpoints saved -> {exp_path}")
            except Exception as e:
                print(f"[MathRunner] warning: failed to save llmshap state: {e}")

    def restore(self, model_dir):
        """Restore policy's networks from a saved model and load llmshap if present."""
        # try MAS restore (if implemented)
        try:
            self.mas.restore(model_dir)
        except Exception:
            pass

        # Attempt to load llmshap value head and optimizer if llmshap allocator exists
        if hasattr(self, "llmshap_estimator") and self.llmshap_estimator is not None:
            checkpoint_dir = model_dir
            # If model_dir is run root, locate the first steps_* dir with llmshap checkpoint.
            if os.path.isdir(model_dir) and not os.path.exists(os.path.join(model_dir, "llmshap_value_head.pth")):
                for entry in os.listdir(model_dir):
                    entry_path = os.path.join(model_dir, entry)
                    if os.path.isdir(entry_path) and entry.startswith("steps_") and os.path.exists(os.path.join(entry_path, "llmshap_value_head.pth")):
                        checkpoint_dir = entry_path
                        break

            try:
                loaded_value_head, loaded_optimizer = load_llmshap_checkpoint(
                    self.llmshap_estimator,
                    checkpoint_dir,
                    map_location="cpu",
                )
                if loaded_value_head:
                    print(f"[MathRunner] Loaded llmshap value head from {checkpoint_dir}")
                if loaded_optimizer:
                    print(f"[MathRunner] Loaded llmshap optimizer from {checkpoint_dir}/optimizers.pt")
                if not loaded_value_head:
                    print(f"[MathRunner] no llmshap checkpoint found at {checkpoint_dir}")
                else:
                    pass
            except Exception as e:
                print(f"[MathRunner] warning: failed to load llmshap state: {e}")

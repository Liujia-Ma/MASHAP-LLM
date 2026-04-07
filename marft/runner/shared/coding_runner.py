import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from tensorboardX import SummaryWriter
from marft.mas import MAS
from .qcritic_mode_utils import (
    DEFAULT_ABSENCE_MESSAGE_TEMPLATE,
    build_qcritic_joint_tokens,
    build_masked_coalition_allocator,
    build_real_coalition_allocator,
)

class CodingRunner:
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
        
        if self.all_args.reward_allocation in {"qcritic_rollout", "qcritic_masked"}:
            self.qcritic_state_tokenizer = AutoTokenizer.from_pretrained(
                self.all_args.model_name_or_path,
                use_fast=False,
                padding_side="left",
            )
            if self.qcritic_state_tokenizer.pad_token is None:
                self.qcritic_state_tokenizer.pad_token = self.qcritic_state_tokenizer.eos_token

        if self.all_args.reward_allocation == "qcritic_rollout":
            self.masked_coalition_allocator = build_masked_coalition_allocator(self.all_args, self.mas, for_rollout=True)
            self.real_coalition_allocator = build_real_coalition_allocator(self.all_args)
            for env in self.envs.envs:
                env.set_qcritic_rollout_allocator_fn(self._allocate_qcritic_rollout_rewards)
        elif self.all_args.reward_allocation == "qcritic_masked":
            self.masked_coalition_allocator = build_masked_coalition_allocator(self.all_args, self.mas, for_rollout=False)
            for env in self.envs.envs:
                env.set_qcritic_masked_allocator_fn(self._allocate_qcritic_masked_rewards)

        self.run_dir = config["run_dir"]
        self._make_log_dir()
        self.writter = SummaryWriter(self.log_dir)


    def run(self):
        training_steps = self.resume_training_updates
        next_obs = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = next_obs.copy()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        progress_bar = tqdm(total=episodes, desc=f"Start running...", position=0, leave=True)

        for episode in range(episodes):

            # if eval
            if self.all_args.use_eval and episode % self.all_args.eval_interval == 0:
                torch.cuda.empty_cache()
                self.eval(training_steps)

            total_num_steps = self.resume_steps + (episode + 1) * self.episode_length * self.n_rollout_threads
            episode_global_scores = []
            sr_losses = []
            sr_grad_norms = []
            for step in range(self.episode_length):
                torch.cuda.empty_cache()
                rollout_obs, actions, action_tokens, values, log_probs = self.mas.infer_for_rollout(self.buffer.obs[self.buffer.cur_batch_index, step])
                next_obs, rewards, dones, infos = self.envs.step(actions)

                # insert data into buffer
                data = next_obs, rollout_obs, rewards, dones, values, actions, action_tokens, log_probs
                self.insert(data)

                for i in range(self.n_rollout_threads):
                    global_step = self.resume_steps + episode * self.episode_length * self.n_rollout_threads + step * self.n_rollout_threads + i
                    episode_global_scores.append(float(infos[i].get("total_score", infos[i].get("episodic_return", 0.0))))
                    if self.all_args.reward_allocation == "qcritic_masked":
                        sr_loss = infos[i].get("qcritic_loss", None)
                        sr_grad = infos[i].get("qcritic_grad_norm", None)
                        if sr_loss is not None:
                            sr_losses.append(float(sr_loss))
                        if sr_grad is not None:
                            sr_grad_norms.append(float(sr_grad))
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
                self.save(training_steps)

            # log info
            if episode % self.log_interval == 0:
                if self.all_args.reward_allocation == "terminal":
                    avg_step_reward = np.mean(self.buffer.rewards[self.buffer.pre_batch_index, :, :, -1])
                else:
                    avg_step_reward = float(np.mean(episode_global_scores)) if len(episode_global_scores) > 0 else 0.0
                # Log per-agent reward only for Shapley allocation modes.
                # For baseline terminal mode, keep logs minimal and unchanged.
                per_agent_reward = None
                if self.all_args.reward_allocation != "terminal":
                    per_agent_reward = np.mean(self.buffer.rewards[self.buffer.pre_batch_index], axis=(0, 1))
                if hasattr(self.buffer, "action_level_advantages"):
                    per_agent_advantage = np.mean(self.buffer.action_level_advantages[self.buffer.pre_batch_index], axis=(0, 1))
                elif hasattr(self.buffer, "tppo_advantages"):
                    per_agent_advantage = np.mean(self.buffer.tppo_advantages[self.buffer.pre_batch_index], axis=(0, 1, 3))
                else:
                    per_agent_advantage = None
                progress_bar.set_description(
                    f"Episode {episode}/{episodes}"
                    f"(total step num: {total_num_steps} | average step reward: {avg_step_reward})",
                )
                train_infos["average_step_rewards"] = avg_step_reward
                if per_agent_reward is not None:
                    for agent_id, agent_reward in enumerate(per_agent_reward):
                        train_infos[f"reward/agent_{agent_id}"] = float(agent_reward)
                if per_agent_advantage is not None:
                    for agent_id, agent_adv in enumerate(per_agent_advantage):
                        train_infos[f"advantage/agent_{agent_id}"] = float(agent_adv)
                if self.all_args.reward_allocation == "qcritic_masked":
                    if len(sr_losses) > 0:
                        train_infos["qcritic/loss"] = float(np.mean(sr_losses))
                    if len(sr_grad_norms) > 0:
                        train_infos["qcritic/grad_norm"] = float(np.mean(sr_grad_norms))
                self.log_train(train_infos, total_num_steps)
                self.writter.add_scalar('average_reward', avg_step_reward, training_steps)
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
        state_tokens = self.qcritic_state_tokenizer(
            [agent_states],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.all_args.max_new_tokens * self.num_agents,
        )["input_ids"].to(self.mas.qcritic_device).long()  # [1, T_state]
        state_only_joint_tokens = state_tokens.unsqueeze(1)  # [1, 1, T_state]
        self.masked_coalition_allocator.update_critic(state_only_joint_tokens, torch.tensor([global_score], device=self.mas.qcritic_device))
        joint_tokens = build_qcritic_joint_tokens(
            tokenizer=self.masked_coalition_allocator.tokenizer,
            device=self.mas.qcritic_device,
            profiles=self.mas.profiles,
            actions=actions,
            coalition_indices=tuple(range(self.num_agents)),
            original_problem=original_problem,
            agent_states=None,
            include_state=False,
            max_new_tokens=self.all_args.max_new_tokens,
        )
        rewards = self.masked_coalition_allocator.compute_shapley_values(joint_tokens)
        target = torch.tensor([global_score], device=self.mas.qcritic_device, dtype=torch.float32)
        sums = rewards.sum(dim=1, keepdim=True)
        safe = torch.where(torch.abs(sums) < 1e-8, torch.ones_like(sums), sums)
        rewards = rewards * (target.view(-1, 1) / safe)
        stats = getattr(self.masked_coalition_allocator, "last_update_stats", {})
        return rewards[0].detach().float().cpu().tolist(), stats

    def _allocate_qcritic_rollout_rewards(
        self,
        actions: np.ndarray,
        base_state: str,
        global_score: float,
        original_problem: str,
    ):
        state_tokens = self.qcritic_state_tokenizer(
            [base_state],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.all_args.max_new_tokens * self.num_agents,
        )["input_ids"].to(self.mas.qcritic_device).long()  # [1, T_state]
        state_only_joint_tokens = state_tokens.unsqueeze(1)  # [1, 1, T_state]
        self.masked_coalition_allocator.update_critic(state_only_joint_tokens, torch.tensor([global_score], device=self.mas.qcritic_device))

        def rollout_fn(coalition_indices: tuple[int, ...]):
            return self.mas.get_actions_sequential_counterfactual(
                np.array([[base_state for _ in range(self.num_agents)]], dtype=np.object_),
                coalition_indices=coalition_indices,
                absence_message_template=DEFAULT_ABSENCE_MESSAGE_TEMPLATE,
            )[0]

        def coalition_value_fn(effective_actions: np.ndarray, coalition_indices: tuple[int, ...]) -> float:
            joint_tokens = build_qcritic_joint_tokens(
                tokenizer=self.masked_coalition_allocator.tokenizer,
                device=self.mas.qcritic_device,
                profiles=self.mas.profiles,
                actions=effective_actions,
                coalition_indices=coalition_indices,
                original_problem=original_problem,
                agent_states=None,
                include_state=False,
                max_new_tokens=self.all_args.max_new_tokens,
            )
            value = self.masked_coalition_allocator.estimate_coalition_value(joint_tokens, coalition_indices)
            return float(value[0].detach().item())

        def coalition_has_answer_fn(coalition_indices: tuple[int, ...]) -> bool:
            coalition_set = set(coalition_indices)
            return any(i in coalition_set and self.mas.profiles[i]["with_answer"] for i in range(self.num_agents))

        rewards, debug = self.real_coalition_allocator.allocate(
            actions=actions,
            rollout_fn=rollout_fn,
            coalition_value_fn=coalition_value_fn,
            coalition_has_answer_fn=coalition_has_answer_fn,
            total_score_precomputed=float(global_score),
        )
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
        eval_obs = self.eval_envs.reset()
        eval_env_infos = {}
        _, eval_actions, _, _, _ = self.mas.infer_for_rollout(eval_obs, evaluating=True)
        eval_next_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
        effective_eval_rewards = []
        for i in range(self.num_agents):
            eval_env_infos[f"eval_rewards/{self.mas.profiles[i]['role']}"] = eval_rewards[:, i]
            if self.mas.profiles[i]['with_answer']:
                effective_eval_rewards.extend(eval_rewards[:, i])
        eval_env_infos["eval_rewards/effective"] = effective_eval_rewards
        print(f"eval rewards: {np.mean(effective_eval_rewards)}")
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

    def log_eval(self, eval_infos, training_steps):
        for k, v in eval_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, training_steps)

    def save(self, steps):
        """Save the MAS policies and critic networks."""
        self.mas.save(self.save_dir, steps)
        self.trainer.save_optimizers(self.save_dir, steps)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.mas.restore(model_dir)

"""
MAPPO implementation for DroneSurroundEnv.
- 更强的 Actor / Critic
- Actor 输出经过 tanh 限制到 [-5,5]
- 修正 GAE advantage 计算方式
- 在 train() 中加入 4 阶段 Curriculum 的「滑动窗口」升级/回退逻辑
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


__all__ = [
    'MultiAgentEnvWrapper', 'SharedActor', 'CentralCritic', 'MAPPOConfig', 'MAPPO'
]


class MultiAgentEnvWrapper:
    """Wrap joint env into per-agent observations (39 -> 3x13) and actions (6 -> 3x2)."""
    def __init__(self, env, n_agents: int = 3, obs_per_agent: int = 13, act_per_agent: int = 2):
        self.env = env
        self.n_agents = n_agents
        self.obs_per_agent = obs_per_agent
        self.act_per_agent = act_per_agent

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs_flat, info = self.env.reset()
        return self._split_obs(obs_flat), info

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        if isinstance(actions, list):
            actions = np.array(actions, dtype=np.float32)
        actions = actions.reshape(self.n_agents * self.act_per_agent)
        obs_flat, reward, terminated, truncated, info = self.env.step(actions)
        obs_agents = self._split_obs(obs_flat)
        rewards = np.full((self.n_agents,), reward, dtype=np.float32)
        return obs_agents, rewards, terminated, truncated, info

    def _split_obs(self, obs_flat: np.ndarray) -> np.ndarray:
        obs = obs_flat.reshape(self.n_agents, self.obs_per_agent).astype(np.float32)

        # 观测归一化
        obs[:, 0:2] /= 200.0        # 位置
        obs[:, 2:4] /= 10.0         # 速度
        obs[:, 4:6] /= 200.0        # 相对目标
        obs[:, 6:7] /= 200.0        # 距离
        obs[:, 7:11] /= 200.0       # 相对队友
        obs[:, 11:13] /= 200.0      # 到边界距离

        return obs

    def close(self):
        self.env.close()


class SharedActor(nn.Module):
    """共享策略网络，输出动作均值 (mu) 和 log_std；mu 经过 tanh 限幅。"""
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), max_action: float = 5.0):
        super().__init__()
        self.max_action = max_action
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = self.body(obs)
        mu_raw = self.mu_head(x)
        mu = torch.tanh(mu_raw) * self.max_action
        log_std = self.log_std.clamp(-2.0, 1.0).expand_as(mu)
        return mu, log_std


class CentralCritic(nn.Module):
    """集中式价值函数网络（更深一点）"""
    def __init__(self, global_obs_dim: int, hidden=(256, 256, 256)):
        super().__init__()
        layers = []
        last = global_obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    n_steps: int = 1024
    n_epochs: int = 10
    device: str = 'cpu'


class MAPPO:
    def __init__(self, env_wrapper: MultiAgentEnvWrapper, config: MAPPOConfig = MAPPOConfig()):
        self.env = env_wrapper
        self.cfg = config
        self.device = torch.device(self.cfg.device)
        self.n_agents = self.env.n_agents
        self.obs_dim = self.env.obs_per_agent
        self.act_dim = self.env.act_per_agent
        self.global_obs_dim = self.n_agents * self.obs_dim
        self.max_action = 5.0

        self.actor = SharedActor(self.obs_dim, self.act_dim, hidden=(256, 256), max_action=self.max_action).to(self.device)
        self.critic = CentralCritic(self.global_obs_dim, hidden=(256, 256, 256)).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.best_mean_reward = -float('inf')

        # Curriculum 滑动窗口历史（记录最近几次 eval 的成功率）
        self.curriculum_eval_history: List[float] = []
        self.curriculum_eval_stage: Optional[int] = None  # 当前记录对应的 stage

        # ===== per-stage best model tracking =====
        # key: stage (int or 'global'), value: best mean reward
        self.stage_best_mean: Dict[Any, float] = {}
        # where to save stage-best models by default
        self.stage_best_dir = "./models"

    def select_action(self, obs_agents: np.ndarray):
        obs_t = torch.as_tensor(obs_agents, dtype=torch.float32, device=self.device)
        mu, log_std = self.actor(obs_t)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        action = torch.clamp(action, -self.max_action, self.max_action)
        return (
            action.detach().cpu().numpy(),
            log_prob.detach().cpu().numpy(),
            mu.detach().cpu().numpy(),
            log_std.detach().cpu().numpy()
        )

    def evaluate_actions(self, obs_agents_t: torch.Tensor, actions_t: torch.Tensor):
        mu, log_std = self.actor(obs_agents_t)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        log_probs = dist.log_prob(actions_t).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy, mu, log_std

    def compute_advantages(self, rewards, values, dones, last_value):
        """
        标准 GAE：
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t = delta_t + gamma * lambda * A_{t+1}
        """
        T = rewards.shape[0]
        adv = np.zeros((T,), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            next_value = last_value if t == T - 1 else values[t + 1]
            r_t = float(rewards[t][0])  # 所有 agent reward 一样，取一个即可
            delta = r_t + self.cfg.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    def _update_curriculum(self, train_base_env, eval_base_env, success_rate: float):
        """
        使用「过去 3 次 eval 成功率的滑动窗口」来决定升/降 stage。
        - 只有当连续 3 次 eval 都在同一个 stage 时才做判断
        - 平均成功率 > 60% 且 stage < 6 → advance
        - 平均成功率 < 20% 且 stage > 0 → degrade
        """
        if train_base_env is None or not hasattr(train_base_env, 'get_curriculum_stage'):
            return

        current_stage = train_base_env.get_curriculum_stage()

        # 如果 stage 改变了，重置历史
        if self.curriculum_eval_stage is None or self.curriculum_eval_stage != current_stage:
            self.curriculum_eval_stage = current_stage
            self.curriculum_eval_history = []

        # 记录当前 stage 下的成功率
        self.curriculum_eval_history.append(success_rate)
        if len(self.curriculum_eval_history) > 3:
            self.curriculum_eval_history.pop(0)

        # 不满 3 次先不调整
        if len(self.curriculum_eval_history) < 3:
            return

        avg_success = float(np.mean(self.curriculum_eval_history))

        target_stage = current_stage
        # 升级：平均成功率 > 60%，且还没到最高 stage（3）
        if avg_success > 60.0 and current_stage < 6:
            train_base_env.advance_curriculum()
            target_stage = train_base_env.get_curriculum_stage()
            print(f"[Curriculum][AVG over 3 evals={avg_success:.1f}%] ↑ Advance to Stage {target_stage}")
            # 升级后清空历史
            self.curriculum_eval_stage = target_stage
            self.curriculum_eval_history = []
        # 回退：平均成功率 < 20%，且当前 stage > 0
        elif avg_success < 20.0 and current_stage > 0:
            train_base_env.degrade_curriculum()
            target_stage = train_base_env.get_curriculum_stage()
            print(f"[Curriculum][AVG over 3 evals={avg_success:.1f}%] ↓ Degrade to Stage {target_stage}")
            self.curriculum_eval_stage = target_stage
            self.curriculum_eval_history = []

        # 同步评估环境阶段
        if eval_base_env is not None and hasattr(eval_base_env, 'set_curriculum_stage'):
            eval_base_env.set_curriculum_stage(target_stage)

    def train(self, total_timesteps: int = 200_000, eval_env: MultiAgentEnvWrapper = None,
              eval_freq: int = 10_000, n_eval_episodes: int = 5, checkpoint_freq: int = 10_000,
              save_best_path: str = "./models/mappo_best.pt", final_model_path: str = None,
              show_progress: bool = True, silent_mode: bool = True, plot_freq: int = 10,
              stage_save_dir: str = None):
        obs_agents, _ = self.env.reset()
        global_obs = obs_agents.reshape(-1)

        episode_rewards = []
        current_ep_reward = 0.0

        plot_episode_rewards = []
        plot_step_numbers = []
        plot_avg_distances = []
        plot_max_distances = []
        plot_min_distances = []
        current_ep_distances = []

        obs_buf = []
        actions_buf = []
        logprob_buf = []
        values_buf = []
        rewards_buf = []
        dones_buf = []

        timestep = 0
        last_eval_time = 0
        pbar = tqdm(
            total=total_timesteps,
            desc="Training",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            ncols=120,
            smoothing=0.1,
            leave=True,
            position=0
        ) if show_progress else None

        try:
            while timestep < total_timesteps:
                # 收集 rollout
                for step in range(self.cfg.n_steps):
                    obs_buf.append(obs_agents.copy())

                    global_obs_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    value = self.critic(global_obs_t).detach().cpu().numpy()
                    values_buf.append(float(value.squeeze()))

                    actions, log_probs, mu, log_std = self.select_action(obs_agents)
                    actions_buf.append(actions)
                    logprob_buf.append(log_probs)

                    next_obs_agents, rewards, terminated, truncated, info = self.env.step(actions)
                    done_flag = float(terminated or truncated)
                    rewards_buf.append(rewards)
                    dones_buf.append(done_flag)

                    current_ep_reward += float(rewards.mean())

                    if 'avg_distance' in info:
                        current_ep_distances.append(info['avg_distance'])

                    obs_agents = next_obs_agents
                    global_obs = obs_agents.reshape(-1)

                    timestep += 1
                    if pbar is not None:
                        pbar.update(1)

                    if terminated or truncated:
                        episode_rewards.append(current_ep_reward)
                        plot_episode_rewards.append(current_ep_reward)
                        plot_step_numbers.append(timestep)

                        if current_ep_distances:
                            avg_dist = float(np.mean(current_ep_distances))
                            max_dist = float(np.max(current_ep_distances))
                            min_dist = float(np.min(current_ep_distances))
                        else:
                            avg_dist, max_dist, min_dist = 9.0, 10.0, 8.0

                        plot_avg_distances.append(avg_dist)
                        plot_max_distances.append(max_dist)
                        plot_min_distances.append(min_dist)

                        if len(episode_rewards) % plot_freq == 0:
                            self._plot_progress(
                                plot_step_numbers, plot_episode_rewards,
                                plot_avg_distances, plot_max_distances, plot_min_distances
                            )

                        current_ep_reward = 0.0
                        current_ep_distances = []
                        obs_agents, _ = self.env.reset()
                        global_obs = obs_agents.reshape(-1)

                    # ===== 评估 + 课程调整 =====
                    if eval_env is not None and timestep % eval_freq == 0 and timestep != last_eval_time:
                        last_eval_time = timestep
                        mean_rew, success_rate, reason_counts, avg_distance = self._evaluate(eval_env, n_eval_episodes)
                        reason_str = ", ".join([f"{k}:{v}" for k, v in reason_counts.items()])

                        print(f"\n Eval @ Step {timestep}: Reward={mean_rew:.1f} | Success={success_rate:.1f}% | [{reason_str}]", end="")

                        # —— 课程学习滑动窗口逻辑 —— #
                        try:
                            train_base = getattr(self.env, 'env', None)
                            eval_base = getattr(eval_env, 'env', None)
                            self._update_curriculum(train_base, eval_base, success_rate)
                        except Exception as e:
                            print(f"\n[Curriculum] update error: {e}")

                        # ===== per-stage best-model saving =====
                        try:
                            # Determine current stage if env supports it
                            train_base_env = getattr(self.env, 'env', None)
                            if train_base_env is not None and hasattr(train_base_env, 'get_curriculum_stage'):
                                current_stage = train_base_env.get_curriculum_stage()
                            else:
                                current_stage = 'global'

                            # Use provided stage_save_dir or fallback
                            save_dir = stage_save_dir if stage_save_dir is not None else self.stage_best_dir
                            os.makedirs(save_dir, exist_ok=True)

                            stage_key = f"stage_{current_stage}"
                            prev_best = self.stage_best_mean.get(stage_key, -float('inf'))

                            # First-time reaching a stage -> treat as new best (save)
                            first_time_reached = stage_key not in self.stage_best_mean

                            if mean_rew > prev_best or first_time_reached:
                                self.stage_best_mean[stage_key] = mean_rew
                                stage_path = os.path.join(save_dir, f"mappo_best_stage_{current_stage}.pt")
                                torch.save({
                                    'actor': self.actor.state_dict(),
                                    'critic': self.critic.state_dict(),
                                    'config': self.cfg.__dict__,
                                    'stage': current_stage,
                                    'mean_reward': mean_rew,
                                    'timestep': timestep
                                }, stage_path)

                                if first_time_reached:
                                    print(f" 🎯 First time reaching Stage {current_stage}! Saving as best model for this stage: {stage_path}")
                                else:
                                    print(f" ⭐ New best for Stage {current_stage}! Saving: {stage_path}")

                            # Also keep a global best if requested
                            if save_best_path is not None:
                                if mean_rew > self.best_mean_reward:
                                    self.best_mean_reward = mean_rew
                                    print(" NEW BEST! Saving...")
                                    self.save(save_best_path)
                                else:
                                    print()
                        except Exception as e:
                            print(f"[Save] error while saving best models: {e}")

                        if pbar is not None:
                            try:
                                stage = getattr(getattr(self.env, 'env', None), 'get_curriculum_stage', lambda: None)()
                                pbar.set_postfix({
                                    'stage': stage if stage is not None else '-',
                                    'mean_rew': f"{mean_rew:.1f}"
                                })
                            except Exception:
                                pass

                    # ===== 保存检查点 =====
                    if checkpoint_freq and checkpoint_freq > 0 and timestep % checkpoint_freq == 0:
                        ckpt_path = f"./models/mappo_checkpoint_{timestep}.pt"
                        self.save(ckpt_path)

                    if timestep >= total_timesteps:
                        break

                # ===== 计算优势（GAE） =====
                global_obs_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                last_value = float(self.critic(global_obs_t).detach().cpu().numpy().squeeze())

                obs_arr = np.array(obs_buf, dtype=np.float32)
                actions_arr = np.array(actions_buf, dtype=np.float32)
                logprob_arr = np.array(logprob_buf, dtype=np.float32)
                values_arr = np.array(values_buf, dtype=np.float32)
                rewards_arr = np.array(rewards_buf, dtype=np.float32)
                dones_arr = np.array(dones_buf, dtype=np.float32)

                advantages, returns = self.compute_advantages(
                    rewards_arr, values_arr, dones_arr, last_value
                )
                advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
                returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

                T = actions_arr.shape[0]
                obs_policy = obs_arr.reshape(T * self.n_agents, self.obs_dim)
                actions_policy = actions_arr.reshape(T * self.n_agents, self.act_dim)
                old_logprob_policy = logprob_arr.reshape(T * self.n_agents)
                adv_policy = np.repeat(advantages, self.n_agents)

                obs_policy_t = torch.as_tensor(obs_policy, dtype=torch.float32, device=self.device)
                actions_policy_t = torch.as_tensor(actions_policy, dtype=torch.float32, device=self.device)
                old_logprob_policy_t = torch.as_tensor(old_logprob_policy, dtype=torch.float32, device=self.device)
                adv_policy_t = torch.as_tensor(adv_policy, dtype=torch.float32, device=self.device)
                adv_policy_t = (adv_policy_t - adv_policy_t.mean()) / (adv_policy_t.std() + 1e-8)

                global_obs_t_batch = torch.as_tensor(obs_arr.reshape(T, -1), dtype=torch.float32, device=self.device)
                returns_t_batch = returns_t
                values_old_t = torch.as_tensor(values_arr, dtype=torch.float32, device=self.device)

                # ===== PPO 更新 =====
                batch_size = self.cfg.batch_size
                idxs = np.arange(T * self.n_agents)
                for epoch in range(self.cfg.n_epochs):
                    np.random.shuffle(idxs)
                    for start in range(0, len(idxs), batch_size):
                        mb_idx = idxs[start:start + batch_size]
                        mb_obs = obs_policy_t[mb_idx]
                        mb_actions = actions_policy_t[mb_idx]
                        mb_old_logp = old_logprob_policy_t[mb_idx]
                        mb_adv = adv_policy_t[mb_idx]

                        new_logp, entropy, _, _ = self.evaluate_actions(mb_obs, mb_actions)
                        ratio = torch.exp(new_logp - mb_old_logp)
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * mb_adv
                        policy_loss = -torch.min(surr1, surr2).mean() - self.cfg.ent_coef * entropy.mean()

                        self.actor_opt.zero_grad()
                        policy_loss.backward()
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                        self.actor_opt.step()

                    values_new = self.critic(global_obs_t_batch).squeeze(-1)
                    value_loss_unclipped = (values_new - returns_t_batch).pow(2)
                    v_clipped = values_old_t + (values_new - values_old_t).clamp(-self.cfg.clip_range, self.cfg.clip_range)
                    value_loss_clipped = (v_clipped - returns_t_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean() * self.cfg.vf_coef

                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                    self.critic_opt.step()

                obs_buf.clear(); actions_buf.clear(); logprob_buf.clear(); values_buf.clear(); rewards_buf.clear(); dones_buf.clear()

        finally:
            if pbar is not None:
                pbar.close()
            if plot_step_numbers:
                self._plot_progress(
                    plot_step_numbers, plot_episode_rewards,
                    plot_avg_distances, plot_max_distances, plot_min_distances
                )

        if final_model_path:
            self.save(final_model_path)

    def _plot_progress(self, step_numbers, episode_rewards, avg_distances, max_distances, min_distances):
        if len(episode_rewards) < 2:
            return
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            step_arr = np.array(step_numbers)
            reward_arr = np.array(episode_rewards, dtype=np.float32)

            # Raw reward as thin line (low alpha)
            # raw reward: light purple (lavender) to match user's preference
            axes[0].plot(step_arr, reward_arr, color='#b19cd9', alpha=0.75, linewidth=1.5, label='Raw Reward')

            # EMA smoothing (alpha=0.1)
            alpha = 0.1
            ema_reward = np.zeros_like(reward_arr)
            ema_reward[0] = reward_arr[0]
            for i in range(1, len(reward_arr)):
                ema_reward[i] = alpha * reward_arr[i] + (1 - alpha) * ema_reward[i - 1]
            axes[0].plot(step_arr, ema_reward, color='orange', linewidth=2.5, label=f'EMA (α={alpha})')

            axes[0].set_xlabel('Training Steps')
            axes[0].set_ylabel('Episode Reward')
            axes[0].set_title('MAPPO Training Reward')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Distance plot: raw lines + EMA
            max_arr = np.array(max_distances, dtype=np.float32)
            avg_arr = np.array(avg_distances, dtype=np.float32)
            min_arr = np.array(min_distances, dtype=np.float32)

            axes[1].plot(step_arr, max_arr, color='red', alpha=0.4, linewidth=1, label='Max Distance')
            axes[1].plot(step_arr, avg_arr, color='blue', alpha=0.4, linewidth=1, label='Avg Distance')
            axes[1].plot(step_arr, min_arr, color='green', alpha=0.4, linewidth=1, label='Min Distance')

            # EMA for distances
            ema_max = np.zeros_like(max_arr)
            ema_avg = np.zeros_like(avg_arr)
            ema_min = np.zeros_like(min_arr)
            if len(max_arr) > 0:
                ema_max[0] = max_arr[0]
                ema_avg[0] = avg_arr[0]
                ema_min[0] = min_arr[0]
                for i in range(1, len(max_arr)):
                    ema_max[i] = alpha * max_arr[i] + (1 - alpha) * ema_max[i - 1]
                    ema_avg[i] = alpha * avg_arr[i] + (1 - alpha) * ema_avg[i - 1]
                    ema_min[i] = alpha * min_arr[i] + (1 - alpha) * ema_min[i - 1]

                axes[1].plot(step_arr, ema_max, color='darkred', linewidth=2.0, label='EMA Max')
                axes[1].plot(step_arr, ema_avg, color='darkblue', linewidth=2.0, label='EMA Avg')
                axes[1].plot(step_arr, ema_min, color='darkgreen', linewidth=2.0, label='EMA Min')

            axes[1].set_xlabel('Training Steps')
            axes[1].set_ylabel('Distance (m)')
            axes[1].set_title('Pursuit Distance (Max/Avg/Min)')
            axes[1].legend(loc='upper right', fontsize='small')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('training_progress.png', dpi=100)
            plt.close()
        except Exception:
            pass

    def _evaluate(self, eval_env: MultiAgentEnvWrapper, n_episodes: int):
        rewards = []
        success_count = 0
        reason_counts = {}
        distances = []

        for _ in range(n_episodes):
            obs_agents, info = eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_distances = []

            while not done:
                obs_t = torch.as_tensor(obs_agents, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    mu, _ = self.actor(obs_t)
                action = mu.cpu().numpy()
                next_obs_agents, rew_arr, terminated, truncated, info = eval_env.step(action)
                ep_reward += float(rew_arr.mean())

                if 'avg_distance' in info:
                    ep_distances.append(info['avg_distance'])

                done = terminated or truncated
                obs_agents = next_obs_agents

            reason = info.get('termination_reason', 'unknown')
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if reason == 'success':
                success_count += 1
            rewards.append(ep_reward)

            if ep_distances:
                distances.append(float(np.mean(ep_distances)))

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        success_rate = success_count / n_episodes * 100.0
        avg_distance = float(np.mean(distances)) if distances else None

        return mean_reward, success_rate, reason_counts, avg_distance

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'config': self.cfg.__dict__
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        for k, v in data['config'].items():
            setattr(self.cfg, k, v)


if __name__ == "__main__":
    try:
        from drone_surround_env import DroneSurroundEnv
        base_env = DroneSurroundEnv(curriculum=True)
        wrapped = MultiAgentEnvWrapper(base_env)
        algo = MAPPO(wrapped)
        algo.train(
            total_timesteps=5000,
            eval_env=MultiAgentEnvWrapper(DroneSurroundEnv(curriculum=True)),
            eval_freq=2000,
            n_eval_episodes=2,
            checkpoint_freq=0
        )
        algo.save("./models/mappo_quicktest.pt")
    except Exception as e:
        print("Quick test failed/skipped:", e)

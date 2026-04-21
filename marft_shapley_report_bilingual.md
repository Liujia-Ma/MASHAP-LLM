# MARFT-A、Shapley Reward 与 Critic

## 1. MARFT-A 的训练流程

### 1.1 问题重写：从 LaMAS 到顺序决策

MARFT 的核心思想是：把具有动态依赖关系的多智能体工作流重写成一个**顺序决策过程**。对于任意 joint action

$$
a^{1:n}=(a^1,a^2,\dots,a^n),
$$

其 joint advantage 可以分解为局部 advantage 之和：

$$
A^{\pi}(s,a^{1:n})=\sum_{m=1}^{n}A^{\pi}(s,a^{1:m-1},a^m).
$$

这意味着：如果第 $m$ 个智能体能看到前驱智能体的动作 $a^{1:m-1}$，那么最大化它自己的局部 advantage，就等价于推动整个系统的 joint advantage 最大化。

### 1.2 顺序执行与轨迹采样

在时刻 $t$，第 $m$ 个智能体的动作按如下方式生成：

$$
a_t^m \sim p_{\pi_m}(a_t^m\mid s_t,a_t^{1:m-1}).
$$

因此一条 rollout 轨迹为：

$$
\tau=\{(s_t,a_t^{1:n},r_t,s_{t+1})\}_{t=0}^{T-1},
$$

其中

$$
r_t=R(s_t,a_t^{1:n}).
$$

### 1.3 MARFT-A 的 PPO 目标

MARFT-A 采用 action-level 的 clipped PPO 目标：

$$
L(\theta)=\frac{1}{nT}\sum_{m=1}^{n}\sum_{t=0}^{T-1}
\min\Big[
r_t^m(\theta)\hat A_t,
\operatorname{clip}(r_t^m(\theta),1-\epsilon,1+\epsilon)\hat A_t
\Big],
$$

其中 importance ratio 为

$$
r_t^m(\theta)=
\frac{\pi_{\theta_m}^m(a_t^m\mid s_t,\hat a_t^{1:m-1})}
{\pi_{\theta_m^{\mathrm{old}}}^m(a_t^m\mid s_t,\hat a_t^{1:m-1})}.
$$

这里的直观含义是：每个智能体都在“给定前驱动作”的条件下，在 trust region 内更新自己的策略。

### 1.4 Critic 与 GAE

MARFT-A 使用一个 central critic $V_\phi$ 来估计状态价值。对一条轨迹中的第 $t$ 步，TD residual 可写为：

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
$$

然后使用 GAE 计算优势：

$$
\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
$$

也可以递推地写成：

$$
\hat A_t=\delta_t+\gamma\lambda \hat A_{t+1}.
$$

在 baseline MARFT-A 中，这个 $\hat A_t$ 通常是**系统级别**的 advantage，随后被各个智能体在各自的 PPO ratio 中共享使用。

### 1.5 每个智能体如何更新

对于第 $m$ 个智能体，它的 actor 更新目标就是最大化：

$$
L_m(\theta_m)=\frac{1}{T}\sum_{t=0}^{T-1}
\min\Big[
r_t^m(\theta_m)\hat A_t,
\operatorname{clip}(r_t^m(\theta_m),1-\epsilon,1+\epsilon)\hat A_t
\Big].
$$

因此，虽然所有智能体共享同一个系统轨迹与同一个 central critic，但在策略更新阶段，每个智能体仍然有自己独立的 PPO ratio：

$$
r_t^m(\theta_m)=\frac{\pi_{\theta_m}^m(\cdot)}{\pi_{\theta_m^{\mathrm{old}}}^m(\cdot)}.
$$

### 1.6 Critic 如何更新

用 GAE 得到 value target：

$$
\hat V_t = V_\phi(s_t)+\hat A_t.
$$

于是 critic 的训练目标是最小化平方损失：

$$
L_V(\phi)=\sum_t\left\|V_\phi(s_t)-\hat V_t\right\|^2.
$$

因此，baseline MARFT-A 的训练流程可以概括为：

1. 与环境交互，按顺序生成多智能体动作，收集轨迹；
2. 使用 central critic 计算 $V_\phi(s_t)$；
3. 用 GAE 计算 $\hat A_t$；
4. 用 $\hat V_t=V_\phi(s_t)+\hat A_t$ 更新 critic；
5. 用 clipped PPO 目标分别更新每个智能体的策略。

---

## 2. 计划中的修改：用精确 Shapley 边际贡献替代原始 Reward

### 2.1 现有问题

当前 baseline MARFT 的 reward 是系统级 reward：

$$
r_t=R(s_t,a_t^{1:n}).
$$

这会带来一个信用分配问题：

- 有些智能体并不直接输出最终答案；
- 但它们的行为会显著影响后续智能体的输入和最终结果；
- 如果 reward 只来自最终结果，那么 credit 容易集中到最后直接输出答案的 executor。

### 2.2 新设想：把 Shapley 值直接作为每个智能体的 reward

令玩家集合为固定工作流中的智能体槽位：

$$
N=\{1,2,\dots,n\}.
$$

对任意 coalition $C\subseteq N$，定义其价值函数：

$$
v(C)=U\big(\tau^{(C)}\big),
$$

其中：

- $\tau^{(C)}$ 表示只让 coalition 中的智能体正常执行、其他智能体用 baseline/no-op 替代后，**按 MARFT 的真实顺序重新 rollout** 得到的反事实轨迹；
- $U(\cdot)$ 表示对该轨迹的评分函数（例如最终正确率、pass rate、verifier score 或其他组合效用）。

然后第 $i$ 个智能体的精确 Shapley 边际贡献定义为：

$$
\phi_i(v)=\sum_{C\subseteq N\setminus\{i\}}
\frac{|C|!(n-|C|-1)!}{n!}
\big[v(C\cup\{i\})-v(C)\big].
$$

我们计划直接定义每个智能体的 reward 为：

$$
r_{i,t}^{\mathrm{shap}}=\phi_i(v_t).
$$

如果做 trajectory-level 版本，则可以简写为：

$$
r_i^{\mathrm{shap}}=\phi_i(v).
$$

### 2.3 后续仍沿用 GAE

一旦每个智能体都有自己的 reward 序列，就可以继续使用标准 GAE，但要改成**按智能体分别计算**。

对第 $i$ 个智能体：

$$
\delta_{i,t}=r_{i,t}^{\mathrm{shap}}+\gamma V_i(s_{t+1})-V_i(s_t),
$$

或如果使用 agent-conditioned critic：

$$
\delta_{i,t}=r_{i,t}^{\mathrm{shap}}+\gamma V_\phi(s_{t+1},i)-V_\phi(s_t,i).
$$

然后：

$$
\hat A_{i,t}=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{i,t+l}.
$$

价值目标为：

$$
\hat V_{i,t}=V_i(s_t)+\hat A_{i,t},
$$

或

$$
\hat V_{i,t}=V_\phi(s_t,i)+\hat A_{i,t}.
$$

然后 actor 继续用 PPO 更新：

$$
L_i(\theta_i)=\frac{1}{T}\sum_t
\min\Big[
r_t^i(\theta_i)\hat A_{i,t},
\operatorname{clip}(r_t^i(\theta_i),1-\epsilon,1+\epsilon)\hat A_{i,t}
\Big].
$$

也就是说，我的设想是：

**先用精确 Shapley 做 reward 分解，再继续沿用 PPO + GAE。**

### 2.4 这样做的理论意义

这个修改将带来两个变化：

1. reward 不再是单一全局标量，而是 per-agent 的 credit-assigned reward；
2. GAE 不再共享同一个系统 advantage，而是每个智能体有自己的 $\hat A_{i,t}$。

这意味着：

- 原始 MARFT 更像是在解决“顺序优化与稳定训练”；
- 加入 Shapley reward 后，系统开始显式解决“信用分配”问题。

---

## 3. 第三个问题：单一 critic 能否准确预测每个智能体的价值？

### 3.1 当前 baseline MARFT 的 critic

当前系统中只有一个 central critic：

$$
V_\phi(s),
$$

其中输入状态 $s$ 包括：

- 问题 $q$；
- 所有智能体的信息/profile；
- 整个求解过程中的状态与历史；
- 多智能体执行产生的上下文。

它的训练目标是：

$$
\hat V_t = V_\phi(s_t)+\hat A_t,
$$

并通过

$$
L_V(\phi)=\sum_t\left\|V_\phi(s_t)-\hat V_t\right\|^2
$$

更新。

### 3.2 关键疑问

如果 reward 已经被改成 per-agent 的 Shapley reward：

$$
r_{i,t}^{\mathrm{shap}},
$$

那么真正想预测的价值就应该是：

$$
V_i^{\pi}(s_t)=\mathbb E_\pi\left[\sum_{k\ge t}\gamma^{k-t}r_{i,k}^{\mathrm{shap}}\mid s_t\right].
$$

这时，单一 critic $V_\phi(s)$ 面临一个语义问题：

- 它到底是在预测 team-level return？
- 还是某个 agent 的 return？
- 如果不同 agent 的 reward 分布不同，一个不带 agent identity 的单一 critic 是否足够表达这些差异？

### 3.3 我的初步判断

我目前的判断是：

#### 情况 A：继续使用单一 critic $V_\phi(s)$

这在数学上未必“报错”，但它更像是在学习一个**混合的 team-level baseline**，而不是每个智能体自己的价值函数。

这会导致：

- actor 端使用的是 per-agent $\hat A_{i,t}$；
- critic 端预测的却是一个共享值；
- 二者的语义可能不完全对齐。

#### 情况 B：改成 agent-conditioned critic

更自然的写法是：

$$
V_\phi(s,i),
$$

其中 $i$ 是 agent identity / role embedding。

这样 critic 预测的是：

$$
V_\phi(s_t,i)\approx V_i^{\pi}(s_t).
$$

对应的 TD residual 变成：

$$
\delta_{i,t}=r_{i,t}^{\mathrm{shap}}+\gamma V_\phi(s_{t+1},i)-V_\phi(s_t,i).
$$

这样 reward、value、advantage 三者会更一致。

#### 情况 C：每个智能体一个独立 critic

最直接但参数开销更大的方式是：

$$
V_{\phi_i}(s_t)
$$

分别拟合每个智能体自己的累计回报。

这在小规模 DUO / TRIO 中是可行的，但扩展性较弱。

### 3.4 我目前最想向教授确认的问题

> 如果系统使用了 per-agent 的 Shapley reward，那么是否应该同步把单一 central critic 改成 agent-conditioned critic，甚至改成多 critic 结构？

因为如果不改 critic，那么虽然 actor 更新在形式上还能继续进行，但 value target 的语义可能已经和 actor 侧的 reward/advantage 不一致。

---

## 4. 我当前的阶段性结论

1. **MARFT-A 的 baseline 训练流程** 本质上是：顺序 rollout + central critic + GAE + clipped PPO；
2. **我当前加入的改动** 是：用精确 Shapley 边际贡献替代原始系统级 reward，给每个智能体分配自己的 reward；
3. 然后继续沿用 GAE 和 PPO，但 GAE 应该改为 **per-agent GAE**；
4. 这进一步引出了第三个核心问题：**单一 critic 是否还能准确建模每个智能体的价值**；
5. 我当前倾向于认为：如果 reward 已经 per-agent 化，那么 critic 至少应该升级成 **agent-conditioned critic**，否则 actor 与 critic 的目标语义会失配。

---

# English Version

## 1. Training Pipeline of MARFT-A

### 1.1 Reformulating LaMAS as Sequential Decision-Making

The key idea of MARFT is to rewrite a dynamic multi-agent workflow as a **sequential decision-making process**. For any joint action

$$
a^{1:n}=(a^1,a^2,\dots,a^n),
$$

its joint advantage can be decomposed into a sum of local advantages:

$$
A^{\pi}(s,a^{1:n})=\sum_{m=1}^{n}A^{\pi}(s,a^{1:m-1},a^m).
$$

This implies that if agent $m$ is aware of predecessor actions $a^{1:m-1}$, then maximizing its own local advantage is equivalent to improving the overall joint advantage.

### 1.2 Sequential Execution and Trajectory Collection

At time step $t$, the action of agent $m$ is generated as

$$
a_t^m \sim p_{\pi_m}(a_t^m\mid s_t,a_t^{1:m-1}).
$$

Therefore, one rollout trajectory is

$$
\tau=\{(s_t,a_t^{1:n},r_t,s_{t+1})\}_{t=0}^{T-1},
$$

where

$$
r_t=R(s_t,a_t^{1:n}).
$$

### 1.3 PPO Objective in MARFT-A

MARFT-A adopts an action-level clipped PPO objective:

$$
L(\theta)=\frac{1}{nT}\sum_{m=1}^{n}\sum_{t=0}^{T-1}
\min\Big[
r_t^m(\theta)\hat A_t,
\operatorname{clip}(r_t^m(\theta),1-\epsilon,1+\epsilon)\hat A_t
\Big],
$$

where the importance ratio is

$$
r_t^m(\theta)=
\frac{\pi_{\theta_m}^m(a_t^m\mid s_t,\hat a_t^{1:m-1})}
{\pi_{\theta_m^{\mathrm{old}}}^m(a_t^m\mid s_t,\hat a_t^{1:m-1})}.
$$

Intuitively, each agent updates its policy within a trust region conditioned on predecessor actions.

### 1.4 Critic and GAE

MARFT-A uses a central critic $V_\phi$ to estimate state values. For time step $t$, the TD residual is

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
$$

Then GAE is computed as

$$
\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
$$

Equivalently, it can be written recursively as

$$
\hat A_t=\delta_t+\gamma\lambda \hat A_{t+1}.
$$

In baseline MARFT-A, $\hat A_t$ is typically a **system-level advantage** shared across agents in the PPO update.

### 1.5 How Each Agent Updates

For agent $m$, the actor objective is

$$
L_m(\theta_m)=\frac{1}{T}\sum_{t=0}^{T-1}
\min\Big[
r_t^m(\theta_m)\hat A_t,
\operatorname{clip}(r_t^m(\theta_m),1-\epsilon,1+\epsilon)\hat A_t
\Big].
$$

Hence, while all agents share the same trajectory and central critic, each agent still has its own PPO ratio

$$
r_t^m(\theta_m)=\frac{\pi_{\theta_m}^m(\cdot)}{\pi_{\theta_m^{\mathrm{old}}}^m(\cdot)}.
$$

### 1.6 Critic Update

Using GAE, the value target is

$$
\hat V_t = V_\phi(s_t)+\hat A_t.
$$

The critic is updated by minimizing the squared loss

$$
L_V(\phi)=\sum_t\left\|V_\phi(s_t)-\hat V_t\right\|^2.
$$

Therefore, the baseline MARFT-A pipeline can be summarized as follows:

1. Interact with the environment and generate actions sequentially;
2. Estimate $V_\phi(s_t)$ using the central critic;
3. Compute $\hat A_t$ via GAE;
4. Update the critic using $\hat V_t=V_\phi(s_t)+\hat A_t$;
5. Update each agent’s actor using the clipped PPO objective.

---

## 2. Planned Modification: Replacing the Original Reward with Exact Shapley Marginal Contributions

### 2.1 Current Limitation

In baseline MARFT, the reward is a system-level scalar:

$$
r_t=R(s_t,a_t^{1:n}).
$$

This creates a credit assignment issue:

- some agents do not directly output the final answer;
- but they strongly affect downstream inputs and final performance;
- if reward comes only from final outcome, credit can collapse to the final executor.

### 2.2 New Proposal: Use Shapley Values as Per-Agent Rewards

Let the player set be the fixed slots in the workflow:

$$
N=\{1,2,\dots,n\}.
$$

For any coalition $C\subseteq N$, define its value as

$$
v(C)=U\big(\tau^{(C)}\big),
$$

where:

- $\tau^{(C)}$ is the **counterfactual trajectory** obtained by activating only the agents in $C$, replacing others with baseline/no-op actions, and rerolling the workflow in the true MARFT order;
- $U(\cdot)$ is the trajectory utility, such as final correctness, pass rate, verifier score, or a composite utility.

Then the exact Shapley marginal contribution of agent $i$ is

$$
\phi_i(v)=\sum_{C\subseteq N\setminus\{i\}}
\frac{|C|!(n-|C|-1)!}{n!}
\big[v(C\cup\{i\})-v(C)\big].
$$

We plan to directly define the reward of agent $i$ as

$$
r_{i,t}^{\mathrm{shap}}=\phi_i(v_t).
$$

For a trajectory-level formulation, this can be simplified as

$$
r_i^{\mathrm{shap}}=\phi_i(v).
$$

### 2.3 Continue Using GAE After Reward Replacement

Once each agent has its own reward stream, standard GAE can still be used, but it should now be computed **per agent**.

For agent $i$:

$$
\delta_{i,t}=r_{i,t}^{\mathrm{shap}}+\gamma V_i(s_{t+1})-V_i(s_t),
$$

or, with an agent-conditioned critic,

$$
\delta_{i,t}=r_{i,t}^{\mathrm{shap}}+\gamma V_\phi(s_{t+1},i)-V_\phi(s_t,i).
$$

Then

$$
\hat A_{i,t}=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{i,t+l}.
$$

The corresponding value target becomes

$$
\hat V_{i,t}=V_i(s_t)+\hat A_{i,t},
$$

or

$$
\hat V_{i,t}=V_\phi(s_t,i)+\hat A_{i,t}.
$$

The actor update remains PPO:

$$
L_i(\theta_i)=\frac{1}{T}\sum_t
\min\Big[
r_t^i(\theta_i)\hat A_{i,t},
\operatorname{clip}(r_t^i(\theta_i),1-\epsilon,1+\epsilon)\hat A_{i,t}
\Big].
$$

In other words, the proposal is:

**First replace the original reward with exact Shapley-based per-agent rewards, then continue training with PPO + GAE.**

### 2.4 Theoretical Meaning of This Modification

This modification changes the optimization semantics in two ways:

1. reward is no longer a single global scalar, but a per-agent credit-assigned reward;
2. GAE is no longer shared across the team, but becomes $\hat A_{i,t}$ for each individual agent.

Thus:

- original MARFT mainly solves **sequential optimization and stable training**;
- adding Shapley rewards makes the system explicitly address **credit assignment**.

---

## 3. Third Question: Can a Single Critic Accurately Predict the Value of Every Agent?

### 3.1 The Current Critic in Baseline MARFT

The current system uses a single central critic:

$$
V_\phi(s),
$$

where the state $s$ includes:

- the question $q$,
- all agent profiles/information,
- the whole solution process and interaction history,
- and the full multi-agent context.

Its target is defined as

$$
\hat V_t = V_\phi(s_t)+\hat A_t,
$$

and the critic is trained with

$$
L_V(\phi)=\sum_t\left\|V_\phi(s_t)-\hat V_t\right\|^2.
$$

### 3.2 The Core Concern

If the reward has already been replaced by per-agent Shapley rewards

$$
r_{i,t}^{\mathrm{shap}},
$$

then the value we truly want to estimate should be

$$
V_i^{\pi}(s_t)=\mathbb E_\pi\left[\sum_{k\ge t}\gamma^{k-t}r_{i,k}^{\mathrm{shap}}\mid s_t\right].
$$

Now the single critic $V_\phi(s)$ faces a semantic issue:

- Is it predicting a team-level return?
- Or the return of a specific agent?
- If reward distributions differ across agents, can a single critic without agent identity represent all of them faithfully?

### 3.3 My Current Judgment

#### Case A: Keep a single critic $V_\phi(s)$

This may still run numerically, but it behaves more like a **shared team-level baseline** than a true agent-specific value function.

This implies:

- the actor uses per-agent $\hat A_{i,t}$,
- but the critic predicts a shared value,
- so the semantics of actor and critic may become misaligned.

#### Case B: Use an agent-conditioned critic

A more natural form is

$$
V_\phi(s,i),
$$

where $i$ is an agent identity or role embedding.

Then the critic directly predicts

$$
V_\phi(s_t,i)\approx V_i^{\pi}(s_t).
$$

The TD residual becomes

$$
\delta_{i,t}=r_{i,t}^{\mathrm{shap}}+\gamma V_\phi(s_{t+1},i)-V_\phi(s_t,i).
$$

This makes reward, value, and advantage better aligned.

#### Case C: Use one critic per agent

The most direct but more expensive design is

$$
V_{\phi_i}(s_t),
$$

with a separate critic for each agent.

This is feasible for small systems such as DUO or TRIO, but scales less gracefully.

### 3.4 The Main Question I Want to Raise to My Professor

> If the system already uses per-agent Shapley rewards, should the single central critic also be replaced by an agent-conditioned critic, or even by multiple critics?

Because if the critic remains unchanged, the actor side can still be updated formally, but the semantics of the value target may no longer match the semantics of the actor-side reward and advantage.

---

## 4. My Current Interim Conclusions

1. **The baseline MARFT-A training pipeline** is essentially: sequential rollout + central critic + GAE + clipped PPO.
2. **The current change I have made** is to replace the original system-level reward with exact Shapley marginal contributions, so that each agent receives its own reward.
3. Then I would still use GAE and PPO, and the advantage A of each agent is more precise..
4. This naturally raises the third core question: **can a single critic still model the value of each agent accurately?**
5. My current tendency is that once reward becomes per-agent, the critic should at least be upgraded to an **agent-conditioned critic**; otherwise the semantics of actor and critic may become inconsistent.

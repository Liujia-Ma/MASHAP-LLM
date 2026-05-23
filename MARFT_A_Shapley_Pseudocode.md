# MARFT-A-Shapley 伪代码（LLMShap / PureShap）

本文给出基于当前代码实现的两种 Shapley 奖励分配模式伪代码：`llmshap` 与 `pureshap`。对应主流程参考：

- `mashap_llm/runner/shared/math_runner.py`
- `mashap_llm/reward/llmshap_allocator.py`
- `mashap_llm/reward/pureshap_allocator.py`
- `mashap_llm/critics/llmshap_estimator.py`
- `mashap_llm/buffers/action_level_buffer.py`
- `mashap_llm/algorithms/appo_trainer.py`

---

## 0. 对原始 baseline 伪代码的修正点（结合当前实现）

相较于图中的 baseline 伪代码，当前实现有以下关键修正：

1. **奖励不是单一标量 $r_t$**，而是每步的**按智能体奖励向量** $\mathbf{r_t}=[r_{t,1},...,r_{t,n}]$（在 Shapley 模式下由联盟边际贡献分配得到）。
2. **每个环境步都会进行奖励分配**（环境 `step` 内通过 callback），而不是 rollout 结束后统一“补”奖励。
3. **优势计算是 action-level 且按 agent 顺序耦合**（`ActionBuffer.compute_gae_and_returns` 的递推与标准单智能体 GAE 形式不同）。
4. `llmshap` 不只是“算 Shapley”，还包含**每步对联盟价值估计器的训练**（所有 coalition 监督）。

---

## 1. 主算法：MARFT-A with Shapley Reward Allocation

```text
Algorithm 1: MARFT-A-Shapley (mode ∈ {llmshap, pureshap})
Input:
  agent policies {π_i}, centralized critic V_φ,
  mode, episode_length T, PPO/APPO hyperparameters
Output:
  updated {π_i}, V_φ, (and estimator parameters in llmshap mode)

Initialize replay buffer D
for episode = 1..E do
    obs ← env.reset()

    for t = 0..T-1 do
        # (A) 顺序多智能体生成动作
        (rollout_obs_t, a_t^{1:n}, action_tokens_t, v_t^{1:n}, logp_t^{1:n})
            ← MAS.infer_for_rollout(obs)

        # (B) 环境推进 + 奖励分配（核心差异）
        (obs_next, r_t^{1:n}, done_t, info_t) ← env.step(a_t^{1:n})
        # env.step 内部:
        #   if mode == baseline:
        #       reward vector = [0,...,global_score]
        #   if mode == llmshap:
        #       reward vector = AllocateLLMShap(...)
        #   if mode == pureshap:
        #       reward vector = AllocatePureShap(...)

        D.add(rollout_obs_t, a_t^{1:n}, action_tokens_t, v_t^{1:n},
              r_t^{1:n}, obs_next, done_t, logp_t^{1:n})
        obs ← obs_next
    end for

    # (C) 计算 returns / advantages（action-level GAE）
    v_{T}^{1:n} ← MAS.get_next_values(obs)
    ComputeActionLevelGAE(D, v_{T}^{1:n})

    # (D) APPO/PPO 更新
    for epoch = 1..ppo_epoch do
        sample minibatch B from D
        update critic V_φ with value loss using returns in B
        update each policy π_i with clipped policy objective using advantages in B
    end for

    D.after_update()
end for
```

---

## 2. 子算法：LLMShap 奖励分配（每步执行）

```text
Algorithm 2: AllocateLLMShap(actions, base_state, global_score, problem, gt)
Input:
  当前完整联盟动作 actions, 原状态 base_state, 环境全局分 global_score
Output:
  per-agent reward vector φ ∈ R^n

Define all coalitions C ⊆ N
Cache full-coalition actions: A(N) = actions

Define rollout_fn(S):
    if A(S) cached: return A(S)
    else:
        run counterfactual sequential generation under coalition S
        (non-members receive deterministic absence message)
        cache and return A(S)

Define env_score(S):
    if S has no answer-capable agent: return 0
    evaluate coalition answers on verifier (math_verify/coding_verify)
    return averaged score

# Step-1: 训练联盟价值估计器（critic-side estimator）
for each coalition S in powerset(N):
    if S == ∅:
        target y_S = 0
        actions_S = absence-only actions
    else:
        actions_S = rollout_fn(S)
        y_S = env_score(S)
    tokens_S = build_joint_tokens(problem, actions_S, coalition=S)
    estimator.update_critic(tokens_S, y_S)

# Step-2: 使用估计值做 Shapley 分配（reward-side allocator）
Define coalition_value_fn(S):
    actions_S = rollout_fn(S)
    tokens_S = build_joint_tokens(problem, actions_S, coalition=S)
    return estimator.estimate_coalition_value(tokens_S, S)

φ = ExactShapley(coalition_value_fn)
φ = normalize(φ, total_score = global_score)   # llmshap 当前实现有归一化
return φ
```

---

## 3. 子算法：PureShap 奖励分配（每步执行）

```text
Algorithm 3: AllocatePureShap(actions, base_state, global_score, problem, gt)
Input:
  当前完整联盟动作 actions, 原状态 base_state, 环境全局分 global_score
Output:
  per-agent reward vector φ ∈ R^n

Define all coalitions C ⊆ N
Initialize coalition score table v:
    v(N) = global_score    # full coalition cache hit, 不重复 rollout

for each coalition S in C \ {N}:
    if S == ∅ or S has no answer-capable agent:
        v(S) = 0           # 硬编码 0（剪枝）
    else:
        actions_S = counterfactual rollout under S
        v(S) = verifier_score(actions_S, gt)
end for

φ = ExactShapleyFromTable(v)
assert sum_i φ_i == global_score (within tolerance)
return φ
```

---

## 4. 精确 Shapley 计算（两模式共用核心）

```text
Algorithm 4: ExactShapley(v)
for i in N:
    φ_i = 0
    for S ⊆ N\{i}:
        w = |S|! (|N|-|S|-1)! / |N|!
        φ_i += w * ( v(S ∪ {i}) - v(S) )
return φ
```

---

## 5. Action-level GAE（当前代码语义）

`ActionBuffer.compute_gae_and_returns` 采用反向 `(step, agent)` 遍历：

```text
for step = T-1..0:
    for agent = n-1..0:
        if agent == n-1:
            δ = r[step,agent] + γ * V[step+1,0] * mask[step+1,0] - V[step,agent]
            gae = δ + γλ * mask[step+1,0] * gae
        else:
            δ = r[step,agent] + γ * V[step,agent+1] * mask[step,agent+1] - V[step,agent]
            gae = δ + γλ * mask[step,agent+1] * gae
        Return[step,agent] = V[step,agent] + gae
        Adv[step,agent] = gae
```

> 注：这是该仓库 action-level 版本的实现细节；与标准单智能体 GAE 的索引形式不同，但与当前训练代码一致。

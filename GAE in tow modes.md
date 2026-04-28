# GAE in llmshap / pureshap Modes

## 1. Current Experiment Configuration

| Param                           |       llmshap |      pureshap |
| ------------------------------- | ------------: | ------------: |
| algorithm_name                  |          APPO |          APPO |
| n_agents                        |             2 |             2 |
| n_rollout_threads               |             1 |             1 |
| episode_length                  |             8 |             8 |
| horizon (`MathEnv.max_steps`) |             2 |             2 |
| gamma                           | **0.0** | **0.0** |
| gae_lambda                      |          0.95 |          0.95 |
| ppo_epoch                       |             1 |             1 |
| num_mini_batch                  |             1 |             1 |
| llmshap_layers                  |             3 |             \ |

因此一次参数更新前会收集 8 个环境 step。由于 `horizon=2`，每 2 个 step 环境就 done 并自动 reset，所以这 8 个 step 可以理解为 4 段短 rollout（每段 2 步）。

So one update collects 8 environment steps. Since `horizon=2`, the env is done and auto-resets every 2 steps, so these 8 steps can be viewed as 4 short rollouts (2 steps each).

---

## 2. GAE Equations Actually Used in APPO (Code-Aligned)

Implementation: `marft/buffers/action_level_buffer.py::compute_gae_and_returns`.

Let `t` be the environment step and `i` the agent index (`i=0` reasoner, `i=1` actor).

### 2.1 TD 残差 / TD Residual

For the last agent (`i = N-1`):

$$
\delta_{t,N-1}
= r_{t,N-1}
+ \gamma \, V_{t+1,0}\, m_{t+1,0}
- V_{t,N-1}.
$$

For non-last agents (`i < N-1`):

$$
\delta_{t,i}
= r_{t,i}
+ \gamma \, V_{t,i+1}\, m_{t,i+1}
- V_{t,i}.
$$

### 2.2 GAE 递推 / GAE Recurrence

The code uses one shared `gae`, recursively over reversed `(t, i)` order:

$$
A_{t,i}
= \delta_{t,i}
+ \gamma \lambda \, m_{\text{next}} \, A_{\text{next}}.
$$

And:

$$
R_{t,i} = V_{t,i} + A_{t,i}.
$$

### 2.3 Simplification Under Current Config ($\gamma=0$\)

Because ($\gamma=0$\), all bootstrap terms vanish:

$$
\delta_{t,i} = r_{t,i} - V_{t,i}, \quad
A_{t,i} = \delta_{t,i}, \quad
R_{t,i} = r_{t,i}.
$$

This is exactly why **reward leakage** is avoided: future value is no longer propagated into current advantages via $\gamma$\.

---

## 3. 公式变量逐项解释 / Variable-by-Variable Explanation

| 符号 / Symbol    | 中文解释                                                           | English Explanation                                                          |
| ---------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| $t$            | 环境 step 索引（本配置一次更新收集$t=0,\dots,7$）                | Environment step index (one update collects$t=0,\dots,7$)                  |
| $i$            | 智能体索引（0=reasoner, 1=actor）                                  | Agent index (0=reasoner, 1=actor)                                            |
| $r_{t,i}$      | 分配给智能体$i$ 的 step 奖励（llmshap/pureshap 的 Shapley 奖励） | Per-step reward allocated to agent$i$ (Shapley reward in llmshap/pureshap) |
| $V_{t,i}$      | critic 对该位置的 value 预测                                       | Critic value prediction at that position                                     |
| $m$            | mask（done 后为 0，否则为 1）                                      | Mask (0 if done, else 1)                                                     |
| $\gamma$       | 折扣因子（当前为 0）                                               | Discount factor (currently 0)                                                |
| $\lambda$      | GAE 参数（当前 0.95）                                              | GAE lambda (currently 0.95)                                                  |
| $\delta_{t,i}$ | TD 残差                                                            | TD residual                                                                  |
| $A_{t,i}$      | 优势（advantage）                                                  | Advantage                                                                    |
| $R_{t,i}$      | return（用于 critic 目标）                                         | Return (critic target)                                                       |

---

## 4. Per-Step State / Input / Output (MATH Env)

### 4.1 state 结构 / State Structure

Initial state from `MathEnv.reset()`:

```text
<|im_start|>problem: {MATH题目文本}<|im_end|>\n
```

`MathEnv.step()` appends:

```text
reasoner: {action_0}\n
actor: {action_1}\n
judge: The answer is incorrect/correct.\n
```

### 4.2 One-Step I/O Shapes (APPO)

- 输入到 MAS / Input to MAS: `obs` shape = `(n_rollout_threads, n_agents) = (1,2)`，元素是 state 字符串
- MAS 输出 / MAS output:
  - `rollout_obs` `(1,2)`（每个 agent 的条件化 prompt）
  - `actions` `(1,2)`（reasoner/actor 文本动作）
  - `values` `(1,2)`（critic 预测）
  - `log_probs` `(1,2)`
- 环境输出 / Env output:
  - `next_obs` `(1,2)`
  - `rewards` `(1,2)`
  - `dones` `(1,2)`
  - `infos`（含 `total_score`, `reward_vector`, `state` 等）

---

## 5. Reward Allocation Difference: llmshap vs pureshap

For 2 agents (planner, actor), coalition set is: ($\emptyset,\{P\},\{A\},\{P,A\}$).

- `pureshap`: uses real/counterfactual rollout + verifier score (coalitions without actor are forced to 0), then exact Shapley.
- `llmshap`: first learns/predicts coalition values with `CounterfactualEstimator`, then computes Shapley and normalizes to current `global_score`.

---

## 6. How 8 Steps (4 Rollouts) Update: Concrete Example

### 6.1 使用的 MATH 样例 / MATH Sample Used

来自 `train.json` 的一个题目（节选）：

A sample from `train.json` (excerpt):

> Circle $\omega$ has radius 5 and is centered at \(O\). Point \(A\) lies outside $\omega$ such that \(OA=13\)...
> `final_answer = 17`

### 6.2 8 步采样与更新节奏 / 8-Step Collection and Update Rhythm

一次 update 的 8 个 step：

8 steps for one update:

1. `t=0,1`：rollout-1（第 1 题，`horizon=2` 后 done）
2. `t=2,3`：rollout-2（环境自动 reset 到第 2 题）
3. `t=4,5`：rollout-3
4. `t=6,7`：rollout-4

Then:

1. `next_value = mas.get_next_values(obs_t8)`
2. `buffer.compute_gae_and_returns(next_value)`(反向计算所有 ($\delta, A, R$))
3. `trainer.train(...)`（critic 用 $R$，policy 用 $A$）

### 6.3 Numerical Demo for One Step (Current $gamma=0$)

Numbers below are illustrative for formula mechanics (not a verbatim single log line).

Assume at some step \(t\):

$$
r_t=[r_{t,0},r_{t,1}] = [0.28,\,0.72],\quad
V_t=[V_{t,0},V_{t,1}] = [0.10,\,0.65].
$$

Then

$$
\delta_{t,0}=0.28-0.10=0.18,\quad
\delta_{t,1}=0.72-0.65=0.07.
$$

Because $gamma=0$:

$$
A_{t,0}=0.18,\quad A_{t,1}=0.07,\quad
R_{t,0}=0.28,\quad R_{t,1}=0.72.
$$

The same computation is applied to all 8 steps ($t=0\sim7$), producing $8\times2$ advantages for PPO.

---

## 7. 结论 / Takeaway

- 在你当前 llmshap/pureshap 配置下，\(\gamma=0\) 使 GAE 退化为“即时 reward 减当前 value”，有效避免了跨步/跨位的 bootstrap 叠加。
- 两个模式的差别主要在 \(r_{t,i}\) 的来源（预测型 vs 精确型）；进入 APPO 的 \(\delta, A, R\) 计算路径相同。
- Under your current llmshap/pureshap setup, \(\gamma=0\) reduces GAE to immediate reward minus current value, avoiding cross-step/bootstrap accumulation.
- The two modes differ mainly in how \(r_{t,i}\) is produced (estimated vs exact); the APPO \(\delta, A, R\) pipeline is otherwise identical.

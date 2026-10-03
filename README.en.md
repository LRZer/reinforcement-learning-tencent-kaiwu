# Tencent Kaiwu RL Competition: PPO Gorge Chase Agent

[中文](README.md) · [Technical details](docs/method.en.md) · [Results and evidence](docs/results.en.md) · [Data inventory](data/README.md)

This project participated in the Artificial Intelligence Track of the Eastern Regional Competition of the 17th China University Student Service Outsourcing Innovation and Entrepreneurship Competition (descriptive English translation). It trains a Luban No. 7 agent in Tencent Kaiwu's `gorge_chase` environment (峡谷追猎, called “treasure hunting” in the course report). The objective is to collect treasures while evading monsters, survive longer, and decide when to use speed buffs and flash.

The solution combines **PPO, six MLP branches, a local-map CNN, learned gated fusion, BFS features and explicit memory**. Each step encodes the observation and accumulated information into state, selects an action with the policy network, and uses episode trajectories to update policy and value estimates. Historical records show **preliminary rank 37/158, leaderboard average score 997.15 and Regional Third Prize**; see [results](#7-competition-results-and-model-record) for evidence and interpretation.

Reading guide: [task](#1-task-and-technical-challenges) → [workflow](#2-technical-workflow) → [state and memory](#3-state-representation-and-explicit-memory) → [architecture](#4-model-architecture-and-action-selection) → [reward](#5-reward-design) → [PPO](#6-ppo-optimization-and-training-configuration) → [results](#7-competition-results-and-model-record) → [boundaries](#8-implementation-boundaries-and-evidence-scope) → [repository](#9-code-materials-and-contributors).

## 1. Task and technical challenges

The hero moves on a **128 × 128** grid and receives a **21 × 21** local map centered on itself each frame. Two monsters appear over time and pursue the hero. Treasures add score, buffs temporarily increase speed, and flash crosses terrain subject to cooldown. Each decision step executes one discrete action.

<img src="docs/evidence/environment-map.png" width="440" alt="Task-guide map illustration showing hero, monsters, obstacles and resources">

The illustration comes from p. 2 of the [original task guide](docs/reports/task-guide.pdf). Its scoring rules on pp. 3–9 define:

$$S = 1.5 \times \text{survived steps} + 100 \times \text{treasures collected}.$$

| Technical issue | Implemented treatment |
|---|---|
| Local visibility makes target information transient | Remember seen treasure and buff locations, explored regions, visit heat and recent behavior |
| Straight-line distances miss obstacles and connectivity | Eight-neighbor BFS, directional reachable-area ratios, topology summaries and a map CNN |
| Resource collection and safety require joint decisions | Encode threats, resources and terrain separately, combine context through a gate, and condition rewards on risk |
| Policy updates must use returns while retaining exploration | GAE, PPO policy/value clipping, entropy regularization, advantage normalization and gradient clipping |

The environment provides 10 public maps and five hidden evaluation maps. The archived configuration cycles through maps 1–8 for training and uses maps 9 and 10 for lightweight held-out-map validation. Hidden maps are supplied by the competition platform. The table states implementations and design purposes; measured module contributions are addressed under [evidence scope](#8-implementation-boundaries-and-evidence-scope).

## 2. Technical workflow

![Sampling, optimization, model synchronization and held-out-map validation](docs/figures/workflow.en.svg)

| Stage | Data flow | Implementation |
|---|---|---|
| ① Build state | Reset memory for a new episode; read hero, monsters, resources, map and skill state; update caches/history; form a 3601D observation and a separate 16D action mask | [Preprocessor](agent_ppo/feature/preprocessor.py), [observation flattening](agent_ppo/feature/definition.py) |
| ② Select an action | Produce 16 logits and one state value, apply the mask to construct probabilities, and sample during training | [Network](agent_ppo/model/model.py), [agent](agent_ppo/agent.py) |
| ③ Execute and receive feedback | Execute the action; processing the next frame computes reward from score, monster distances, terrain, resources and episode-ending changes | Preprocessor, [training workflow](agent_ppo/workflow/train_workflow.py) |
| ④ Process the trajectory | Store state, mask, action, reward, done, old value and full old action distribution; compute GAE backward after the episode | [Samples and GAE](agent_ppo/feature/definition.py) |
| ⑤ Optimize and synchronize | Send trajectories to the platform pool; the learner samples, performs PPO updates and synchronizes weights; the sampling agent loads `latest` at each new episode | [PPO optimizer](agent_ppo/algorithm/algorithm.py), [platform configuration](conf/configure_app.toml) |
| ⑥ Lightweight validation | Pause training sampling, reuse the same environment with maps 9 and 10, and select maximum-probability actions; trajectories only supply `val_*` statistics | Training workflow |

One decision follows observation → features/memory → network → masked probabilities → action. Gradient updates occur on the learner using probabilities and values recorded when sampling. Each validation round has four episodes total, with a configured 600-second interval. A zero-initialized timer allows the first round after the first training episode. Validation neither sends training samples nor calls gradient updates.

## 3. State representation and explicit memory

### 3.1 Observation layout

The input is **73 structured dimensions + 3528 map dimensions = 3601 dimensions**. The action mask is passed separately to probability processing and sample storage. Fields use reference-scale normalization and, where specified, clipping. See the [feature schema](data/metadata/feature_schema.json) for complete definitions.

| Branch | Size / flattened slice | Information and use |
|---|---|---|
| Hero | 14 / `[0:14]` | Position, progress, scores, treasure count, flash availability/cooldown, speed and monster-stage information |
| Monsters | 16 / `[14:30]` | Two slots: presence, visibility, relative position, Euclidean distance, speed, relative direction and distance bucket |
| Treasures | 14 / `[30:44]` | Visible/cached proportions, visible-target BFS distance, cached-target Euclidean distance/confidence and unknown-region distance |
| Buffs | 8 / `[44:52]` | Effect timer, visible or cached target position/distance, previous-action flash flag and flash count |
| Topology | 13 / `[52:65]` | Openness, obstacle density, neighboring traversable fraction, dead-end risk, wall proximity and eight reachable-area ratios |
| Memory | 8 / `[65:73]` | Revisits, position diversity, stuck flag, displacement, exploration direction, unknown fraction and no-progress flag |
| Local map | 3528 / `[73:3601]` | `8 × 21 × 21` spatial tensor encoded by the CNN |

The eight map channels are **traversability, obstacles, hero, monsters, visible treasures, visible buffs, visit heat and danger heat**. Map danger decays exponentially within a monster's 7×7 neighborhood and takes the maximum where monsters overlap. Rewards use a separate scalar danger measure derived from monster distances and speed.

### 3.2 Local BFS and targets

BFS (breadth-first search) computes reachable distances from the map center with equal-cost eight-neighbor connectivity and supplies visible-treasure ordering. Each direction also starts a search from the neighboring first cell, producing reachable cells / local traversable cells. Openness, dead-end and wall-proximity summaries provide terrain information.

Visible treasures are ordered by BFS distance and candidate value; remembered treasures by Euclidean distance. Buff targeting prefers visible resources, then remembered locations. Monster and buff BFS distances also support diagnostics and rewards. Search results and selected targets enter as features; the network chooses movement or flash.

### 3.3 Memory updates

| Memory | Update rule | Retained information |
|---|---|---|
| Treasure cache | Multiply by 0.995 each preprocessing call; visible locations become 1; candidate threshold >0.35; collection clears the hero's 3×3 neighborhood | Previously seen, uncleared locations and decaying values |
| Buff locations | Retain seen positions for the episode | Respawning resource locations; memory does not establish current availability |
| Visit heat | Multiply by 0.97; add 0.35 at the current location, capped at 1 | Frequently visited recent regions, cropped into a local-map channel |
| Behavior history | 20 positions, 12 actions and six displacements | Revisits, stuck behavior, no progress and flash history |
| Exploration target | Nearest remembered treasure → remembered buff → unknown cell | Direction toward a subsequent exploration target |

Global caches use 128×128 arrays and accumulate observed information only. The preprocessor explicitly maintains memory; the network consumes the current input containing those memory features. Reset clears state each episode, and preprocessing calls advance cache decay and internal timers.

## 4. Model architecture and action selection

![Separate encoders, gated fusion and actor-critic heads](docs/figures/architecture.en.svg)

### 4.1 Encoders

Each structured branch uses its own two-layer MLP (multilayer perceptron): `branch size → Linear32 → ReLU → Linear32 → ReLU`, producing 32D. The map CNN (convolutional neural network) encodes two-dimensional spatial relationships. Shapes below omit the batch dimension.

| Map-encoding stage | Output shape |
|---|---|
| Input map | `8 × 21 × 21` |
| 3×3 convolution, 16 channels + ReLU | `16 × 21 × 21` |
| 3×3 convolution, 32 channels + ReLU | `32 × 21 × 21` |
| 2×2 max pooling | `32 × 10 × 10` |
| 3×3 convolution, 64 channels + ReLU | `64 × 10 × 10` |
| Adaptive average pooling to 1×1, flatten | `64` |
| Linear64 + ReLU | `64` |

All convolutions use stride=1 and padding=1; max pooling uses stride=2. Structured branches encode entity and behavior summaries; the CNN produces a local spatial embedding.

### 4.2 Learned gate and fusion

Let the hero embedding be $h_h\in\mathbb{R}^{32}$ and the map embedding $h_m\in\mathbb{R}^{64}$. Concatenate them into 96D and apply `Linear32 → ReLU → Linear5 → Softmax` to obtain five weights:

$$\alpha=\operatorname{softmax}(\operatorname{MLP}_{gate}([h_h,h_m])),\qquad c=\sum_{i=1}^{5}\alpha_i h_i.$$

The five $h_i$ are monster, treasure, buff, topology and memory embeddings, each 32D. Weights are nonnegative and sum to one; context $c$ is 32D. Hero/map information determines the gate; entity information participates through its branch and corresponding map channels.

Fusion directly concatenates **hero32 + map64 + context32 = 128D**, then applies two `Linear128 → ReLU` layers. Hero and map feed both the gate and fusion; the five context branches are weighted and summed before fusion.

### 4.3 Heads and actions

| Output | Shape | Use |
|---|---|---|
| Actor, the policy head | `128 → 16` logits | Action preferences, converted to masked probabilities for action selection |
| Critic, the value head | `128 → 1` | Future discounted training reward estimate, used by advantage computation and value learning |

Actions **0–7** move east, northeast, north, northwest, west, southwest, south and southeast; **8–15** flash in the corresponding directions. Reference flash distance is 10 for cardinal directions and eight diagonally. The legal-action mask mainly reflects skill availability; a legal movement can still hit a wall. Training samples the probability distribution, while lightweight validation chooses its maximum.

![Input dimensions and parameter allocation](docs/figures/profiles.en.svg)

The network has **75,814 parameters**. The archived checkpoint contains **44 float32 tensors** and matches the current architecture strictly. Module counts and tensor shapes are in the [checkpoint metadata](data/metadata/checkpoint_inspection.json). Actor and critic share the encoders and fusion network.

## 5. Reward design

Training rewards provide feedback on score, risk and behavior after an action. Competition scoring follows the environment formula in Section 1. Step reward sums the following conditional terms.

![Conditional event increments in reward shaping](docs/figures/rewards.en.svg)

| Category | Key configuration and conditions |
|---|---|
| Survival and resources | `0.02 × positive step-score change`, about +0.03 per ordinary surviving step; +1.20 per treasure; +0.35 per buff collected |
| Safety | Distance-gain coefficient 0.10, danger-reduction coefficient 0.12 and squeeze-reduction coefficient 0.08; multiply these terms by 1.3 when steps ≥500 or danger >0.55 |
| Terrain | Dead-end and wall coefficients −0.05 and −0.02; coefficient +0.03 for openness above 0.35; all multiplied by `max(danger, 0.25)` |
| Target approach | Treasure coefficient 0.015 when danger <0.35 and no treasure gain; buff coefficient 0.012 when inactive, with no current resource gain, danger <0.55, buff distance ≤18 and treasure distance >4 |
| Invalid movement and loitering | Previous action exists and displacement <0.1: −0.03; danger <0.35, no resource gain and endpoint distance <5 across a 10-position window: −0.10 |
| Episode end | Capture −2.00; step-limit completion +1.20; abnormal truncation −0.50 |

Flash tiers use risk before the action and are checked in high → medium → low order. Let reference flash distance be $R$:

| Risk tier | Condition for positive reward | Satisfied / otherwise |
|---|---|---|
| High: danger ≥0.55 or nearest monster distance ≤6 | Euclidean distance gain ≥0.55R, or danger reduction ≥0.25, or BFS distance gain ≥0.60R | +0.20 / −0.12 |
| Medium: danger ≥0.30 or nearest monster distance ≤10 | Euclidean gain ≥0.35R, or danger reduction ≥0.08, or BFS gain ≥0.40R; additionally openness gain ≥0.05 | +0.08 / −0.04 |
| Low | Treasure collected and current danger <0.35 | +0.04 / −0.05 |

Distance-change terms are normalized and clipped; terrain terms are scaled by danger. The chart shows configured event increments, and multiple terms may apply at one step. See [reward-shaping details](docs/method.en.md#5-reward-shaping) for danger definitions, complete equations and reference thresholds.

## 6. PPO optimization and training configuration

### 6.1 Advantages and value targets from trajectories

Trajectories retain sampling-time state values $V_{old}$ and the full action distribution $\pi_{old}$. GAE (generalized advantage estimation) works backward from the final step, combining immediate reward, next-state value and later feedback to estimate how an outcome differs from expectations:

$$\delta_t=r_t+\gamma(1-d_t)V_{old}(s_{t+1})-V_{old}(s_t),\qquad A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}.$$

Here $d_t$ marks episode end, $r_t$ is shaped reward, $\gamma=0.99$ and $\lambda=0.95$. The value target is $A_t+V_{old}(s_t)$, stored as `reward_sum`. The final next-state value is zero; capture and all truncations count as done. Advantages are normalized within each learner batch.

### 6.2 Policy, value and joint losses

PPO (proximal policy optimization) compares the selected action's new and stored probabilities, uses advantages to adjust the policy, and clips objective gains from excessive probability changes:

$$\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)},\qquad L_{policy}=-\mathbb{E}\left[\min\left(\rho_t A_t,\operatorname{clip}(\rho_t,0.8,1.2)A_t\right)\right].$$

The critic fits the value target with value clipping; entropy $H(\pi)$ encourages action exploration. The combined loss is:

$$L=L_{policy}+0.5L_{value}-0.005H(\pi).$$

This code's $L_{value}$ already includes `0.5 × max(ordinary squared error, clipped-value squared error)`. Each 512-sample learner batch runs four epochs, reshuffling each epoch and updating in mini-batches of 256. Gradient norm is clipped to 0.5 before Adam updates. See [PPO and samples](docs/method.en.md#4-ppo-and-samples) for the sample fields and exact calculations.

### 6.3 Archived settings and monitoring

| Environment setting | Value | Optimization / sampling setting | Value |
|---|---|---|---|
| Training / lightweight validation maps | 1–8 / 9 and 10 | γ / GAE λ | 0.99 / 0.95 |
| Treasures / buffs | 10 / 2 | Adam learning rate | 3 × 10⁻⁴ |
| Buff respawn / flash cooldown | 200 / 200 steps | Policy/value clip range | 0.2 |
| Second monster / monster speedup | 200 / 300 steps | Entropy / value coefficient | 0.005 / 0.5 |
| Episode step limit | 1000 | Epochs / mini-batch | 4 / 256 |
| Validation episodes / interval | Four total per round / 600 seconds | Learner batch / pool capacity | 512 / 2048 |
| Model sync / workflow-save interval | 60 / 1800 seconds | Pool sampling / removal | Uniform / FIFO |

The platform uses a MinSize limiter and a preload ratio of 0.5 of pool capacity. These describe pool configuration. Monitoring separates environment score, summed shaped reward, survival steps, treasure count and episode-ending state. Optimization diagnostics include losses, entropy, probability ratios, clip fraction and gradient norm. See the [metric dictionary](data/metadata/metrics.csv) and [configuration snapshot](data/metadata/configuration.json).

## 7. Competition results and model record

![Historical competition results and evidence sources](docs/figures/results.en.svg)

| Record | Value | Original evidence |
|---|---|---|
| Preliminary rank | **37 / 158** | Report p. 3 text; screenshot confirms rank 37, while total team count comes from report text |
| Leaderboard average score | **997.15** | [Leaderboard screenshot](docs/evidence/leaderboard.png), team `ChatGPT` |
| Completed / total episodes | **150 / 150** | Completed-episode column in the same screenshot |
| Regional award | **Third Prize, Artificial Intelligence Track** | [Certificate](docs/evidence/award-certificate.png), June 2026, Eastern Regional Competition |
| Final status | Did not advance | Report p. 3 text |

150/150 counts completed evaluations; it cannot establish win rate or the fraction surviving to the step limit. The historical platform mean 997.15 cannot be decomposed into mean survival and mean treasures alone. Full context is in the [course report](docs/reports/course-report.pdf); machine-readable records are available as [JSON](data/results/competition.json) and [CSV](data/results/competition.csv).

The archived model is [model.ckpt-34055.pkl](ckpt/model.ckpt-34055.pkl), 316,415 bytes, storing a network `state_dict`. Platform metadata records `train_step=34055` and export date 2026-04-27. The report calls it the final training package, but the leaderboard does not show a submitted model ID, so checkpoint records and score evidence are listed separately. This identifier is not interpreted as environment interactions; see the [model record](docs/results.en.md#checkpoint-record).

## 8. Implementation boundaries and evidence scope

| Scope | Meaning of the available implementation / materials |
|---|---|
| Local BFS | Unreachable or out-of-view targets fall back to distance 20; diagonal search checks only the destination; directional search may return through the center, producing identical scores within one component |
| Stateful preprocessing | Original lightweight validation processes the same frame repeatedly, advancing caches, timers and history again; official platform evaluation uses another workflow |
| Environment and reference constants | Environment speedup is configured at step 300, while some stage/reward references use 500; a 10-position loiter window spans nine position changes under one call per frame |
| Structural statistics | Input sizes, parameter counts and reward-event charts come from code or static weight inspection; per-state gate activations were not retained |
| Performance evidence | Leaderboard mean, rank and certificate are retained; per-episode scores, training/validation curves, hidden-map breakdowns, hardware/timing records and ablations are unavailable |

Individual module gains, score variance, flash success rate and generalization improvement therefore have no quantitative record. See [implementation details](docs/method.en.md#7-implementation-boundaries) and [evidence documentation](docs/results.en.md) for semantics, statistical interpretation and figure provenance.

## 9. Code, materials and contributors

| Path | Contents |
|---|---|
| [agent_ppo/](agent_ppo/) | Agent entry point, features/memory, network, PPO optimization, rewards and training workflow |
| [agent_diy/](agent_diy/) | Original unimplemented platform DIY template |
| [conf/](conf/) | Platform pool, model synchronization and evaluation configuration |
| [ckpt/](ckpt/) | Model 34055, index and `kaiwu.json` |
| [archive/](archive/) | Previous configuration referencing 54627; those weights were not retained |
| [data/](data/) | Competition CSV/JSON, feature schema, configuration, metrics, checkpoint metadata and original-file hashes |
| [docs/](docs/) | Bilingual technical/results documents, two original PDFs, leaderboard/certificate evidence and five bilingual figure types in SVG/PNG |
| [scripts/](scripts/) / [tests/](tests/) | Figure generation, weight/integrity inspection tools and checks of original numerical logic |

Of 57 source files, 40 are preserved byte for byte. Hashes and exclusion reasons for 16 Python cache files and one platform signing artifact are recorded in the [source inventory](data/metadata/original_files.json). Online training trajectories and original map assets were not saved in the supplied directory. The [synthetic observation](data/examples/synthetic_observation.json) is a labeled interface example. See the [data inventory](data/README.md) for provenance and purpose.

The report lists **Liu Runzhang (刘润章), Chen Xudong (陈旭东) and Nan Yibo (南怡波)**, each contributing approximately one third of solution development and iteration. Chen prepared the report, Nan the slides, and Liu the explanatory video. This archive includes the original report and task guide; slides and video were not supplied. Third-party code and material attribution is documented in [NOTICE](NOTICE.md).

Algorithm references: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

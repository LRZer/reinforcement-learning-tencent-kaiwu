# Tencent Kaiwu: PPO Treasure-Hunting Agent

[中文](README.md) · [Technical details](docs/method.en.md) · [Result evidence](docs/results.en.md) · [Data inventory](data/README.md)

This project develops an agent for Tencent Kaiwu's `gorge_chase` environment (峡谷追猎). The agent controls Luban No. 7, collects treasures, evades pursuing monsters, and decides when to use speed buffs and directional flash. It combines **PPO, separate MLP/CNN encoders and a learned type gate** with local BFS features, target caches and explicit behavioral memory.

At each step, nearby terrain, threats, resources and recent behavior become features. A neural network selects a movement or flash action, and training updates that policy from reward feedback.

The project participated in the **17th China University Student Service Outsourcing Innovation and Entrepreneurship Competition, Eastern Regional Competition, Artificial Intelligence Track** (descriptive English translation). This repository archives the competition code, the final package's checkpoint, the original report and task guide, with bilingual technical figures and structured result data.

Reading order: [task and scoring](#task-and-scoring) → [network and observation](#network-and-observation) → [reward design](#reward-design) → [technical workflow and training](#technical-workflow-and-training). Field definitions, equations and implementation boundaries are collected in the [technical documentation](docs/method.en.md).

## Competition results

![Competition results and evidence](docs/figures/results.en.svg)

| Item | Result | Evidence |
|---|---:|---|
| Preliminary rank | **37 / 158** | Report p. 3 text; screenshot confirms rank 37 |
| Leaderboard average score | **997.15** | Leaderboard screenshot, team `ChatGPT` |
| Completed / total episodes | **150 / 150** | Same screenshot |
| Regional award | **Third Prize** | Certificate on p. 4, June 2026 |
| Final qualification | Did not advance | Report p. 3 text |

These are historical results recorded in the supplied materials. The episode count describes completed evaluations, not a win rate. Per-episode scores, training curves, validation logs and ablation data are unavailable; no score variance, module improvement or held-out-map result is claimed. See [evidence and interpretation](docs/results.en.md).

## Task and scoring

The hero moves in a **128 × 128** grid and observes a **21 × 21** local region. Two monsters appear over time and pursue the hero. Treasures add score, buffs temporarily increase movement speed, and flash crosses terrain subject to a cooldown.

<img src="docs/evidence/environment-map.png" width="400" alt="Task-guide illustration of an environment map">

Map illustration from task-guide p. 2.

The [task guide](docs/reports/task-guide.pdf), pp. 3–9, defines the episode score as:

$$S = 1.5 \times \text{survived steps} + 100 \times \text{treasures collected}.$$

The environment offers 10 public maps and 5 hidden evaluation maps. The archived local configuration trains on maps 1–8 and reserves maps 9 and 10 for lightweight validation. Hidden maps are provided by the competition platform. **Training reward differs from competition score:** it also includes safety, terrain, flash quality and anti-loitering terms.

## Implemented work

| Component | Implementation | Source |
|---|---|---|
| State representation | 73 structured dimensions + 8 map channels, 3601 dimensions total | [preprocessor.py](agent_ppo/feature/preprocessor.py) |
| Local search and memory | Eight-neighbor BFS, reachable-area ratios, decaying treasure cache, buff-location cache, exploration and visit memory | Same file |
| Policy network | Six independent MLP encoders, a map CNN, a five-branch context gate and actor/critic heads | [model.py](agent_ppo/model/model.py) |
| Optimization | PPO policy/value clipping, GAE, advantage normalization, entropy regularization and gradient clipping | [algorithm.py](agent_ppo/algorithm/algorithm.py), [definition.py](agent_ppo/feature/definition.py) |
| Reward shaping | Treasure, survival, danger, squeeze, terrain, buff, flash quality and loitering in safe states | [configuration](agent_ppo/conf/conf.py), preprocessor |
| Training workflow | Episode trajectories, model synchronization, periodic saving and separate `val_*` monitoring | [train_workflow.py](agent_ppo/workflow/train_workflow.py) |

These describe implemented functions. Their individual contributions to the competition score have not been measured by ablation experiments.

The CNN encodes spatial terrain; BFS supplies local distances around obstacles and connectivity features; caches retain previously observed resources. The policy network chooses the action. Treasure confidence is multiplied by 0.995 and visit heat by 0.97 per preprocessing call. Exploration targets follow remembered treasure → remembered buff → unknown cell. The preprocessor maintains memory, and the network consumes the current observation containing those memory features. Global caches contain only previously observed information.

## Network and observation

![Actor-Critic network architecture](docs/figures/architecture.en.svg)

The **actor** produces action preferences, converted to probabilities for movement or flash. The **critic** estimates future discounted training rewards from the current state; training uses this estimate to assess whether an action's outcome was better or worse than expected. Both heads share the encoders and fusion network.

| Input branch | Dimensions | Contents |
|---|---:|---|
| Hero | 14 | Position, progress, scores, skill cooldown, movement speed |
| Monsters | 16 | Presence, visibility, relative position, distance and speed of two monsters |
| Treasures | 14 | Visible-target BFS distance, cached-target Euclidean distance and exploration frontier |
| Buffs | 8 | Estimated timer, visible or cached targets and flash history |
| Topology | 13 | Openness, obstacle density, branches, dead-end risk, wall proximity and eight reachable-area ratios |
| Memory | 8 | Revisits, position diversity, displacement, exploration direction and no-progress flag |
| Local map | 3528 | `8 × 21 × 21`: traversable cells, obstacles, hero, monsters, treasures, buffs, visit heat and danger |

The layer sequence below omits the batch dimension:

| Stage | Layers and shapes | Role |
|---|---|---|
| Structured encoders | Six separate MLPs, each `input size → 32 → 32`, with ReLU after each layer | Encode hero, monsters, treasures, buffs, topology and memory independently |
| Map encoder | `8×21×21 → Conv16 → Conv32 → MaxPool2 → Conv64 → global average pool → Linear64` | Convolutions use 3×3 kernels and padding=1; convolutions and projection are followed by ReLU; spatial size after max pooling is 10×10; output is 64D |
| Learned gate | `hero32 + map64 → Linear32 → ReLU → Linear5 → Softmax` | Five weights sum to 1; their weighted sum of monster, treasure, buff, topology and memory embeddings forms context32 |
| Fusion | `hero32 + map64 + context32 → 128 → 128`, both layers followed by ReLU | Hero and map embeddings enter fusion directly and also determine the context weights |
| Output heads | Actor: `128 → 16`; critic: `128 → 1` | Produce action logits and one state-value estimate |

For example, changing monster and resource information on the same terrain changes the context-branch embeddings; gate weights are computed from hero and map embeddings. Per-state gate weights were not retained, so branch weights in specific situations cannot be reported.

There are **16 actions**: 0–7 move east, northeast, north, northwest, west, southwest, south and southeast; 8–15 flash in the same directions. After the network produces logits, the agent applies the environment's 16D legal-action mask to construct probabilities. Training samples actions; validation takes the maximum probability. The mask mainly reflects skill availability; a legal movement can still hit a wall.

The model has **75,814 parameters**. The archived checkpoint contains **44 tensors**, loads strictly into the network and passes numeric and forward-output checks. See [method details](docs/method.en.md) for tensor shapes, parameter allocation and reward formulas.

## Reward design

The step reward sums increments for scoring, risk and action quality. The competition leaderboard uses the environment score defined above.

![Event increments in the training reward](docs/figures/rewards.en.svg)

| Category | Key values and conditions | Behavior addressed |
|---|---|---|
| Score and resources | About +0.03 per ordinary surviving step; +1.20 per treasure; +0.35 per buff collected | Survival and resource collection |
| Safety and terrain | Distance-gain coefficient 0.10, danger-reduction coefficient 0.12, squeeze-reduction coefficient 0.08; penalties for dead ends and wall proximity | Distance from monsters, relief from two-monster pressure and terrain choice |
| Target progress | Treasure-approach coefficient 0.015 when danger <0.35 and no treasure is collected; buff-approach coefficient 0.012 with additional effect-state and target-distance conditions | Progress toward targets under lower risk |
| Action quality | −0.03 for displacement <0.1; −0.10 when safe, with no resource gain and endpoint distance <5 across a 10-position window | Wall collisions and local loitering |
| Flash quality | High-risk effective escape +0.20, otherwise −0.12; medium-risk escape into more open terrain +0.08, otherwise −0.04; low-risk safe treasure gain +0.04, otherwise −0.05 | Flash evaluated by pre-action risk, distance, danger and terrain changes |
| Episode end | Capture −2.00; step-limit completion +1.20; abnormal truncation −0.50 | Distinguish episode-ending conditions |

Coefficients are multiplied by the corresponding distance changes, risk or terrain measures. Several events may add at the same step. “Effective escape” is a reward condition. These values describe code configuration; see [reward shaping](docs/method.en.md#5-reward-shaping) for all equations and thresholds.

## Technical workflow and training

![Sampling and validation workflow](docs/figures/workflow.en.svg)

1. **Build state:** reset caches and history for each episode; process the current observation into local BFS, resource targets, risk, map and memory features, then flatten to 3601 dimensions.
2. **Choose and execute an action:** the network produces logits and `V(s)`; sample after masking. The environment executes the action, and preprocessing the next observation computes reward from the resulting changes.
3. **Collect the trajectory:** store state, mask, action, reward, done flag, old value and the full old action distribution. At episode end, compute GAE backward to obtain advantage `A` and value target `A + V`; the final next-state value is zero.
4. **Update the model:** send the trajectory to the platform pool. The learner takes 512 samples, normalizes advantages, and performs four PPO epochs with mini-batches of 256. Pool sampling is uniform and old-sample removal is FIFO.
5. **Synchronize and validate:** platform configuration synchronizes models every minute; the training workflow loads `latest` at the start of an episode. Validation uses maximum-probability actions on maps 9 and 10, four episodes total per round, reporting `val_*` means without sending trajectories to the pool.

GAE (generalized advantage estimation) combines rewards, current/next state values and later feedback to estimate how an outcome differs from the value prediction. PPO constructs its policy objective from the selected action's new probability divided by its stored behavior-policy probability. Clipping uses `[0.8, 1.2]` to limit objective gains from large probability changes. The critic fits `A + V`, and entropy regularization encourages continued action exploration.

The joint loss is `L = L_policy + 0.5 L_value − 0.005 H(π)`. In this code, `L_value` already includes `0.5 × max(ordinary squared error, clipped squared error)`. See [PPO and samples](docs/method.en.md#4-ppo-and-samples) for policy/value clipping, advantage normalization and gradient clipping.

| Environment | Archived value | PPO | Archived value |
|---|---|---|---|
| Training / validation maps | 1–8 / 9 and 10 | γ / GAE λ | 0.99 / 0.95 |
| Treasures / buffs | 10 / 2 | Adam learning rate | 3 × 10⁻⁴ |
| Buff respawn / flash cooldown | 200 / 200 steps | Policy/value clip range | 0.2 |
| Second monster / monster speedup | 200 / 300 steps | Entropy / value coefficient | 0.005 / 0.5 |
| Episode step limit | 1000 | Epochs / mini-batch | 4 / 256 |
| Lightweight validation | 4 episodes total per round, 600-second interval | Learner batch / pool capacity | 512 / 2048 |

Training monitors distinguish environment score, summed shaped reward, steps, treasures and episode-ending states; optimization diagnostics include losses, entropy and probability ratios. See the [metric dictionary](data/metadata/metrics.csv). The 600-second value is the configured validation interval; its zero-initialized timer allows the first round after the first training episode.

Original lightweight validation repeats preprocessing of the same frame. The environment speeds up monsters at step 300, while some feature/reward references use 500. BFS supplies approximate local connectivity, and its eight-neighbor search does not fully enforce diagonal movement constraints. See [implementation boundaries](docs/method.en.md#7-implementation-boundaries) for precise semantics; historical validation results are unavailable.

## Repository map

```text
agent_ppo/       Original PPO agent, features, network, rewards and workflow
agent_diy/       Original unimplemented DIY template; not a comparison baseline
conf/           Original platform runtime configuration
ckpt/           Checkpoint 34055, id_list and kaiwu.json
archive/        Previous evaluation configuration: 54627 (weights unavailable)
data/           Competition CSV/JSON, feature/weight metadata and SHA-256 inventory
docs/           Bilingual technical/results documentation, PDFs, evidence and figures
scripts/        Weight/integrity inspection and figure tools
tests/          Offline checks of the original numerical code
```

The supplied directory contains 57 files. Forty are preserved byte for byte with provenance; 16 reproducible Python cache files and one platform signing artifact are excluded, with hashes and reasons in the [archive inventory](data/metadata/original_files.json). Historical trajectories and map data were not supplied. The [synthetic observation](data/examples/synthetic_observation.json) is an interface fixture, not a recorded game.

## Contributors and sources

The report lists Liu Runzhang (刘润章), Chen Xudong (陈旭东) and Nan Yibo (南怡波), each contributing approximately one third of solution development and iteration. Chen prepared the report, Nan the slides, and Liu the explanatory video. The supplied directory contains no slide deck or video. The [report](docs/reports/course-report.pdf) and [task guide](docs/reports/task-guide.pdf) are preserved as supplied; see [NOTICE](NOTICE.md) for third-party attribution.

Algorithm references: [PPO](https://arxiv.org/abs/1707.06347), [GAE](https://arxiv.org/abs/1506.02438). SVG and PNG figures are in `docs/figures/`; see [figure provenance](docs/results.en.md#figure-provenance).

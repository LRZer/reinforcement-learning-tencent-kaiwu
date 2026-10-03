#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
当前版本的配置目标：
1. 不改 observation 总维度，不去联动修改 model.py / definition.py。
2. 在现有多分支结构上，加入局部 BFS 特征与宝箱/buff 缓存。
3. 用更贴近现象的“防磨蹭奖励”替代过重的停顿惩罚。
4. 让 buff 有明确价值，但不把 buff 抬成唯一主目标。
"""


class Config:
    # --------------------------------------------------
    # Observation structure
    # --------------------------------------------------
    MAP_SIZE = 21
    MAP_CHANNELS = 8
    MAP_FLAT_DIM = MAP_CHANNELS * MAP_SIZE * MAP_SIZE  # 8 * 21 * 21 = 3528

    # 维度保持不变，避免联动修改模型结构
    HERO_DIM = 14
    MONSTER_DIM = 16
    TREASURE_DIM = 14
    BUFF_DIM = 8
    TOPO_DIM = 13
    MEMORY_DIM = 8

    ACTION_NUM = 16
    VALUE_NUM = 1

    # Flat obs layout:
    # [hero | monster | treasure | buff | topo | memory | map_flat]
    FEATURES = [HERO_DIM, MONSTER_DIM, TREASURE_DIM, BUFF_DIM, TOPO_DIM, MEMORY_DIM]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)            # 73
    DIM_OF_OBSERVATION = FEATURE_LEN + MAP_FLAT_DIM   # 3601

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    BRANCH_HIDDEN_DIM = 32
    MAP_EMBED_DIM = 64
    FUSION_DIM = 128

    # --------------------------------------------------
    # PPO
    # --------------------------------------------------
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 3e-4
    BETA_START = 0.005
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    GRAD_CLIP_RANGE = 0.5
    EPS = 1e-8
    ADV_NORM = True
    VALUE_CLIP = True

    PPO_EPOCHS = 4
    PPO_MINI_BATCH_SIZE = 256

    # --------------------------------------------------
    # Environment assumptions / soft defaults
    # --------------------------------------------------
    DEFAULT_MAX_STEP = 1000
    DEFAULT_FLASH_CD = 100
    DEFAULT_BUFF_DURATION = 50
    DEFAULT_MONSTER_INTERVAL = 300
    DEFAULT_MONSTER_SPEEDUP_STEP = 500

    MAX_TREASURE_COUNT = 10
    MAX_BUFF_COUNT = 2

    # --------------------------------------------------
    # Reward scales
    # --------------------------------------------------
    # 主目标：活久一点 + 吃到箱子
    REWARD_STEP_GAIN_SCALE = 0.02
    REWARD_TREASURE_GAIN_SCALE = 1.2

    # 生存 shaping：保持简单稳定
    REWARD_SAFE_DELTA_SCALE = 0.10
    REWARD_DANGER_SCALE = 0.12
    REWARD_SQUEEZE_SCALE = 0.08

    # 地形 shaping：轻量即可，不要过度主导 reward
    REWARD_DEADEND_SCALE = 0.05
    REWARD_OPENNESS_SCALE = 0.03

    # 宝箱推进：只有在比较安全时才鼓励
    REWARD_TREASURE_APPROACH_SCALE = 0.015
    TREASURE_SAFE_DANGER_TH = 0.35

    # buff：适度提高，但不把策略带偏
    REWARD_BUFF_PICK = 0.35
    REWARD_BUFF_APPROACH_SCALE = 0.012
    BUFF_APPROACH_MAX_DIST = 18.0
    BUFF_APPROACH_TREASURE_DIST_TH = 4.0
    BUFF_APPROACH_DANGER_TH = 0.55

    # 动作质量
    REWARD_INVALID_MOVE = -0.03
    REWARD_FLASH_HIGH_GOOD = 0.20
    REWARD_FLASH_HIGH_BAD = -0.12
    REWARD_FLASH_MID_GOOD = 0.08
    REWARD_FLASH_MID_BAD = -0.04
    REWARD_FLASH_LOW_BAD = -0.05
    REWARD_FLASH_TREASURE_SAFE = 0.04
    FLASH_HIGH_DANGER_TH = 0.55
    FLASH_HIGH_DIST_TH = 6.0
    FLASH_MID_DANGER_TH = 0.30
    FLASH_MID_DIST_TH = 10.0
    FLASH_DIST_GAIN_RATIO = 0.55
    FLASH_PATH_GAIN_RATIO = 0.60
    FLASH_MID_DIST_GAIN_RATIO = 0.35
    FLASH_MID_PATH_GAIN_RATIO = 0.40
    FLASH_OPENNESS_GAIN_TH = 0.05
    FLASH_SAFE_DANGER_TH = 0.35

    # 终局
    REWARD_TERMINATED = -2.0
    REWARD_COMPLETED = 1.2
    REWARD_ABNORMAL = -0.5

    # --------------------------------------------------
    # 防磨蹭奖励（你提出的 10 步 L2 版本）
    # --------------------------------------------------
    # 如果 10 步之后还离 10 步前的位置很近，说明在小范围反复摩擦/剐蹭。
    REWARD_LOITER = -0.10
    LOITER_WINDOW = 10
    LOITER_L2_THRESHOLD = 5.0
    LOITER_DANGER_TH = 0.35

    # --------------------------------------------------
    # BFS / 记忆 / 缓存
    # --------------------------------------------------
    RECENT_POS_QUEUE = 20
    RECENT_ACTION_QUEUE = 12
    RECENT_PROGRESS_WINDOW = 6
    REVISIT_DECAY = 0.97
    TREASURE_MEMORY_DECAY = 0.995

    # 归一化时用到的参考距离。21x21 局部图里，20 已经足够覆盖绝大多数可达距离。
    BFS_DIST_NORM = 20.0
    TARGET_DIST_NORM = 20.0

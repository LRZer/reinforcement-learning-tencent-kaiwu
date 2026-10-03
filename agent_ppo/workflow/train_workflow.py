#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
训练 + 本地轻量验证 workflow。

目标：
1. 训练图固定为 1~8（由 train_env_conf.toml 控制）。
2. 在同一个 workflow 里，周期性地用地图 9、10 跑少量验证 episode。
3. 验证 episode 不发送样本、不参与训练，只上报 val_* 指标。

为什么这样做：
- 你现在最大的风险不是“模型没提升”，而是“训练图上越来越熟，误以为泛化变强”。
- 这个文件的作用就是把 train 和 val 从流程上拆开。

注意：
- 这里是“本地轻量验证”，用于快速看趋势。
- 真正正式评估，仍然建议在平台上单独创建评估任务，只跑 9、10 图。
"""

import copy
import os
import time
import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process, flatten_obs
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


# ---------------------------------------------
# 本地验证配置
# ---------------------------------------------
VAL_MAPS = [9, 10]
VAL_MAP_RANDOM = False
VAL_EPISODES_EACH_ROUND = 4          # 每轮验证跑几局；先小一点，别把训练打断太久
VAL_RUN_INTERVAL_SECONDS = 600       # 每 10 分钟做一轮验证
SAVE_MODEL_INTERVAL_SECONDS = 1800   # 每 30 分钟保存一次


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    train_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if train_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    val_conf = _build_val_conf(train_conf)

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        train_conf=train_conf,
        val_conf=val_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            # 训练样本只来自 train episode
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= SAVE_MODEL_INTERVAL_SECONDS:
                agent.save_model(id="latest")
                last_save_model_time = now

            # 周期性跑一轮验证。验证不送样本，只上报 val_* 指标。
            episode_runner.maybe_run_validation(now)


class EpisodeRunner:
    def __init__(self, env, agent, train_conf, val_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.train_conf = train_conf
        self.val_conf = val_conf
        self.logger = logger
        self.monitor = monitor

        self.train_episode_cnt = 0
        self.val_round_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.last_val_time = 0

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None and self.logger:
                    self.logger.info(f"training_metrics is {training_metrics}")

            env_obs = self.env.reset(self.train_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.train_episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            final_diag = remain_info.get("diag", {})

            if self.logger:
                self.logger.info(f"[TRAIN] Episode {self.train_episode_cnt} start")

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)
                _, env_obs = self.env.step(act)

                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = bool(env_obs["terminated"])
                truncated = bool(env_obs["truncated"])
                step += 1
                done = terminated or truncated

                next_obs_data, next_remain = self.agent.observation_process(env_obs)
                reward = np.array(next_remain.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])
                final_diag = next_remain.get("diag", final_diag)

                flat_obs = flatten_obs(obs_data)
                frame = SampleData(
                    obs=flat_obs,
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.int64),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    if self.logger:
                        result_str = "TERMINATED" if terminated else "TRUNCATED"
                        self.logger.info(
                            f"[TRAIN GAMEOVER] episode:{self.train_episode_cnt} steps:{step} "
                            f"result:{result_str} total_reward:{total_reward:.3f}"
                        )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        train_monitor_data = self._build_episode_monitor_data(
                            prefix="train",
                            total_reward=total_reward,
                            terminated=terminated,
                            truncated=truncated,
                            step=step,
                            final_diag=final_diag,
                        )
                        train_monitor_data["train_episode_cnt"] = self.train_episode_cnt
                        self.monitor.put_data({os.getpid(): train_monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector, gamma=Config.GAMMA, lamda=Config.LAMDA)
                        yield collector
                    break

                obs_data = next_obs_data

    # --------------------------------------------------
    # Validation loop
    # --------------------------------------------------
    def maybe_run_validation(self, now):
        if now - self.last_val_time < VAL_RUN_INTERVAL_SECONDS:
            return

        self.last_val_time = now
        self.val_round_cnt += 1

        if self.logger:
            self.logger.info(
                f"[VAL] start validation round {self.val_round_cnt}, maps={VAL_MAPS}, episodes={VAL_EPISODES_EACH_ROUND}"
            )

        metrics = []
        for episode_idx in range(VAL_EPISODES_EACH_ROUND):
            item = self._run_one_validation_episode(episode_idx)
            if item is not None:
                metrics.append(item)

        if not metrics:
            if self.logger:
                self.logger.warning("[VAL] no valid validation episodes collected")
            return

        val_monitor_data = {
            "val_reward": round(float(np.mean([m["reward"] for m in metrics])), 4),
            "val_total_score": round(float(np.mean([m["total_score"] for m in metrics])), 4),
            "val_steps": round(float(np.mean([m["steps"] for m in metrics])), 4),
            "val_treasures": round(float(np.mean([m["treasures"] for m in metrics])), 4),
            "val_terminated_rate": round(float(np.mean([m["terminated_rate"] for m in metrics])), 4),
            "val_round_cnt": self.val_round_cnt,
        }

        if self.logger:
            self.logger.info(f"[VAL] summary {val_monitor_data}")

        if self.monitor:
            self.monitor.put_data({os.getpid(): val_monitor_data})

    def _run_one_validation_episode(self, episode_idx):
        env_obs = self.env.reset(self.val_conf)
        if handle_disaster_recovery(env_obs, self.logger):
            return None

        self.agent.reset(env_obs)
        # 验证前加载最新模型。这样验证的是“当前最接近训练最新状态”的策略。
        self.agent.load_model(id="latest")

        done = False
        step = 0
        total_reward = 0.0

        # exploit：固定走最大概率动作，减少随机性
        obs_data, remain_info = self.agent.observation_process(env_obs)
        final_diag = remain_info.get("diag", {})

        while not done:
            act = self.agent.exploit(env_obs)
            _, env_obs = self.env.step(act)

            if handle_disaster_recovery(env_obs, self.logger):
                return None

            terminated = bool(env_obs["terminated"])
            truncated = bool(env_obs["truncated"])
            done = terminated or truncated
            step += 1

            _, remain_info = self.agent.observation_process(env_obs)
            reward = np.array(remain_info.get("reward", [0.0]), dtype=np.float32)
            total_reward += float(reward[0])
            final_diag = remain_info.get("diag", final_diag)

            if done:
                result_str = "TERMINATED" if terminated else "TRUNCATED"
                if self.logger:
                    self.logger.info(
                        f"[VAL GAMEOVER] round:{self.val_round_cnt} ep:{episode_idx + 1} steps:{step} "
                        f"result:{result_str} total_reward:{total_reward:.3f}"
                    )

                max_step = int(final_diag.get("max_step", Config.DEFAULT_MAX_STEP))
                completed = float(truncated and not terminated and final_diag.get("steps", step) >= max_step)
                abnormal = float(truncated and not terminated and final_diag.get("steps", step) < max_step)
                _ = completed, abnormal  # 目前先不单独上报，后面你要加也很容易。

                return {
                    "reward": total_reward,
                    "total_score": float(final_diag.get("total_score", 0.0)),
                    "steps": float(final_diag.get("steps", step)),
                    "treasures": float(final_diag.get("treasures", 0.0)),
                    "terminated_rate": float(terminated),
                }

        return None

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _build_episode_monitor_data(self, prefix, total_reward, terminated, truncated, step, final_diag):
        max_step = int(final_diag.get("max_step", Config.DEFAULT_MAX_STEP))
        completed = float(truncated and not terminated and final_diag.get("steps", step) >= max_step)
        abnormal = float(truncated and not terminated and final_diag.get("steps", step) < max_step)

        data = {
            f"{prefix}_reward": round(total_reward, 4),
            f"{prefix}_total_score": round(float(final_diag.get("total_score", 0.0)), 4),
            f"{prefix}_step_score": round(float(final_diag.get("step_score", 0.0)), 4),
            f"{prefix}_treasure_score": round(float(final_diag.get("treasure_score", 0.0)), 4),
            f"{prefix}_treasures": round(float(final_diag.get("treasures", 0.0)), 4),
            f"{prefix}_steps": round(float(final_diag.get("steps", step)), 4),
            f"{prefix}_flash_count": round(float(final_diag.get("flash_count", 0.0)), 4),
            f"{prefix}_terminated_rate": round(float(terminated), 4),
            f"{prefix}_completed_rate": round(completed, 4),
            f"{prefix}_abnormal_trunc": round(abnormal, 4),
            f"{prefix}_final_danger": round(float(final_diag.get("danger", 0.0)), 4),
            f"{prefix}_final_trea_dist": round(float(final_diag.get("best_treasure_dist", 0.0)), 4),
            f"{prefix}_last_flash_ready": round(float(final_diag.get("flash_ready", 0.0)), 4),
            f"{prefix}_final_visible_tre": round(float(final_diag.get("visible_treasures", 0.0)), 4),
        }
        return data


def _build_val_conf(train_conf):
    """
    从训练配置复制一份验证配置，只改地图。
    这样可保证：宝箱数、buff 数、CD、步数等验证配置和训练主配置一致。
    """
    val_conf = copy.deepcopy(train_conf)

    # read_usr_conf 返回的一般是 dict 结构，核心配置在 env_conf 里。
    env_conf = val_conf.get("env_conf", val_conf)
    env_conf["map"] = VAL_MAPS
    env_conf["map_random"] = VAL_MAP_RANDOM
    return val_conf

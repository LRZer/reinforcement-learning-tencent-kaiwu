#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config

ObsData = create_cls(
    "ObsData",
    hero=None,
    monster=None,
    treasure=None,
    buff=None,
    topo=None,
    memory=None,
    map_obs=None,
    legal_action=None,
)

ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# IMPORTANT:
# Keep SampleData replay-friendly. The framework expects fixed-size ndarray fields,
# not nested Python objects.
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


def flatten_obs(obs_data):
    return np.concatenate(
        [
            np.asarray(obs_data.hero, dtype=np.float32).reshape(-1),
            np.asarray(obs_data.monster, dtype=np.float32).reshape(-1),
            np.asarray(obs_data.treasure, dtype=np.float32).reshape(-1),
            np.asarray(obs_data.buff, dtype=np.float32).reshape(-1),
            np.asarray(obs_data.topo, dtype=np.float32).reshape(-1),
            np.asarray(obs_data.memory, dtype=np.float32).reshape(-1),
            np.asarray(obs_data.map_obs, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def sample_process(list_sample_data, gamma=0.99, lamda=0.95):
    if not list_sample_data:
        return list_sample_data

    for i in range(len(list_sample_data) - 1):
        done_flag = float(np.asarray(list_sample_data[i].done).reshape(-1)[0])
        next_value = np.asarray(list_sample_data[i + 1].value, dtype=np.float32)
        list_sample_data[i].next_value = ((1.0 - done_flag) * next_value).astype(np.float32)

    list_sample_data[-1].next_value = np.zeros_like(
        np.asarray(list_sample_data[-1].value, dtype=np.float32), dtype=np.float32
    )
    _calc_gae(list_sample_data, gamma=gamma, lamda=lamda)
    return list_sample_data


def _calc_gae(list_sample_data, gamma=0.99, lamda=0.95):
    gae = np.zeros_like(np.asarray(list_sample_data[0].value, dtype=np.float32), dtype=np.float32)
    for sample in reversed(list_sample_data):
        reward = np.asarray(sample.reward, dtype=np.float32)
        value = np.asarray(sample.value, dtype=np.float32)
        next_value = np.asarray(sample.next_value, dtype=np.float32)
        done_flag = float(np.asarray(sample.done).reshape(-1)[0])
        mask = 1.0 - done_flag
        delta = reward + gamma * next_value * mask - value
        gae = delta + gamma * lamda * mask * gae
        sample.advantage = gae.astype(np.float32)
        sample.reward_sum = (gae + value).astype(np.float32)

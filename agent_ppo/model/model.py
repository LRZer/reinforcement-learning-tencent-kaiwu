#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc(in_dim, out_dim, gain=1.0):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            make_fc(in_dim, hidden_dim),
            nn.ReLU(),
            make_fc(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        c = Config.MAP_CHANNELS
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(make_fc(64, Config.MAP_EMBED_DIM), nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.proj(x)


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_v1_flat_replay"
        self.device = device
        h = Config.BRANCH_HIDDEN_DIM

        self.hero_encoder = MLPEncoder(Config.HERO_DIM, h)
        self.monster_encoder = MLPEncoder(Config.MONSTER_DIM, h)
        self.treasure_encoder = MLPEncoder(Config.TREASURE_DIM, h)
        self.buff_encoder = MLPEncoder(Config.BUFF_DIM, h)
        self.topo_encoder = MLPEncoder(Config.TOPO_DIM, h)
        self.memory_encoder = MLPEncoder(Config.MEMORY_DIM, h)
        self.map_encoder = MapEncoder()

        gate_in = h + Config.MAP_EMBED_DIM
        self.type_gate = nn.Sequential(make_fc(gate_in, 32), nn.ReLU(), make_fc(32, 5))

        fusion_in = h + Config.MAP_EMBED_DIM + h
        self.fusion = nn.Sequential(
            make_fc(fusion_in, Config.FUSION_DIM),
            nn.ReLU(),
            make_fc(Config.FUSION_DIM, Config.FUSION_DIM),
            nn.ReLU(),
        )
        self.actor_head = make_fc(Config.FUSION_DIM, Config.ACTION_NUM, gain=0.01)
        self.critic_head = make_fc(Config.FUSION_DIM, Config.VALUE_NUM, gain=1.0)

    def _split_obs(self, obs):
        idx = 0
        hero = obs[:, idx: idx + Config.HERO_DIM]; idx += Config.HERO_DIM
        monster = obs[:, idx: idx + Config.MONSTER_DIM]; idx += Config.MONSTER_DIM
        treasure = obs[:, idx: idx + Config.TREASURE_DIM]; idx += Config.TREASURE_DIM
        buff = obs[:, idx: idx + Config.BUFF_DIM]; idx += Config.BUFF_DIM
        topo = obs[:, idx: idx + Config.TOPO_DIM]; idx += Config.TOPO_DIM
        memory = obs[:, idx: idx + Config.MEMORY_DIM]; idx += Config.MEMORY_DIM
        map_flat = obs[:, idx: idx + Config.MAP_FLAT_DIM]
        map_obs = map_flat.view(-1, Config.MAP_CHANNELS, Config.MAP_SIZE, Config.MAP_SIZE)
        return hero, monster, treasure, buff, topo, memory, map_obs

    def forward(self, obs, inference=False):
        hero_obs, monster_obs, treasure_obs, buff_obs, topo_obs, memory_obs, map_obs = self._split_obs(obs)

        hero_emb = self.hero_encoder(hero_obs)
        monster_emb = self.monster_encoder(monster_obs)
        treasure_emb = self.treasure_encoder(treasure_obs)
        buff_emb = self.buff_encoder(buff_obs)
        topo_emb = self.topo_encoder(topo_obs)
        memory_emb = self.memory_encoder(memory_obs)
        map_emb = self.map_encoder(map_obs)

        gate_logits = self.type_gate(torch.cat([hero_emb, map_emb], dim=1))
        gate = torch.softmax(gate_logits, dim=1)
        branches = torch.stack([monster_emb, treasure_emb, buff_emb, topo_emb, memory_emb], dim=1)
        context = (gate.unsqueeze(-1) * branches).sum(dim=1)

        fused = self.fusion(torch.cat([hero_emb, map_emb, context], dim=1))
        logits = self.actor_head(fused)
        value = self.critic_head(fused)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()

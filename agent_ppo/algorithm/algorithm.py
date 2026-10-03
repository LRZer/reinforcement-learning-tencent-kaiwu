#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import torch
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM
        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        if not list_sample_data:
            return

        # 当前工程里，样本字段是扁平 obs + 其他训练字段
        obs = torch.as_tensor(np.stack([f.obs for f in list_sample_data]), dtype=torch.float32, device=self.device)
        legal_action = torch.as_tensor(
            np.stack([f.legal_action for f in list_sample_data]), dtype=torch.float32, device=self.device
        )
        act = torch.as_tensor(np.stack([f.act for f in list_sample_data]), dtype=torch.long, device=self.device).view(-1, 1)

        # 这里保留整条旧策略分布，后面按动作 one-hot 提取 old_action_prob
        old_prob = torch.as_tensor(np.stack([f.prob for f in list_sample_data]), dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(np.stack([f.reward for f in list_sample_data]), dtype=torch.float32, device=self.device)
        advantage = torch.as_tensor(np.stack([f.advantage for f in list_sample_data]), dtype=torch.float32, device=self.device)
        old_value = torch.as_tensor(np.stack([f.value for f in list_sample_data]), dtype=torch.float32, device=self.device)
        reward_sum = torch.as_tensor(np.stack([f.reward_sum for f in list_sample_data]), dtype=torch.float32, device=self.device)

        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if advantage.dim() == 1:
            advantage = advantage.unsqueeze(-1)
        if old_value.dim() == 1:
            old_value = old_value.unsqueeze(-1)
        if reward_sum.dim() == 1:
            reward_sum = reward_sum.unsqueeze(-1)

        adv_mean = advantage.mean().item()
        adv_std = advantage.std(unbiased=False).item()
        ret_mean = reward_sum.mean().item()

        if Config.ADV_NORM:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False).clamp(min=1e-6))

        batch_size = obs.shape[0]
        mini_batch_size = min(getattr(Config, "PPO_MINI_BATCH_SIZE", batch_size), batch_size)
        ppo_epochs = max(1, int(getattr(Config, "PPO_EPOCHS", 1)))

        self.model.set_train_mode()

        total_loss_meter = []
        value_loss_meter = []
        policy_loss_meter = []
        entropy_loss_meter = []
        clip_frac_meter = []
        approx_kl_meter = []
        ratio_mean_meter = []
        ratio_std_meter = []
        explained_var_meter = []
        grad_norm_meter = []

        for _ in range(ppo_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start:start + mini_batch_size]
                mb_obs = obs[idx]
                mb_legal = legal_action[idx]
                mb_act = act[idx]
                mb_old_prob = old_prob[idx]
                mb_adv = advantage[idx]
                mb_old_value = old_value[idx]
                mb_reward_sum = reward_sum[idx]

                self.optimizer.zero_grad()
                logits, value_pred = self.model(mb_obs)
                total_loss, info = self._compute_loss(
                    logits=logits,
                    value_pred=value_pred,
                    legal_action=mb_legal,
                    old_action=mb_act,
                    old_prob=mb_old_prob,
                    advantage=mb_adv,
                    old_value=mb_old_value,
                    reward_sum=mb_reward_sum,
                )
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
                self.optimizer.step()

                total_loss_meter.append(float(total_loss.item()))
                value_loss_meter.append(float(info["value_loss"].item()))
                policy_loss_meter.append(float(info["policy_loss"].item()))
                entropy_loss_meter.append(float(info["entropy_loss"].item()))
                clip_frac_meter.append(float(info["clip_frac"]))
                approx_kl_meter.append(float(info["approx_kl"]))
                ratio_mean_meter.append(float(info["ratio_mean"]))
                ratio_std_meter.append(float(info["ratio_std"]))
                explained_var_meter.append(float(info["explained_var"]))
                grad_norm_meter.append(float(grad_norm))

        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "cum_reward": round(float(reward.mean().item()), 4),
                "total_loss": round(float(np.mean(total_loss_meter)), 4),
                "value_loss": round(float(np.mean(value_loss_meter)), 4),
                "policy_loss": round(float(np.mean(policy_loss_meter)), 4),
                "entropy_loss": round(float(np.mean(entropy_loss_meter)), 4),
                "grad_clip_norm": round(float(np.mean(grad_norm_meter)), 4),
                "clip_frac": round(float(np.mean(clip_frac_meter)), 4),
                "explained_var": round(float(np.mean(explained_var_meter)), 4),
                "approx_kl": round(float(np.mean(approx_kl_meter)), 6),
                "ratio_mean": round(float(np.mean(ratio_mean_meter)), 4),
                "ratio_std": round(float(np.mean(ratio_std_meter)), 4),
                "adv_mean": round(float(adv_mean), 4),
                "adv_std": round(float(adv_std), 4),
                "ret_mean": round(float(ret_mean), 4),
            }
            if self.logger:
                self.logger.info(f"[train] {monitor_data}")
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})
            self.last_report_monitor_time = now

    def _compute_loss(self, logits, value_pred, legal_action, old_action, old_prob, advantage, old_value, reward_sum):
        prob_dist = self._masked_softmax(logits, legal_action)

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True).clamp(1e-9)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        log_ratio = torch.log(new_prob) - torch.log(old_action_prob)
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantage
        surr2 = ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * advantage
        policy_loss = -torch.minimum(surr1, surr2).mean()
        clip_frac = float(((ratio > (1 + self.clip_param)) | (ratio < (1 - self.clip_param))).float().mean().item())
        approx_kl = float((old_action_prob.log() - new_prob.log()).mean().item())

        if Config.VALUE_CLIP:
            value_clip = old_value + (value_pred - old_value).clamp(-self.clip_param, self.clip_param)
            value_loss = 0.5 * torch.maximum((reward_sum - value_pred) ** 2, (reward_sum - value_clip) ** 2).mean()
        else:
            value_loss = 0.5 * ((reward_sum - value_pred) ** 2).mean()

        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1.0))).sum(1).mean()
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        y = reward_sum.detach().cpu().numpy().reshape(-1)
        y_hat = value_pred.detach().cpu().numpy().reshape(-1)
        var_y = np.var(y)
        explained_var = 0.0 if var_y < 1e-8 else 1.0 - np.var(y - y_hat) / (var_y + 1e-8)

        ratio_mean = float(ratio.mean().item())
        ratio_std = float(ratio.std(unbiased=False).item())
        return total_loss, {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "clip_frac": clip_frac,
            "approx_kl": approx_kl,
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "explained_var": explained_var,
        }

    @staticmethod
    def _masked_softmax(logits, legal_action):
        masked = logits - 1e10 * (1.0 - legal_action)
        return torch.nn.functional.softmax(masked, dim=1)

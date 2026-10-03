#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
监控面板：
1. 保留算法指标。
2. 保留训练指标。
3. 新增验证指标（val_*）。

注意：
- val_* 指标只有在 train_workflow.py 里真正跑了验证 episode 才会有值。
- 如果你只替换 monitor_builder.py，不替换 workflow，是不会自动出现 val 数据的。
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()
    config_dict = (
        monitor.title("峡谷追猎")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="当前回报", name_en="cum_reward", type="line")
            .add_metric(metrics_name="cum_reward", expr="avg(cum_reward{})").end_panel()
        .add_panel(name="总损失", name_en="total_loss", type="line")
            .add_metric(metrics_name="total_loss", expr="avg(total_loss{})").end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
            .add_metric(metrics_name="value_loss", expr="avg(value_loss{})").end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
            .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})").end_panel()
        .add_panel(name="熵", name_en="entropy_loss", type="line")
            .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})").end_panel()
        .add_panel(name="梯度范数", name_en="grad_clip_norm", type="line")
            .add_metric(metrics_name="grad_clip_norm", expr="avg(grad_clip_norm{})").end_panel()
        .add_panel(name="clip比例", name_en="clip_frac", type="line")
            .add_metric(metrics_name="clip_frac", expr="avg(clip_frac{})").end_panel()
        .add_panel(name="解释方差", name_en="explained_var", type="line")
            .add_metric(metrics_name="explained_var", expr="avg(explained_var{})").end_panel()
        .end_group()

        .add_group(group_name="训练表现", group_name_en="train")
        .add_panel(name="训练总奖励", name_en="train_reward", type="line")
            .add_metric(metrics_name="train_reward", expr="avg(train_reward{})").end_panel()
        .add_panel(name="训练总分", name_en="train_total_score", type="line")
            .add_metric(metrics_name="train_total_score", expr="avg(train_total_score{})").end_panel()
        .add_panel(name="训练步数", name_en="train_steps", type="line")
            .add_metric(metrics_name="train_steps", expr="avg(train_steps{})").end_panel()
        .add_panel(name="训练宝箱数", name_en="train_treasures", type="line")
            .add_metric(metrics_name="train_treasures", expr="avg(train_treasures{})").end_panel()
        .add_panel(name="训练终止率", name_en="train_terminated_rate", type="line")
            .add_metric(metrics_name="train_terminated_rate", expr="avg(train_terminated_rate{})").end_panel()
        .end_group()

        .add_group(group_name="验证表现", group_name_en="val")
        .add_panel(name="验证总奖励", name_en="val_reward", type="line")
            .add_metric(metrics_name="val_reward", expr="avg(val_reward{})").end_panel()
        .add_panel(name="验证总分", name_en="val_total_score", type="line")
            .add_metric(metrics_name="val_total_score", expr="avg(val_total_score{})").end_panel()
        .add_panel(name="验证步数", name_en="val_steps", type="line")
            .add_metric(metrics_name="val_steps", expr="avg(val_steps{})").end_panel()
        .add_panel(name="验证宝箱数", name_en="val_treasures", type="line")
            .add_metric(metrics_name="val_treasures", expr="avg(val_treasures{})").end_panel()
        .add_panel(name="验证终止率", name_en="val_terminated_rate", type="line")
            .add_metric(metrics_name="val_terminated_rate", expr="avg(val_terminated_rate{})").end_panel()
        .end_group()
        .build()
    )
    return config_dict

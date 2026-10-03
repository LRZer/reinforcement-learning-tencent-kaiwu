#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
改进版 preprocessor：
1. 保持现有 observation 分支维度不变，不联动修改模型结构。
2. 加入局部 21x21 BFS 特征：
   - 可见宝箱/缓存宝箱的局部最短路距离
   - 可见/缓存 buff 的局部最短路距离
   - 8 个方向的可达区域质量（BFS 版本）
3. 加入宝箱/buff 缓存：
   - 宝箱缓存：看见就记住，吃到后清理英雄附近
   - buff 缓存：看见后永久记住位置，因为 buff 会刷新
4. 加入“防磨蹭奖励”：
   - 回看前 10 步位置
   - 如果当前仍然离 10 步前很近，并且危险不高，则判定为磨蹭

注意：
- 这一版重点是让智能体更会“走得通”和“记得住”，而不是继续堆复杂 reward。
- 因为维度保持不变，所以主要修改集中在 conf.py 和 preprocessor.py。
"""

from collections import deque
import math
import numpy as np
from agent_ppo.conf.conf import Config


class Preprocessor:
    def __init__(self):
        self.reset()

    # ==================================================
    # Lifecycle
    # ==================================================
    def reset(self):
        self.step_no = 0
        self.prev_step_score = 0.0
        self.prev_treasure_score = 0.0
        self.prev_total_score = 0.0
        self.prev_treasures_collected = 0
        self.prev_buff_collected = 0

        self.prev_min_monster_dist = None
        self.prev_min_monster_path_dist = None
        self.prev_best_treasure_dist = None
        self.prev_best_buff_dist = None
        self.prev_danger = 0.0
        self.prev_openness = 0.0
        self.prev_squeeze = 0.0
        self.prev_pos = None

        self.flash_cd_remain = 0
        self.flash_cd_cfg = Config.DEFAULT_FLASH_CD
        self.buff_timer = 0

        # 全局记忆图
        self.visit_heat = np.zeros((128, 128), dtype=np.float32)
        self.explored_mask = np.zeros((128, 128), dtype=np.float32)
        self.treasure_memory = np.zeros((128, 128), dtype=np.float32)
        self.buff_memory = np.zeros((128, 128), dtype=np.float32)

        self.recent_positions = deque(maxlen=Config.RECENT_POS_QUEUE)
        self.recent_actions = deque(maxlen=Config.RECENT_ACTION_QUEUE)
        self.recent_progress = deque(maxlen=Config.RECENT_PROGRESS_WINDOW)

    # ==================================================
    # Main entry
    # ==================================================
    def feature_process(self, env_obs, last_action):
        self.step_no += 1
        self.recent_actions.append(int(last_action) if last_action is not None else -1)

        observation = self._pick(env_obs, "observation", env_obs)
        frame_state = self._pick(observation, "frame_state", self._pick(env_obs, "frame_state", {}))
        env_info = self._pick(observation, "env_info", self._pick(env_obs, "env_info", {}))
        map_info = self._pick(
            observation,
            "map_info",
            np.ones((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int32),
        )
        legal_action = self._pick(
            observation,
            "legal_act",
            self._pick(observation, "legal_action", [1] * Config.ACTION_NUM),
        )

        hero_state = self._extract_hero(frame_state)
        hero_pos = self._extract_pos(self._pick(hero_state, "pos", {}))
        monsters = [self._normalize_monster(m) for m in self._as_list(self._pick(frame_state, "monsters", []))]
        organs = [self._normalize_organ(o) for o in self._as_list(self._pick(frame_state, "organs", []))]

        max_step = int(self._pick(env_info, "max_step", Config.DEFAULT_MAX_STEP) or Config.DEFAULT_MAX_STEP)
        finished_steps = int(
            self._pick(env_info, "finished_steps", self._pick(observation, "step_no", self.step_no)) or self.step_no
        )
        step_score = float(
            self._pick(env_info, "step_score", self._pick(hero_state, "step_score", self.prev_step_score)) or 0.0
        )
        treasure_score = float(
            self._pick(env_info, "treasure_score", self._pick(hero_state, "treasure_score", self.prev_treasure_score))
            or 0.0
        )
        total_score = float(self._pick(env_info, "total_score", step_score + treasure_score) or 0.0)
        treasures_collected = int(
            self._pick(env_info, "treasures_collected", self._pick(hero_state, "treasure_collected_count", 0)) or 0
        )
        flash_count = int(self._pick(env_info, "flash_count", 0) or 0)
        buff_collected = int(self._pick(env_info, "collected_buff", 0) or 0)

        # 闪现 CD 跟踪
        self.flash_cd_cfg = int(self._pick(env_info, "flash_cooldown", self.flash_cd_cfg) or self.flash_cd_cfg)
        self.flash_cd_remain = max(0, self.flash_cd_remain - 1)
        if last_action is not None and 8 <= int(last_action) < 16:
            self.flash_cd_remain = self.flash_cd_cfg
        if len(legal_action) >= 16 and any(bool(x) for x in legal_action[8:16]):
            self.flash_cd_remain = 0

        # buff 持续时间跟踪
        self.buff_timer = max(0, self.buff_timer - 1)
        if buff_collected > self.prev_buff_collected:
            self.buff_timer = Config.DEFAULT_BUFF_DURATION

        visible_treasures = [o for o in organs if o["sub_type"] == 1 and o["status"] == 1]
        visible_buffs = [o for o in organs if o["sub_type"] == 2 and o["status"] == 1]

        treasure_gained = (treasure_score > self.prev_treasure_score) or (
            treasures_collected > self.prev_treasures_collected
        )

        # 更新全局探索 / 记忆缓存
        self._reveal_exploration(hero_pos)
        self._update_treasure_memory(visible_treasures, hero_pos, treasure_gained)
        self._update_buff_memory(visible_buffs)
        self._update_visit_heat(hero_pos)

        # 先建立局部 traversable 和 BFS，后续多个分支共用
        traversable_local = self._make_local_traversable(map_info)
        bfs_dist = self._bfs_from_center(traversable_local)

        # 构造多分支特征
        map_tensor, topo_summary = self._build_map_tensor(
            traversable_local, hero_pos, monsters, visible_treasures, visible_buffs
        )
        hero_feat = self._build_hero_feat(
            hero_pos, step_score, treasure_score, total_score, treasures_collected, finished_steps, max_step, monsters
        )
        monster_feat, monster_aux = self._build_monster_feat(monsters, hero_pos, bfs_dist)
        treasure_feat, treasure_aux = self._build_treasure_feat(visible_treasures, hero_pos, topo_summary, bfs_dist)
        buff_feat, buff_aux = self._build_buff_feat(visible_buffs, hero_pos, flash_count, bfs_dist)
        topo_feat = self._build_topo_feat(traversable_local, topo_summary)
        memory_feat, memory_aux = self._build_memory_feat(hero_pos)

        self._assert_dims(hero_feat, monster_feat, treasure_feat, buff_feat, topo_feat, memory_feat, map_tensor)

        reward, reward_detail = self._compute_reward(
            env_obs=env_obs,
            hero_pos=hero_pos,
            step_score=step_score,
            treasure_score=treasure_score,
            total_score=total_score,
            treasures_collected=treasures_collected,
            buff_collected=buff_collected,
            monster_aux=monster_aux,
            treasure_aux=treasure_aux,
            buff_aux=buff_aux,
            topo_summary=topo_summary,
            memory_aux=memory_aux,
            last_action=last_action,
            max_step=max_step,
            finished_steps=finished_steps,
        )

        diag = {
            "reward": float(reward[0]),
            "total_score": total_score,
            "step_score": step_score,
            "treasure_score": treasure_score,
            "treasures": treasures_collected,
            "steps": finished_steps,
            "flash_count": flash_count,
            "flash_ready": float(self.flash_cd_remain == 0),
            "visible_treasures": len(visible_treasures),
            "visible_buffs": len(visible_buffs),
            "danger": float(monster_aux["danger"]),
            "min_monster_path_dist": float(monster_aux["min_path_dist"]),
            "best_treasure_dist": float(treasure_aux["best_dist"]),
            "best_treasure_bfs": float(treasure_aux["best_bfs"]),
            "best_buff_bfs": float(buff_aux["best_bfs"]),
            "openness": float(topo_summary["openness"]),
            "revisit_ratio": float(memory_aux["revisit_ratio"]),
            "loiter_flag": float(memory_aux["loiter_flag"]),
            "reward_detail": reward_detail,
            "max_step": max_step,
            "terminated": bool(self._pick(env_obs, "terminated", False)),
            "truncated": bool(self._pick(env_obs, "truncated", False)),
        }

        # 状态回写
        self.prev_step_score = step_score
        self.prev_treasure_score = treasure_score
        self.prev_total_score = total_score
        self.prev_treasures_collected = treasures_collected
        self.prev_buff_collected = buff_collected
        self.prev_min_monster_dist = monster_aux["min_dist"]
        self.prev_min_monster_path_dist = monster_aux["min_path_dist"]
        self.prev_best_treasure_dist = treasure_aux["best_dist"]
        self.prev_best_buff_dist = buff_aux["best_dist"]
        self.prev_danger = monster_aux["danger"]
        self.prev_openness = topo_summary["openness"]
        self.prev_squeeze = monster_aux["squeeze"]
        self.prev_pos = hero_pos

        legal_action = self._normalize_legal_action(legal_action)
        return (
            hero_feat,
            monster_feat,
            treasure_feat,
            buff_feat,
            topo_feat,
            memory_feat,
            map_tensor,
            legal_action,
            reward,
            diag,
        )

    # ==================================================
    # Feature builders
    # ==================================================
    def _assert_dims(self, hero_feat, monster_feat, treasure_feat, buff_feat, topo_feat, memory_feat, map_tensor):
        assert hero_feat.shape[0] == Config.HERO_DIM
        assert monster_feat.shape[0] == Config.MONSTER_DIM
        assert treasure_feat.shape[0] == Config.TREASURE_DIM
        assert buff_feat.shape[0] == Config.BUFF_DIM
        assert topo_feat.shape[0] == Config.TOPO_DIM
        assert memory_feat.shape[0] == Config.MEMORY_DIM
        assert map_tensor.shape == (Config.MAP_CHANNELS, Config.MAP_SIZE, Config.MAP_SIZE)

    def _build_hero_feat(self, hero_pos, step_score, treasure_score, total_score, treasures_collected, finished_steps, max_step, monsters):
        x, z = hero_pos
        progress = min(1.0, finished_steps / max(max_step, 1))
        remain_ratio = 1.0 - progress
        speed = 2.0 if self.buff_timer > 0 else 1.0
        monster2_alive = 1.0 if len(monsters) >= 2 else 0.0
        high_pressure = 1.0 if any(m["speed"] > 1 for m in monsters) else 0.0
        speedup_progress = min(1.0, finished_steps / max(Config.DEFAULT_MONSTER_SPEEDUP_STEP, 1))
        return np.array(
            [
                x / 127.0,
                z / 127.0,
                progress,
                remain_ratio,
                min(total_score / 2500.0, 1.5),
                min(step_score / 1500.0, 1.5),
                min(treasure_score / 1000.0, 1.5),
                treasures_collected / max(Config.MAX_TREASURE_COUNT, 1),
                1.0 if self.flash_cd_remain == 0 else 0.0,
                min(self.flash_cd_remain / max(self.flash_cd_cfg, 1), 1.0),
                speed / 2.0,
                monster2_alive,
                high_pressure,
                speedup_progress,
            ],
            dtype=np.float32,
        )

    def _build_monster_feat(self, monsters, hero_pos, bfs_dist):
        hero_x, hero_z = hero_pos
        ordered = []
        path_dists = []
        for m in monsters:
            mx, mz = m["pos"]
            dx, dz = mx - hero_x, mz - hero_z
            dist = math.sqrt(dx * dx + dz * dz)
            path_dist = self._lookup_bfs(bfs_dist, self._global_to_local(hero_pos, (mx, mz)))
            in_view = 1.0 if max(abs(dx), abs(dz)) <= 10 else 0.0
            ordered.append({**m, "dist": dist, "path_dist": path_dist, "in_view": in_view, "dx": dx, "dz": dz})
            path_dists.append(path_dist)
        ordered.sort(key=lambda x: x["dist"])

        feat = []
        for i in range(2):
            if i < len(ordered):
                m = ordered[i]
                feat.extend(
                    [
                        1.0,
                        m["in_view"],
                        np.clip(m["dx"] / 10.0, -1.0, 1.0),
                        np.clip(m["dz"] / 10.0, -1.0, 1.0),
                        min(m["dist"] / 30.0, 1.5),
                        min(m["speed"] / 2.0, 1.5),
                        min(m["hero_relative_direction"] / 8.0, 1.0),
                        min(m["hero_l2_distance"] / 5.0, 1.0),
                    ]
                )
            else:
                feat.extend([0.0] * 8)

        min_dist = ordered[0]["dist"] if ordered else 30.0
        second_dist = ordered[1]["dist"] if len(ordered) > 1 else 30.0
        min_path_dist = min(path_dists) if path_dists else Config.BFS_DIST_NORM
        squeeze = 1.0 - min(1.0, (min_dist + second_dist) / 40.0)
        danger = self._danger_score(min_dist=min_dist, second_dist=second_dist, monsters=ordered)
        aux = {
            "min_dist": float(min_dist),
            "second_dist": float(second_dist),
            "min_path_dist": float(min_path_dist),
            "danger": float(danger),
            "squeeze": float(squeeze),
        }
        return np.asarray(feat, dtype=np.float32), aux

    def _build_treasure_feat(self, visible_treasures, hero_pos, topo_summary, bfs_dist):
        """
        TREASURE_DIM = 14，保持不变。
        结构：
        0: 当前可见宝箱占比
        1: 记忆中宝箱占比（粗略）
        2~6: 最优可见宝箱 [exists, dx, dz, bfs_dist, value]
        7~11: 最优缓存宝箱 [exists, dx, dz, dist, memory_score]
        12: 最近可见宝箱 BFS 距离
        13: 最近 frontier/未知区域的近似距离（用来帮助探索）
        """
        hero_x, hero_z = hero_pos
        visible_candidates = []
        for t in visible_treasures:
            tx, tz = t["pos"]
            dx, dz = tx - hero_x, tz - hero_z
            dist = math.sqrt(dx * dx + dz * dz)
            local_coord = self._global_to_local(hero_pos, (tx, tz))
            bfs_v = self._lookup_bfs(bfs_dist, local_coord)
            straight = max(abs(dx), abs(dz))
            value = 1.0 / (dist + 1.0) + 0.15 * topo_summary["openness"] - 0.03 * straight
            visible_candidates.append({
                "dx": dx, "dz": dz, "dist": dist, "bfs": bfs_v, "value": value,
            })
        visible_candidates.sort(key=lambda x: (x["bfs"], -x["value"]))

        # 记忆中的宝箱：不限于当前视野
        mem_positions = np.argwhere(self.treasure_memory > 0.35)
        mem_candidates = []
        for row, col in mem_positions:
            tx, tz = int(col), int(row)
            dx, dz = tx - hero_x, tz - hero_z
            dist = math.sqrt(dx * dx + dz * dz)
            mem_score = float(self.treasure_memory[tz, tx])
            mem_candidates.append({"dx": dx, "dz": dz, "dist": dist, "score": mem_score})
        mem_candidates.sort(key=lambda x: x["dist"])

        visible_ratio = min(len(visible_treasures) / Config.MAX_TREASURE_COUNT, 1.0)
        memory_ratio = min(len(mem_candidates) / Config.MAX_TREASURE_COUNT, 1.0)
        feat = [visible_ratio, memory_ratio]

        # slot1: 最优可见宝箱（用 BFS 距离）
        if visible_candidates:
            c = visible_candidates[0]
            feat.extend([
                1.0,
                np.clip(c["dx"] / 10.0, -1.0, 1.0),
                np.clip(c["dz"] / 10.0, -1.0, 1.0),
                min(c["bfs"] / Config.BFS_DIST_NORM, 1.5),
                np.clip(c["value"], -1.0, 1.5),
            ])
            best_visible_bfs = c["bfs"]
            best_visible_dist = c["dist"]
        else:
            feat.extend([0.0] * 5)
            best_visible_bfs = Config.BFS_DIST_NORM
            best_visible_dist = 15.0

        # slot2: 最优缓存宝箱（不一定在视野内，用全局记忆位置 + 欧式距离）
        if mem_candidates:
            c = mem_candidates[0]
            feat.extend([
                1.0,
                np.clip(c["dx"] / 20.0, -1.0, 1.0),
                np.clip(c["dz"] / 20.0, -1.0, 1.0),
                min(c["dist"] / Config.TARGET_DIST_NORM, 1.5),
                np.clip(c["score"], 0.0, 1.0),
            ])
        else:
            feat.extend([0.0] * 5)

        frontier_dist = self._nearest_frontier_dist(hero_pos)
        feat.extend([
            min(best_visible_bfs / Config.BFS_DIST_NORM, 1.5),
            min(frontier_dist / Config.TARGET_DIST_NORM, 1.5),
        ])

        aux = {
            "best_dist": float(best_visible_dist),
            "best_bfs": float(best_visible_bfs),
            "visible_count": len(visible_treasures),
        }
        return np.asarray(feat, dtype=np.float32), aux

    def _build_buff_feat(self, visible_buffs, hero_pos, flash_count, bfs_dist):
        """
        BUFF_DIM = 8，保持不变。
        使用可见 buff + 缓存 buff 的最近目标。
        """
        hero_x, hero_z = hero_pos
        best_visible = None
        for b in visible_buffs:
            bx, bz = b["pos"]
            dx, dz = bx - hero_x, bz - hero_z
            dist = math.sqrt(dx * dx + dz * dz)
            local_coord = self._global_to_local(hero_pos, (bx, bz))
            bfs_v = self._lookup_bfs(bfs_dist, local_coord)
            item = {"dx": dx, "dz": dz, "dist": dist, "bfs": bfs_v}
            if best_visible is None or item["bfs"] < best_visible["bfs"]:
                best_visible = item

        # 缓存 buff：看见过后一直保留，便于学出循环吃 buff 路线
        mem_positions = np.argwhere(self.buff_memory > 0.5)
        best_memory = None
        for row, col in mem_positions:
            bx, bz = int(col), int(row)
            dx, dz = bx - hero_x, bz - hero_z
            dist = math.sqrt(dx * dx + dz * dz)
            item = {"dx": dx, "dz": dz, "dist": dist}
            if best_memory is None or item["dist"] < best_memory["dist"]:
                best_memory = item

        # 目标优先：当前可见 buff > 缓存 buff
        target = best_visible if best_visible is not None else best_memory
        if target is None:
            has_target = 0.0
            dx_n = dz_n = 0.0
            dist_n = 1.0
            best_dist = Config.TARGET_DIST_NORM
            best_bfs = Config.BFS_DIST_NORM
        else:
            has_target = 1.0
            dx_n = np.clip(target["dx"] / 20.0, -1.0, 1.0)
            dz_n = np.clip(target["dz"] / 20.0, -1.0, 1.0)
            dist_n = min(target["dist"] / Config.TARGET_DIST_NORM, 1.5)
            best_dist = target["dist"]
            best_bfs = target.get("bfs", Config.BFS_DIST_NORM)

        feat = [
            1.0 if self.buff_timer > 0 else 0.0,
            min(self.buff_timer / Config.DEFAULT_BUFF_DURATION, 1.0),
            has_target,
            dx_n,
            dz_n,
            dist_n,
            1.0 if (len(self.recent_actions) > 0 and self.recent_actions[-1] >= 8) else 0.0,
            min(flash_count / 20.0, 1.0),
        ]
        aux = {"has_target": float(has_target), "best_dist": float(best_dist), "best_bfs": float(best_bfs)}
        return np.asarray(feat, dtype=np.float32), aux

    def _build_topo_feat(self, traversable_local, topo_summary):
        """
        TOPO_DIM = 13，保持不变。
        前 5 维保持旧摘要：
          openness / obstacle_density / branch_count / dead_end_risk / near_wall
        后 8 维改成“按方向 BFS 质量打分”：
          对 8 个方向，计算若第一步走向该方向，能到达的区域比例。
        这样比单纯射线长度更接近“往哪走更通”。
        """
        dir_scores = self._directional_bfs_scores(traversable_local)
        return np.asarray(
            [
                topo_summary["openness"],
                topo_summary["obstacle_density"],
                topo_summary["branch_count"],
                topo_summary["dead_end_risk"],
                topo_summary["near_wall"],
                *dir_scores,
            ],
            dtype=np.float32,
        )

    def _build_memory_feat(self, hero_pos):
        x, z = hero_pos
        self.recent_positions.append((x, z))

        move_dist = 0.0 if self.prev_pos is None else math.sqrt((x - self.prev_pos[0]) ** 2 + (z - self.prev_pos[1]) ** 2)
        unique_ratio = len(set(self.recent_positions)) / max(len(self.recent_positions), 1)
        revisit_ratio = 1.0 - unique_ratio
        stuck_flag = 1.0 if move_dist < 0.1 and self.prev_pos is not None else 0.0

        self.recent_progress.append(move_dist)
        no_progress = 1.0 if len(self.recent_progress) == self.recent_progress.maxlen and sum(self.recent_progress) < 2.0 else 0.0

        target_dx, target_dz, target_dist, target_kind, unexplored_local_ratio = self._explore_target_features(hero_pos)
        loiter_flag = 1.0 if self._is_loitering() else 0.0

        feat = np.asarray(
            [
                revisit_ratio,
                unique_ratio,
                stuck_flag,
                min(move_dist / 2.0, 1.0),
                target_dx,
                target_dz,
                unexplored_local_ratio,
                no_progress,
            ],
            dtype=np.float32,
        )

        aux = {
            "target_dist": float(target_dist),
            "target_kind": float(target_kind),
            "revisit_ratio": float(revisit_ratio),
            "no_progress": float(no_progress),
            "unexplored_local_ratio": float(unexplored_local_ratio),
            "loiter_flag": float(loiter_flag),
        }
        return feat, aux

    # ==================================================
    # Map / topology / memory
    # ==================================================
    def _build_map_tensor(self, traversable_local, hero_pos, monsters, visible_treasures, visible_buffs):
        traversable = traversable_local.astype(np.float32)
        obstacle = 1.0 - traversable

        hero_ch = np.zeros_like(traversable)
        center = Config.MAP_SIZE // 2
        hero_ch[center, center] = 1.0

        monster_ch = np.zeros_like(traversable)
        treasure_ch = np.zeros_like(traversable)
        buff_ch = np.zeros_like(traversable)

        hx, hz = hero_pos

        for m in monsters:
            mx, mz = m["pos"]
            lx, lz = mx - hx + center, mz - hz + center
            if 0 <= lz < Config.MAP_SIZE and 0 <= lx < Config.MAP_SIZE:
                monster_ch[lz, lx] = 1.0

        for t in visible_treasures:
            tx, tz = t["pos"]
            lx, lz = tx - hx + center, tz - hz + center
            if 0 <= lz < Config.MAP_SIZE and 0 <= lx < Config.MAP_SIZE:
                treasure_ch[lz, lx] = 1.0

        for b in visible_buffs:
            bx, bz = b["pos"]
            lx, lz = bx - hx + center, bz - hz + center
            if 0 <= lz < Config.MAP_SIZE and 0 <= lx < Config.MAP_SIZE:
                buff_ch[lz, lx] = 1.0

        visit_local = self._crop_global(self.visit_heat, hero_pos, fill=0.0)

        danger_local = np.zeros_like(traversable)
        for m in monsters:
            mx, mz = m["pos"]
            lx, lz = mx - hx + center, mz - hz + center
            for dz in range(-3, 4):
                for dx in range(-3, 4):
                    yy, xx = lz + dz, lx + dx
                    if 0 <= yy < Config.MAP_SIZE and 0 <= xx < Config.MAP_SIZE:
                        d = math.sqrt(dx * dx + dz * dz)
                        danger_local[yy, xx] = max(danger_local[yy, xx], math.exp(-0.8 * d))

        stacked = np.stack(
            [traversable, obstacle, hero_ch, monster_ch, treasure_ch, buff_ch, visit_local, danger_local],
            axis=0,
        )
        topo = self._analyze_topology(traversable)
        return stacked.astype(np.float32), topo

    def _explore_target_features(self, hero_pos):
        """
        目标优先级：
        1. 已见未吃宝箱
        2. 已见 buff 点（尤其在没有宝箱记忆时）
        3. frontier（未知区域）

        这里只把目标信息作为 feature，不给额外重 reward。
        """
        x, z = hero_pos
        unexplored_local = 1.0 - self._crop_global(self.explored_mask, hero_pos, fill=0.0)
        unexplored_local_ratio = float(np.mean(unexplored_local))

        remembered_t = np.argwhere(self.treasure_memory > 0.35)
        if remembered_t.shape[0] > 0:
            dx_all = remembered_t[:, 1] - x
            dz_all = remembered_t[:, 0] - z
            dist2 = dx_all * dx_all + dz_all * dz_all
            idx = int(np.argmin(dist2))
            tx, tz = int(remembered_t[idx, 1]), int(remembered_t[idx, 0])
            dx, dz = tx - x, tz - z
            dist = math.sqrt(dx * dx + dz * dz)
            return (
                float(np.clip(dx / Config.TARGET_DIST_NORM, -1.0, 1.0)),
                float(np.clip(dz / Config.TARGET_DIST_NORM, -1.0, 1.0)),
                float(min(dist / Config.TARGET_DIST_NORM, 2.0)),
                2.0,
                unexplored_local_ratio,
            )

        remembered_b = np.argwhere(self.buff_memory > 0.5)
        if remembered_b.shape[0] > 0:
            dx_all = remembered_b[:, 1] - x
            dz_all = remembered_b[:, 0] - z
            dist2 = dx_all * dx_all + dz_all * dz_all
            idx = int(np.argmin(dist2))
            bx, bz = int(remembered_b[idx, 1]), int(remembered_b[idx, 0])
            dx, dz = bx - x, bz - z
            dist = math.sqrt(dx * dx + dz * dz)
            return (
                float(np.clip(dx / Config.TARGET_DIST_NORM, -1.0, 1.0)),
                float(np.clip(dz / Config.TARGET_DIST_NORM, -1.0, 1.0)),
                float(min(dist / Config.TARGET_DIST_NORM, 2.0)),
                1.0,
                unexplored_local_ratio,
            )

        frontier_dist = self._nearest_frontier_dist(hero_pos)
        if frontier_dist < 999.0:
            # 用最近未知点近似一个 frontier 方向
            unexplored = np.argwhere(self.explored_mask < 0.5)
            dx_all = unexplored[:, 1] - x
            dz_all = unexplored[:, 0] - z
            dist2 = dx_all * dx_all + dz_all * dz_all
            idx = int(np.argmin(dist2))
            fx, fz = int(unexplored[idx, 1]), int(unexplored[idx, 0])
            dx, dz = fx - x, fz - z
            dist = math.sqrt(dx * dx + dz * dz)
            return (
                float(np.clip(dx / Config.TARGET_DIST_NORM, -1.0, 1.0)),
                float(np.clip(dz / Config.TARGET_DIST_NORM, -1.0, 1.0)),
                float(min(dist / Config.TARGET_DIST_NORM, 2.0)),
                0.0,
                unexplored_local_ratio,
            )

        return 0.0, 0.0, 1.0, 0.0, unexplored_local_ratio

    def _reveal_exploration(self, hero_pos):
        x, z = hero_pos
        r = Config.MAP_SIZE // 2
        x0, x1 = max(0, x - r), min(128, x + r + 1)
        z0, z1 = max(0, z - r), min(128, z + r + 1)
        self.explored_mask[z0:z1, x0:x1] = 1.0

    def _update_treasure_memory(self, visible_treasures, hero_pos, treasure_gained):
        self.treasure_memory *= Config.TREASURE_MEMORY_DECAY
        for t in visible_treasures:
            tx, tz = t["pos"]
            if 0 <= tz < 128 and 0 <= tx < 128:
                self.treasure_memory[tz, tx] = 1.0

        # 吃到箱子后，清理英雄附近一圈宝箱记忆
        if treasure_gained:
            x, z = hero_pos
            for dz in range(-1, 2):
                for dx in range(-1, 2):
                    xx, zz = x + dx, z + dz
                    if 0 <= zz < 128 and 0 <= xx < 128:
                        self.treasure_memory[zz, xx] = 0.0

    def _update_buff_memory(self, visible_buffs):
        """
        buff 点位缓存：
        - 看见一次就记住
        - 不主动清理，因为 buff 会刷新
        """
        for b in visible_buffs:
            bx, bz = b["pos"]
            if 0 <= bz < 128 and 0 <= bx < 128:
                self.buff_memory[bz, bx] = 1.0

    def _update_visit_heat(self, hero_pos):
        self.visit_heat *= Config.REVISIT_DECAY
        x, z = hero_pos
        if 0 <= z < self.visit_heat.shape[0] and 0 <= x < self.visit_heat.shape[1]:
            self.visit_heat[z, x] = min(1.0, self.visit_heat[z, x] + 0.35)

    def _is_loitering(self):
        """
        防磨蹭版本：
        - 回看前 LOITER_WINDOW 步的位置
        - 如果当前仍然离那个位置很近，则视为在小范围磨蹭/摩擦
        """
        if len(self.recent_positions) < Config.LOITER_WINDOW:
            return False

        current = self.recent_positions[-1]
        past = self.recent_positions[-Config.LOITER_WINDOW]
        dx = current[0] - past[0]
        dz = current[1] - past[1]
        l2 = math.sqrt(dx * dx + dz * dz)
        return l2 < Config.LOITER_L2_THRESHOLD

    # ==================================================
    # Reward
    # ==================================================
    def _compute_reward(
        self,
        env_obs,
        hero_pos,
        step_score,
        treasure_score,
        total_score,
        treasures_collected,
        buff_collected,
        monster_aux,
        treasure_aux,
        buff_aux,
        topo_summary,
        memory_aux,
        last_action,
        max_step,
        finished_steps,
    ):
        step_gain = max(0.0, step_score - self.prev_step_score)
        treasure_gain = max(0.0, treasure_score - self.prev_treasure_score)

        min_dist = monster_aux["min_dist"]
        min_path_dist = monster_aux["min_path_dist"]
        danger = monster_aux["danger"]
        squeeze = monster_aux["squeeze"]
        openness = topo_summary["openness"]
        dead_end_risk = topo_summary["dead_end_risk"]
        near_wall = topo_summary["near_wall"]

        reward = 0.0
        detail = {}

        # 1) 基础步数奖励
        r_step = Config.REWARD_STEP_GAIN_SCALE * step_gain
        reward += r_step
        detail["step_gain"] = r_step

        # 2) 宝箱分奖励
        r_treasure = Config.REWARD_TREASURE_GAIN_SCALE * (treasure_gain / 100.0)
        reward += r_treasure
        detail["treasure_gain"] = r_treasure

        # 3) 吃 buff 奖励
        if buff_collected > self.prev_buff_collected:
            reward += Config.REWARD_BUFF_PICK
            detail["buff_pick"] = Config.REWARD_BUFF_PICK
        else:
            detail["buff_pick"] = 0.0

        # 4) 生存 shaping：和怪拉开距离
        # Buff approach shaping. Keep it small and conditional so buff stays a tool, not the only goal.
        best_buff_dist = buff_aux["best_dist"]
        if (
            self.prev_best_buff_dist is not None
            and buff_aux["has_target"] > 0.5
            and self.buff_timer <= 0
            and buff_collected <= self.prev_buff_collected
            and treasure_gain <= 0.0
            and danger < Config.BUFF_APPROACH_DANGER_TH
            and best_buff_dist <= Config.BUFF_APPROACH_MAX_DIST
            and treasure_aux["best_dist"] > Config.BUFF_APPROACH_TREASURE_DIST_TH
        ):
            buff_delta = np.clip((self.prev_best_buff_dist - best_buff_dist) / 3.0, -1.0, 1.0)
            r_buff_approach = Config.REWARD_BUFF_APPROACH_SCALE * buff_delta
            reward += r_buff_approach
            detail["buff_approach"] = r_buff_approach
        else:
            detail["buff_approach"] = 0.0

        dist_delta = 0.0
        if self.prev_min_monster_dist is not None:
            dist_delta = np.clip((min_dist - self.prev_min_monster_dist) / 4.0, -1.0, 1.0)

        pressure_mult = 1.3 if (finished_steps >= Config.DEFAULT_MONSTER_SPEEDUP_STEP or danger > 0.55) else 1.0
        r_safe = Config.REWARD_SAFE_DELTA_SCALE * dist_delta * pressure_mult
        reward += r_safe
        detail["safe_delta"] = r_safe

        # 5) danger shaping
        danger_delta = self.prev_danger - danger
        r_danger = Config.REWARD_DANGER_SCALE * danger_delta * pressure_mult
        reward += r_danger
        detail["danger_delta"] = r_danger

        # 6) 双怪压缩 shaping
        squeeze_delta = self.prev_squeeze - squeeze
        r_squeeze = Config.REWARD_SQUEEZE_SCALE * squeeze_delta * pressure_mult
        reward += r_squeeze
        detail["squeeze_delta"] = r_squeeze

        # 7) 轻量地形 shaping
        topo_pressure = max(danger, 0.25)
        r_deadend = -Config.REWARD_DEADEND_SCALE * dead_end_risk * topo_pressure
        reward += r_deadend
        detail["dead_end"] = r_deadend

        r_wall = -0.02 * near_wall * topo_pressure
        reward += r_wall
        detail["near_wall"] = r_wall

        r_open = Config.REWARD_OPENNESS_SCALE * max(openness - 0.35, 0.0) * topo_pressure
        reward += r_open
        detail["openness"] = r_open

        # 8) 安全时鼓励靠近宝箱
        best_treasure_dist = treasure_aux["best_dist"]
        if (
            self.prev_best_treasure_dist is not None
            and treasure_gain <= 0.0
            and danger < Config.TREASURE_SAFE_DANGER_TH
        ):
            approach_delta = np.clip((self.prev_best_treasure_dist - best_treasure_dist) / 3.0, -1.0, 1.0)
            r_approach = Config.REWARD_TREASURE_APPROACH_SCALE * approach_delta
            reward += r_approach
            detail["treasure_approach"] = r_approach
        else:
            detail["treasure_approach"] = 0.0

        # 9) 本步位移
        moved = 0.0 if self.prev_pos is None else math.sqrt(
            (hero_pos[0] - self.prev_pos[0]) ** 2 + (hero_pos[1] - self.prev_pos[1]) ** 2
        )

        # 10) 撞墙/近乎无位移惩罚
        if last_action is not None and int(last_action) >= 0 and moved < 0.1:
            reward += Config.REWARD_INVALID_MOVE
            detail["invalid_move"] = Config.REWARD_INVALID_MOVE
        else:
            detail["invalid_move"] = 0.0

        # 11) 防磨蹭惩罚
        if (
            danger < Config.LOITER_DANGER_TH
            and treasure_gain <= 0.0
            and buff_collected <= self.prev_buff_collected
            and memory_aux["loiter_flag"] > 0.5
        ):
            reward += Config.REWARD_LOITER
            detail["loiter"] = Config.REWARD_LOITER
        else:
            detail["loiter"] = 0.0

        # 12) 闪现奖励 / 惩罚
        is_flash_action = last_action is not None and 8 <= int(last_action) < 16
        if is_flash_action:
            prev_min_dist = self.prev_min_monster_dist if self.prev_min_monster_dist is not None else min_dist
            prev_path_dist = (
                self.prev_min_monster_path_dist if self.prev_min_monster_path_dist is not None else min_path_dist
            )
            flash_range = self._flash_range(last_action)
            dist_gain = min_dist - prev_min_dist
            path_gain = min_path_dist - prev_path_dist
            danger_drop = self.prev_danger - danger
            openness_gain = openness - self.prev_openness

            was_high_risk = self.prev_danger >= Config.FLASH_HIGH_DANGER_TH or prev_min_dist <= Config.FLASH_HIGH_DIST_TH
            was_mid_risk = self.prev_danger >= Config.FLASH_MID_DANGER_TH or prev_min_dist <= Config.FLASH_MID_DIST_TH
            clearly_escaped = (
                dist_gain >= Config.FLASH_DIST_GAIN_RATIO * flash_range
                or danger_drop >= 0.25
                or path_gain >= Config.FLASH_PATH_GAIN_RATIO * flash_range
            )
            modest_escape = (
                dist_gain >= Config.FLASH_MID_DIST_GAIN_RATIO * flash_range
                or danger_drop >= 0.08
                or path_gain >= Config.FLASH_MID_PATH_GAIN_RATIO * flash_range
            )
            more_open = openness_gain >= Config.FLASH_OPENNESS_GAIN_TH
            treasure_safe = treasure_gain > 0.0 and danger < Config.FLASH_SAFE_DANGER_TH

            if was_high_risk:
                if clearly_escaped:
                    r_flash = Config.REWARD_FLASH_HIGH_GOOD
                    flash_tier = "high_good"
                else:
                    r_flash = Config.REWARD_FLASH_HIGH_BAD
                    flash_tier = "high_bad"
            elif was_mid_risk:
                if modest_escape and more_open:
                    r_flash = Config.REWARD_FLASH_MID_GOOD
                    flash_tier = "mid_good"
                else:
                    r_flash = Config.REWARD_FLASH_MID_BAD
                    flash_tier = "mid_bad"
            elif treasure_safe:
                r_flash = Config.REWARD_FLASH_TREASURE_SAFE
                flash_tier = "low_treasure_safe"
            else:
                r_flash = Config.REWARD_FLASH_LOW_BAD
                flash_tier = "low_bad"

            reward += r_flash
            detail["flash"] = r_flash
            detail["flash_tier"] = flash_tier
            detail["flash_range"] = float(flash_range)
            detail["flash_dist_delta"] = float(dist_gain)
            detail["flash_path_delta"] = float(path_gain)
            detail["flash_danger_delta"] = float(danger_drop)
            detail["flash_open_delta"] = float(openness_gain)
        else:
            detail["flash"] = 0.0
            detail["flash_tier"] = "none"

        # 13) 终局奖励
        terminated = bool(self._pick(env_obs, "terminated", False))
        truncated = bool(self._pick(env_obs, "truncated", False))

        if terminated:
            reward += Config.REWARD_TERMINATED
            detail["terminal"] = Config.REWARD_TERMINATED
        elif truncated:
            if finished_steps >= max_step:
                reward += Config.REWARD_COMPLETED
                detail["terminal"] = Config.REWARD_COMPLETED
            else:
                reward += Config.REWARD_ABNORMAL
                detail["terminal"] = Config.REWARD_ABNORMAL
        else:
            detail["terminal"] = 0.0

        return np.asarray([reward], dtype=np.float32), detail

    # ==================================================
    # BFS helpers
    # ==================================================
    def _make_local_traversable(self, map_info):
        local = np.asarray(map_info, dtype=np.float32)
        if local.shape != (Config.MAP_SIZE, Config.MAP_SIZE):
            fixed = np.ones((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
            h = min(Config.MAP_SIZE, local.shape[0]) if local.ndim >= 2 else 0
            w = min(Config.MAP_SIZE, local.shape[1]) if local.ndim >= 2 else 0
            if h > 0 and w > 0:
                fixed[:h, :w] = local[:h, :w]
            local = fixed
        return (local > 0).astype(np.float32)

    def _bfs_from_center(self, traversable):
        h, w = traversable.shape
        dist = np.full((h, w), np.inf, dtype=np.float32)
        cx = cy = Config.MAP_SIZE // 2
        if traversable[cy, cx] <= 0.5:
            return dist

        q = deque()
        q.append((cx, cy))
        dist[cy, cx] = 0.0
        dirs = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]

        while q:
            x, y = q.popleft()
            base = dist[y, x]
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and traversable[ny, nx] > 0.5 and math.isinf(dist[ny, nx]):
                    dist[ny, nx] = base + 1.0
                    q.append((nx, ny))
        return dist

    def _directional_bfs_scores(self, traversable):
        """
        对 8 个方向分别打分：
        如果第一步走向该方向，那么从那个落点还能到达多少区域。
        分数 = reachable_count / 全局可达格数
        """
        h, w = traversable.shape
        center = Config.MAP_SIZE // 2
        dirs = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
        total_free = max(1.0, float(np.sum(traversable > 0.5)))
        scores = []

        for dx, dy in dirs:
            sx, sy = center + dx, center + dy
            if not (0 <= sx < w and 0 <= sy < h) or traversable[sy, sx] <= 0.5:
                scores.append(0.0)
                continue

            visited = np.zeros((h, w), dtype=np.uint8)
            q = deque()
            q.append((sx, sy))
            visited[sy, sx] = 1
            cnt = 0
            while q:
                x, y = q.popleft()
                cnt += 1
                for ddx, ddy in dirs:
                    nx, ny = x + ddx, y + ddy
                    if 0 <= nx < w and 0 <= ny < h and traversable[ny, nx] > 0.5 and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        q.append((nx, ny))
            scores.append(float(cnt / total_free))
        return scores

    def _lookup_bfs(self, bfs_dist, local_coord):
        if local_coord is None:
            return Config.BFS_DIST_NORM
        lx, lz = local_coord
        if 0 <= lz < Config.MAP_SIZE and 0 <= lx < Config.MAP_SIZE:
            val = float(bfs_dist[lz, lx])
            if math.isinf(val):
                return Config.BFS_DIST_NORM
            return val
        return Config.BFS_DIST_NORM

    @staticmethod
    def _flash_range(last_action):
        direction_idx = max(0, int(last_action) - 8) % 8
        return 10.0 if direction_idx % 2 == 0 else 8.0

    def _global_to_local(self, hero_pos, target_pos):
        hx, hz = hero_pos
        tx, tz = target_pos
        center = Config.MAP_SIZE // 2
        lx = tx - hx + center
        lz = tz - hz + center
        if 0 <= lx < Config.MAP_SIZE and 0 <= lz < Config.MAP_SIZE:
            return lx, lz
        return None

    def _nearest_frontier_dist(self, hero_pos):
        x, z = hero_pos
        frontier = np.argwhere(self.explored_mask < 0.5)
        if frontier.shape[0] == 0:
            return 999.0
        dx = frontier[:, 1] - x
        dz = frontier[:, 0] - z
        dist2 = dx * dx + dz * dz
        return float(math.sqrt(float(np.min(dist2))))

    # ==================================================
    # Other helpers
    # ==================================================
    def _danger_score(self, min_dist, second_dist, monsters):
        base = math.exp(-min_dist / 4.0)
        second_term = 0.4 * math.exp(-second_dist / 5.0) if len(monsters) > 1 else 0.0
        speed_term = 0.2 * max([m["speed"] - 1.0 for m in monsters], default=0.0)
        return float(np.clip(base + second_term + speed_term, 0.0, 1.5))

    def _analyze_topology(self, traversable):
        center = Config.MAP_SIZE // 2
        dirs = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
        run_lengths = []
        for dx, dy in dirs:
            length = 0
            x, y = center, center
            while True:
                x += dx
                y += dy
                if not (0 <= x < Config.MAP_SIZE and 0 <= y < Config.MAP_SIZE):
                    break
                if traversable[y, x] <= 0.5:
                    break
                length += 1
            run_lengths.append(length / 10.0)

        obstacle_density = float(1.0 - traversable.mean())
        openness = float(np.mean(traversable))
        cross = [
            traversable[center + dy, center + dx]
            if 0 <= center + dy < Config.MAP_SIZE and 0 <= center + dx < Config.MAP_SIZE
            else 0
            for dx, dy in dirs
        ]
        branch_count = float(sum(v > 0.5 for v in cross) / 8.0)
        dead_end_risk = float(1.0 - min(sum(v > 0.5 for v in cross) / 4.0, 1.0))
        near_wall = float(
            1.0
            - traversable[
                max(0, center - 1): min(Config.MAP_SIZE, center + 2),
                max(0, center - 1): min(Config.MAP_SIZE, center + 2),
            ].mean()
        )

        return {
            "run_lengths": run_lengths,
            "obstacle_density": obstacle_density,
            "openness": openness,
            "branch_count": branch_count,
            "dead_end_risk": dead_end_risk,
            "near_wall": near_wall,
        }

    def _crop_global(self, global_map, hero_pos, fill=0.0):
        x, z = hero_pos
        r = Config.MAP_SIZE // 2
        out = np.full((Config.MAP_SIZE, Config.MAP_SIZE), fill, dtype=np.float32)

        x0, x1 = x - r, x + r + 1
        z0, z1 = z - r, z + r + 1

        gx0, gx1 = max(0, x0), min(global_map.shape[1], x1)
        gz0, gz1 = max(0, z0), min(global_map.shape[0], z1)

        ox0, oz0 = gx0 - x0, gz0 - z0
        out[oz0:oz0 + (gz1 - gz0), ox0:ox0 + (gx1 - gx0)] = global_map[gz0:gz1, gx0:gx1]
        return out

    def _normalize_legal_action(self, legal_action):
        arr = [1 if bool(x) else 0 for x in list(legal_action)] if legal_action is not None else [1] * Config.ACTION_NUM
        if len(arr) < Config.ACTION_NUM:
            arr = arr + [0] * (Config.ACTION_NUM - len(arr))
        arr = arr[: Config.ACTION_NUM]
        if sum(arr) == 0:
            arr[:8] = [1] * 8
        return arr

    def _extract_hero(self, frame_state):
        heroes = self._pick(frame_state, "heroes", {})
        if isinstance(heroes, (list, tuple)):
            return heroes[0] if heroes else {}
        return heroes or {}

    def _normalize_monster(self, m):
        return {
            "pos": self._extract_pos(self._pick(m, "pos", {})),
            "speed": float(self._pick(m, "speed", 1) or 1),
            "hero_l2_distance": float(self._pick(m, "hero_l2_distance", 5) or 5),
            "hero_relative_direction": float(self._pick(m, "hero_relative_direction", 0) or 0),
        }

    def _normalize_organ(self, o):
        return {
            "sub_type": int(self._pick(o, "sub_type", 0) or 0),
            "status": int(self._pick(o, "status", 0) or 0),
            "pos": self._extract_pos(self._pick(o, "pos", {})),
            "hero_l2_distance": float(self._pick(o, "hero_l2_distance", 5) or 5),
            "hero_relative_direction": float(self._pick(o, "hero_relative_direction", 0) or 0),
        }

    def _extract_pos(self, pos_obj):
        return int(self._pick(pos_obj, "x", 0) or 0), int(self._pick(pos_obj, "z", 0) or 0)

    @staticmethod
    def _pick(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

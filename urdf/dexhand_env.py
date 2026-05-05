import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class DexHandGraspEnv(gym.Env):
    """
    第一阶段：三指 tripod pregrasp / contact learning
    使用：
    - th_tip_geom
    - ff_tip_geom
    - mf_tip_geom
    和 object_geom 的真实接触
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        model_path: str,
        object_geom_name: str = "object_geom",
        object_body_name: str = None,
        thumb_geom_name: str = "th_tip_geom",
        index_geom_name: str = "ff_tip_geom",
        middle_geom_name: str = "mf_tip_geom",
        frame_skip: int = 5,
        max_steps: int = 200,
        action_type: str = "delta",   # "absolute" or "delta"
        delta_scale: float = 0.005,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.step_count = 0

        self.action_type = action_type
        self.delta_scale = delta_scale

        # ------------------------------------------------
        # object geom / body
        # ------------------------------------------------
        self.object_geom_name = object_geom_name
        self.object_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, object_geom_name
        )
        if self.object_gid < 0:
            raise ValueError(f"找不到 object geom: {object_geom_name}")

        self.object_body_name = object_body_name
        if object_body_name is not None:
            self.object_bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, object_body_name
            )
            if self.object_bid < 0:
                raise ValueError(f"找不到 object body: {object_body_name}")
        else:
            self.object_bid = self.model.geom_bodyid[self.object_gid]

        # ------------------------------------------------
        # 三个关键指尖 geom
        # ------------------------------------------------
        # ------------------------------------------------
        # 三个关键手指的“接触组geom”
        # 每根手指 = distal本体(J1) + tip patch(J0)
        # ------------------------------------------------
        self.thumb_geom_names = ["th_distal_geom", "th_tip_geom"]
        self.index_geom_names = ["ff_distal_geom", "ff_tip_geom"]
        self.middle_geom_names = ["mf_distal_geom", "mf_tip_geom"]

        self.thumb_gids = []
        self.index_gids = []
        self.middle_gids = []

        for name in self.thumb_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(f"找不到 thumb geom: {name}")
            self.thumb_gids.append(gid)

        for name in self.index_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(f"找不到 index geom: {name}")
            self.index_gids.append(gid)

        for name in self.middle_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(f"找不到 middle geom: {name}")
            self.middle_gids.append(gid)


        # ------------------------------------------------
        # 关键 site
        # ------------------------------------------------
        self.tripod_site_names = ["th_tip_site", "ff_tip_site", "mf_tip_site"]
        self.palm_site_names = [
            "palm_contact_ff",
            "palm_contact_mf",
            "palm_contact_rf",
            "palm_contact_lf",
        ]

        self.site_ids = {}
        for s in self.tripod_site_names + self.palm_site_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s)
            if sid < 0:
                raise ValueError(f"找不到 site: {s}")
            self.site_ids[s] = sid

        # ------------------------------------------------
        # action / obs
        # ------------------------------------------------
        self.nu = self.model.nu

        # ===== 锁定无名指/小指 actuator 到 0 =====
        self.act_name_to_id = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is not None:
                self.act_name_to_id[name] = i

        self.locked_act_names = [
            "RFJ4", "RFJ3", "RFJ2", "RFJ1",
            "LFJ4", "LFJ3", "LFJ2", "LFJ1",
        ]

        self.locked_act_target = {}
        for n in self.locked_act_names:
            if n in self.act_name_to_id:
                aid = self.act_name_to_id[n]
                lo, hi = self.model.actuator_ctrlrange[aid]

                # 固定到 0；若 0 不在范围内则夹到范围边界
                target = np.clip(0.0, lo, hi)
                self.locked_act_target[aid] = float(target)

        # 记录最近contact
        self.last_contact_sum = 0

        # ===== 四阶段状态机 =====
        # 1: approach
        # 2: contact
        # 3: stabilize_soft
        # 4: freeze_hold
        self.phase = 1

        # 连续接触计数
        self.contact_streak = 0
        self.loss_streak = 0

        # soft stabilize 阶段的计数
        self.stabilize_steps = 0
        self.max_stabilize_steps = 6   # 接触后先微调几步

        # freeze hold 阶段
        self.freeze_ctrl = None
        self.freeze_steps = 0
        self.max_freeze_steps = 12

        # 成功标志
        self.success_hold = False
        # ===== action space =====
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        # ===== 历史状态 =====
        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        self.prev_ctrl = np.zeros(self.nu, dtype=np.float32)

        self.prev_palm_obj_dist = None
        self.prev_tripod_mean_dist = None

        # ===== 先 reset 一次，构建 observation_space =====
        obs = self._reset_sim_and_get_obs()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        print("[DexHandGraspEnv] action_space:", self.action_space)
        print("[DexHandGraspEnv] observation_space:", self.observation_space)


    # =====================================================
    # Gym API
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._reset_sim_and_get_obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        ctrl = self._action_to_ctrl(action)
        self.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, info = self._compute_reward(action)

        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    # =====================================================
    # Reset / Obs
    # =====================================================
    def _reset_sim_and_get_obs(self):
        mujoco.mj_resetData(self.model, self.data)

        self.last_contact_sum = 0

        self.phase = 1
        self.contact_streak = 0
        self.loss_streak = 0

        self.stabilize_steps = 0
        self.freeze_ctrl = None
        self.freeze_steps = 0
        self.success_hold = False


        self.step_count = 0
        self.prev_action[:] = 0.0

        self.prev_ctrl = self._default_ctrl().copy()
        self.data.ctrl[:] = self.prev_ctrl

        self._randomize_object_pose_if_possible()

        mujoco.mj_forward(self.model, self.data)

        palm_center = self._get_palm_center()
        obj_pos = self._get_object_pos()
        tripod_dists = self._get_tripod_dists(obj_pos)

        self.prev_palm_obj_dist = np.linalg.norm(palm_center - obj_pos)
        self.prev_tripod_mean_dist = np.mean(tripod_dists)

        obj_linvel, obj_angvel = self._get_object_velocity()
        self.prev_obj_speed = float(np.linalg.norm(obj_linvel))
        self.best_contact_sum = 0
        self.best_contact_streak = 0

        return self._get_obs()

    def _get_obs(self):
        qpos = self.data.qpos.copy().astype(np.float32)
        qvel = self.data.qvel.copy().astype(np.float32)

        obj_pos = self._get_object_pos().astype(np.float32)
        obj_linvel, obj_angvel = self._get_object_velocity()

        palm_center = self._get_palm_center().astype(np.float32)
        th_pos = self._get_site_pos("th_tip_site").astype(np.float32)
        ff_pos = self._get_site_pos("ff_tip_site").astype(np.float32)
        mf_pos = self._get_site_pos("mf_tip_site").astype(np.float32)

        th_c, ff_c, mf_c = self._get_tripod_contacts()
        contact_vec = np.array([th_c, ff_c, mf_c], dtype=np.float32)

        obs = np.concatenate([
            qpos,
            qvel,
            obj_pos,
            obj_linvel.astype(np.float32),
            obj_angvel.astype(np.float32),
            palm_center - obj_pos,
            th_pos - obj_pos,
            ff_pos - obj_pos,
            mf_pos - obj_pos,
            contact_vec,
            self.prev_action.astype(np.float32),
            self.prev_ctrl.astype(np.float32),
        ], axis=0)

        return obs

    # =====================================================
    # Reward
    # =====================================================
    def _compute_reward(self, action):
        obj_pos = self._get_object_pos()
        palm_center = self._get_palm_center()

        th_pos = self._get_site_pos("th_tip_site")
        ff_pos = self._get_site_pos("ff_tip_site")
        mf_pos = self._get_site_pos("mf_tip_site")

        # ---------- 几何 ----------
        palm_obj_dist = np.linalg.norm(palm_center - obj_pos)

        tripod_dists = np.array([
            np.linalg.norm(th_pos - obj_pos),
            np.linalg.norm(ff_pos - obj_pos),
            np.linalg.norm(mf_pos - obj_pos),
        ], dtype=np.float32)
        tripod_mean_dist = float(np.mean(tripod_dists))

        # dense proximity
        r_palm_dense = np.exp(-4.0 * palm_obj_dist)
        r_tripod_dense = np.exp(-10.0 * tripod_mean_dist)

        # progress
        palm_progress = 0.0
        if self.prev_palm_obj_dist is not None:
            palm_progress = self.prev_palm_obj_dist - palm_obj_dist

        tripod_progress = 0.0
        if self.prev_tripod_mean_dist is not None:
            tripod_progress = self.prev_tripod_mean_dist - tripod_mean_dist

        # 远离惩罚（只罚变远）
        dist_apart = 0.0
        if self.prev_tripod_mean_dist is not None:
            dist_apart = max(tripod_mean_dist - self.prev_tripod_mean_dist, 0.0)

        r_tripod_shape = self._compute_tripod_shape_reward(th_pos, ff_pos, mf_pos, obj_pos)

        # ---------- 接触 ----------
        th_c, ff_c, mf_c = self._get_tripod_contacts()
        contact_sum = th_c + ff_c + mf_c
        self.last_contact_sum = contact_sum

        # ---------- 物体速度 ----------
        obj_linvel, obj_angvel = self._get_object_velocity()
        obj_speed = float(np.linalg.norm(obj_linvel))
        obj_ang_speed = float(np.linalg.norm(obj_angvel))

        speed_increase = 0.0
        if hasattr(self, "prev_obj_speed") and self.prev_obj_speed is not None:
            speed_increase = max(obj_speed - self.prev_obj_speed, 0.0)

        # ---------- streak 更新 ----------
        if contact_sum >= 2:
            self.contact_streak += 1
            self.loss_streak = 0
        else:
            self.contact_streak = 0
            self.loss_streak += 1

        self.best_contact_sum = max(self.best_contact_sum, contact_sum)
        self.best_contact_streak = max(self.best_contact_streak, self.contact_streak)

        prev_phase = self.phase

        # ---------- Phase 转移 ----------
        if self.phase == 1 and contact_sum >= 1:
            self.phase = 2

        if (
            self.phase == 2 and
            self.contact_streak >= 2 and
            contact_sum >= 2 and
            obj_speed < 0.20
        ):
            self.phase = 3
            self.stabilize_steps = 0

        if self.phase == 3:
            self.stabilize_steps += 1

            if (
                contact_sum >= 2 and
                obj_speed < 0.03 and
                obj_ang_speed < 0.30 and
                self.stabilize_steps >= self.max_stabilize_steps
            ):
                self.phase = 4
                self.freeze_ctrl = self.data.ctrl.copy()
                self.freeze_steps = 0

        if self.phase == 4:
            self.freeze_steps += 1

        if self.phase != prev_phase:
            print(
                f"[PHASE TRANSITION] step={self.step_count}, "
                f"{prev_phase} -> {self.phase}, "
                f"contact_sum={contact_sum}, "
                f"tripod_mean_dist={tripod_mean_dist:.4f}, "
                f"obj_speed={obj_speed:.4f}"
            )

        # ---------- 通用惩罚 ----------
        r_ctrl_penalty = -0.0015 * np.sum(np.square(action))
        r_smooth_penalty = -0.0030 * np.sum(np.square(action - self.prev_action))

        # 接触时物体速度大，要明显惩罚
        # 未接触时不要太重，否则接近过程被压死
        if contact_sum > 0:
            r_obj_speed = -1.2 * obj_speed - 0.15 * obj_ang_speed
        else:
            r_obj_speed = -0.15 * obj_speed - 0.02 * obj_ang_speed

        # 速度突然上涨，也要罚
        r_speed_burst = -0.6 * speed_increase

        reward = 0.0

        # ==================================================
        # Phase 1: approach
        # ==================================================
        if self.phase == 1:
            r_contact_bonus = 0.8 if contact_sum >= 1 else 0.0

            reward = (
                1.0 * r_palm_dense +
                2.0 * r_tripod_dense +
                6.0 * palm_progress +
                10.0 * tripod_progress +
                0.4 * r_tripod_shape +
                r_contact_bonus +
                r_ctrl_penalty +
                r_smooth_penalty
            )

        # ==================================================
        # Phase 2: make and keep contact
        # ==================================================
        elif self.phase == 2:
            r_contact_count = 1.2 * contact_sum
            r_multi_contact = 2.0 if contact_sum >= 2 else 0.0
            r_tripod_full = 3.0 if contact_sum == 3 else 0.0

            # 持续接触奖励：鼓励别一碰就掉
            r_persist = 0.8 * min(self.contact_streak, 5)

            # 接触后还继续靠近/包裹
            r_close_keep = 2.0 * np.exp(-12.0 * tripod_mean_dist)

            # 继续变近奖励
            r_progress_keep = 8.0 * max(tripod_progress, 0.0)

            # 一旦开始远离，重罚
            r_apart = -10.0 * dist_apart

            # 丢接触惩罚，连续掉越久越重
            r_drop = -0.8 * min(self.loss_streak, 6) if contact_sum == 0 else 0.0

            # 如果接触很少但速度很大，说明在推
            r_push_penalty = 0.0
            if contact_sum <= 1:
                r_push_penalty = -0.8 * obj_speed

            reward = (
                r_contact_count +
                r_multi_contact +
                r_tripod_full +
                r_persist +
                r_close_keep +
                r_progress_keep +
                0.3 * r_tripod_shape +
                r_apart +
                r_drop +
                r_push_penalty +
                r_obj_speed +
                r_speed_burst +
                1.5 * r_ctrl_penalty +
                1.5 * r_smooth_penalty
            )

        # ==================================================
        # Phase 3: stabilize
        # ==================================================
        elif self.phase == 3:
            r_hold = 4.0 if contact_sum >= 2 else 0.0
            r_hold += 2.0 if contact_sum == 3 else 0.0

            r_stable_close = 2.5 * np.exp(-15.0 * tripod_mean_dist)
            r_low_speed = -1.8 * obj_speed - 0.25 * obj_ang_speed
            r_apart = -8.0 * dist_apart

            # phase3 掉了接触，重罚
            if contact_sum == 0:
                r_drop = -6.0
            elif contact_sum == 1:
                r_drop = -2.0
            else:
                r_drop = 0.0

            # 每稳定一步给一点奖励
            r_stabilize_progress = 0.8

            reward = (
                r_hold +
                r_stable_close +
                r_low_speed +
                r_apart +
                r_drop +
                r_stabilize_progress +
                r_speed_burst +
                1.5 * r_ctrl_penalty +
                1.5 * r_smooth_penalty
            )

        # ==================================================
        # Phase 4: freeze hold
        # ==================================================
        else:
            r_hold = 8.0 if contact_sum >= 2 else 0.0
            r_hold += 3.0 if contact_sum == 3 else 0.0

            r_low_speed = -1.5 * obj_speed - 0.2 * obj_ang_speed

            if contact_sum == 0:
                r_drop = -8.0
            elif contact_sum == 1:
                r_drop = -3.0
            else:
                r_drop = 0.0

            r_freeze_bonus = 1.2

            reward = (
                r_hold +
                r_low_speed +
                r_drop +
                r_freeze_bonus
            )

        # ---------- success ----------
        success = (
            self.phase == 4 and
            self.freeze_steps >= self.max_freeze_steps and
            contact_sum >= 2
        )

        if success:
            reward += 100.0
            self.success_hold = True

        # ---------- update history ----------
        self.prev_palm_obj_dist = palm_obj_dist
        self.prev_tripod_mean_dist = tripod_mean_dist
        self.prev_obj_speed = obj_speed
        self.prev_action = action.copy()

        info = {
            "reward_total": float(reward),
            "phase": int(self.phase),
            "stabilize_steps": int(self.stabilize_steps),
            "freeze_steps": int(self.freeze_steps),
            "palm_obj_dist": float(palm_obj_dist),
            "tripod_mean_dist": float(tripod_mean_dist),
            "r_tripod_shape": float(r_tripod_shape),
            "obj_speed": float(obj_speed),
            "obj_ang_speed": float(obj_ang_speed),
            "th_contact": int(th_c),
            "ff_contact": int(ff_c),
            "mf_contact": int(mf_c),
            "contact_sum": int(contact_sum),
            "contact_streak": int(self.contact_streak),
            "loss_streak": int(self.loss_streak),
            "best_contact_sum": int(self.best_contact_sum),
            "best_contact_streak": int(self.best_contact_streak),
            "success": bool(success),
        }
        return float(reward), info




    # =====================================================
    # Contact
    # =====================================================
    def _geom_pair_in_contact(self, gid_a, gid_b):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == gid_a and g2 == gid_b) or (g1 == gid_b and g2 == gid_a):
                return 1
        return 0

    def _geom_group_in_contact(self, geom_ids, object_gid):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            if g1 == object_gid and g2 in geom_ids:
                return 1
            if g2 == object_gid and g1 in geom_ids:
                return 1
        return 0
    
    def _get_tripod_contacts(self):
        th_c = self._geom_group_in_contact(self.thumb_gids, self.object_gid)
        ff_c = self._geom_group_in_contact(self.index_gids, self.object_gid)
        mf_c = self._geom_group_in_contact(self.middle_gids, self.object_gid)
        return th_c, ff_c, mf_c



    # =====================================================
    # Geometry helpers
    # =====================================================
    def _compute_tripod_shape_reward(self, th_pos, ff_pos, mf_pos, obj_pos):
        eps = 1e-6

        v_th = th_pos - obj_pos
        v_ff = ff_pos - obj_pos
        v_mf = mf_pos - obj_pos

        u_th = v_th / (np.linalg.norm(v_th) + eps)
        u_ff = v_ff / (np.linalg.norm(v_ff) + eps)
        u_mf = v_mf / (np.linalg.norm(v_mf) + eps)

        d_th_ff = np.dot(u_th, u_ff)
        d_th_mf = np.dot(u_th, u_mf)
        d_ff_mf = np.dot(u_ff, u_mf)

        s_th_ff = (1.0 - d_th_ff) / 2.0
        s_th_mf = (1.0 - d_th_mf) / 2.0
        s_ff_mf = (1.0 - d_ff_mf) / 2.0

        reward = 0.4 * s_th_ff + 0.4 * s_th_mf + 0.2 * s_ff_mf
        return float(reward)

    def _get_site_pos(self, site_name):
        sid = self.site_ids[site_name]
        return self.data.site_xpos[sid].copy()

    def _get_palm_center(self):
        pts = []
        for name in self.palm_site_names:
            pts.append(self._get_site_pos(name))
        return np.mean(np.array(pts), axis=0)

    def _get_object_pos(self):
        return self.data.xpos[self.object_bid].copy()

    def _get_tripod_dists(self, obj_pos):
        return np.array([
            np.linalg.norm(self._get_site_pos("th_tip_site") - obj_pos),
            np.linalg.norm(self._get_site_pos("ff_tip_site") - obj_pos),
            np.linalg.norm(self._get_site_pos("mf_tip_site") - obj_pos),
        ], dtype=np.float32)

    def _get_object_velocity(self):
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.object_bid,
            vel,
            0,
        )
        angvel = vel[:3].copy()
        linvel = vel[3:].copy()
        return linvel, angvel

    # =====================================================
    # Control
    # =====================================================
    def _default_ctrl(self):
        ctrl = np.zeros(self.nu, dtype=np.float32)
        for i in range(self.nu):
            low, high = self.model.actuator_ctrlrange[i]
            ctrl[i] = 0.5 * (low + high)
        return ctrl

    def _action_to_ctrl(self, action):
        # Phase 4: 完全冻结
        if self.phase == 4 and self.freeze_ctrl is not None:
            ctrl = self.freeze_ctrl.copy()

            for aid, target in self.locked_act_target.items():
                ctrl[aid] = target

            self.prev_ctrl = ctrl.copy()
            return ctrl

        # 非冻结阶段
        if self.action_type == "absolute":
            ctrl = np.zeros(self.nu, dtype=np.float32)
            for i in range(self.nu):
                lo, hi = self.model.actuator_ctrlrange[i]
                ctrl[i] = lo + (action[i] + 1.0) * 0.5 * (hi - lo)

        elif self.action_type == "delta":
            ctrl = self.prev_ctrl.copy()

            # 分阶段控制尺度
            if self.phase == 1:
                effective_scale = 0.001         # 例如 0.005
            elif self.phase == 2:
                effective_scale = 0.0003                    # 接触阶段更柔和
            elif self.phase == 3:
                effective_scale = 0.0001                    # 稳定微调阶段极小
            else:
                effective_scale = 0.0

            for i in range(self.nu):
                lo, hi = self.model.actuator_ctrlrange[i]
                delta = effective_scale * action[i] * (hi - lo)
                ctrl[i] = np.clip(ctrl[i] + delta, lo, hi)

        else:
            raise ValueError(f"未知 action_type: {self.action_type}")

        # 锁 RF/LF 到 0
        for aid, target in self.locked_act_target.items():
            ctrl[aid] = target

        self.prev_ctrl = ctrl.copy()
        return ctrl




    # =====================================================
    # Termination
    # =====================================================
    def _check_terminated(self):
        obj_pos = self._get_object_pos()

        # 物体飞太远
        if np.linalg.norm(obj_pos) > 5.0:
            return True

        # phase2: 已经进入接触阶段，但连续丢失太久，直接结束
        if self.phase == 2 and self.loss_streak >= 6:
            return True

        # phase3: 进入稳定阶段后又丢失太久，直接结束
        if self.phase == 3 and self.loss_streak >= 6:
            return True

        # phase4 成功
        if self.phase == 4 and self.freeze_steps >= self.max_freeze_steps and self.last_contact_sum >= 2:
            return True

        # phase4 掉了
        if self.phase == 4 and self.loss_streak >= 4:
            return True

        return False





    # =====================================================
    # Object randomization
    # =====================================================
    def _randomize_object_pose_if_possible(self):
        jadr = self.model.body_jntadr[self.object_bid]
        jnum = self.model.body_jntnum[self.object_bid]

        if jnum <= 0:
            return

        jid = jadr
        jtype = self.model.jnt_type[jid]
        if jtype != mujoco.mjtJoint.mjJNT_FREE:
            return

        qpos_adr = self.model.jnt_qposadr[jid]

        self.data.qpos[qpos_adr + 0] += np.random.uniform(-0.01, 0.01)
        self.data.qpos[qpos_adr + 1] += np.random.uniform(-0.01, 0.01)
        self.data.qpos[qpos_adr + 2] += np.random.uniform(-0.005, 0.005)

        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        qvel_adr = self.model.jnt_dofadr[jid]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

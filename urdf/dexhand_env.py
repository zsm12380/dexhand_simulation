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
        delta_scale: float = 0.03,
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
        self.thumb_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, thumb_geom_name
        )
        self.index_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, index_geom_name
        )
        self.middle_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, middle_geom_name
        )

        if self.thumb_gid < 0:
            raise ValueError(f"找不到 thumb geom: {thumb_geom_name}")
        if self.index_gid < 0:
            raise ValueError(f"找不到 index geom: {index_geom_name}")
        if self.middle_gid < 0:
            raise ValueError(f"找不到 middle geom: {middle_geom_name}")

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
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        self.prev_ctrl = np.zeros(self.nu, dtype=np.float32)

        self.prev_palm_obj_dist = None
        self.prev_tripod_mean_dist = None

        obs = self._reset_sim_and_get_obs()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

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

        # 1) 手掌接近物体
        palm_obj_dist = np.linalg.norm(palm_center - obj_pos)
        r_approach_palm_dense = np.exp(-4.0 * palm_obj_dist)

        r_approach_palm_progress = 0.0
        if self.prev_palm_obj_dist is not None:
            r_approach_palm_progress = 5.0 * (self.prev_palm_obj_dist - palm_obj_dist)

        # 2) tripod 三指接近物体
        tripod_dists = np.array([
            np.linalg.norm(th_pos - obj_pos),
            np.linalg.norm(ff_pos - obj_pos),
            np.linalg.norm(mf_pos - obj_pos),
        ], dtype=np.float32)

        tripod_mean_dist = float(np.mean(tripod_dists))
        r_approach_tripod_dense = np.exp(-5.0 * tripod_mean_dist)

        r_approach_tripod_progress = 0.0
        if self.prev_tripod_mean_dist is not None:
            r_approach_tripod_progress = 8.0 * (
                self.prev_tripod_mean_dist - tripod_mean_dist
            )

        # 3) tripod 几何结构
        r_tripod_shape = self._compute_tripod_shape_reward(
            th_pos, ff_pos, mf_pos, obj_pos
        )

        # 4) 真实接触奖励
        th_c, ff_c, mf_c = self._get_tripod_contacts()
        contact_sum = th_c + ff_c + mf_c

        r_contact_count = 1.5 * contact_sum

        r_contact_pair = 0.0
        if th_c and ff_c:
            r_contact_pair += 2.0
        if th_c and mf_c:
            r_contact_pair += 2.0

        r_contact_combo = 0.0
        if contact_sum >= 2:
            r_contact_combo += 2.0
        if contact_sum == 3:
            r_contact_combo += 4.0

        # 5) 控制代价
        r_ctrl_penalty = -0.005 * np.sum(np.square(action))
        r_smooth_penalty = -0.01 * np.sum(np.square(action - self.prev_action))

        # 6) 物体稳定项，避免直接打飞
        obj_linvel, obj_angvel = self._get_object_velocity()
        r_obj_stability = -0.01 * np.linalg.norm(obj_linvel) - 0.005 * np.linalg.norm(obj_angvel)

        reward = (
            1.0 * r_approach_palm_dense +
            1.5 * r_approach_palm_progress +
            2.0 * r_approach_tripod_dense +
            2.0 * r_approach_tripod_progress +
            2.0 * r_tripod_shape +
            r_contact_count +
            r_contact_pair +
            r_contact_combo +
            r_ctrl_penalty +
            r_smooth_penalty +
            r_obj_stability
        )

        success = (
            contact_sum >= 2 and
            tripod_mean_dist < 0.05 and
            r_tripod_shape > 0.55
        )

        if success:
            reward += 10.0

        self.prev_palm_obj_dist = palm_obj_dist
        self.prev_tripod_mean_dist = tripod_mean_dist
        self.prev_action = action.copy()

        info = {
            "reward_total": float(reward),
            "palm_obj_dist": float(palm_obj_dist),
            "tripod_mean_dist": float(tripod_mean_dist),
            "r_tripod_shape": float(r_tripod_shape),
            "th_contact": int(th_c),
            "ff_contact": int(ff_c),
            "mf_contact": int(mf_c),
            "contact_sum": int(contact_sum),
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

    def _get_tripod_contacts(self):
        th_c = self._geom_pair_in_contact(self.thumb_gid, self.object_gid)
        ff_c = self._geom_pair_in_contact(self.index_gid, self.object_gid)
        mf_c = self._geom_pair_in_contact(self.middle_gid, self.object_gid)
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
        if self.action_type == "absolute":
            ctrl = np.zeros(self.nu, dtype=np.float32)
            for i in range(self.nu):
                low, high = self.model.actuator_ctrlrange[i]
                ctrl[i] = low + (action[i] + 1.0) * 0.5 * (high - low)

        elif self.action_type == "delta":
            ctrl = self.prev_ctrl.copy()
            for i in range(self.nu):
                low, high = self.model.actuator_ctrlrange[i]
                delta = self.delta_scale * action[i] * (high - low)
                ctrl[i] = np.clip(ctrl[i] + delta, low, high)
        else:
            raise ValueError(f"未知 action_type: {self.action_type}")

        self.prev_ctrl = ctrl.copy()
        return ctrl

    # =====================================================
    # Termination
    # =====================================================
    def _check_terminated(self):
        obj_pos = self._get_object_pos()
        if np.linalg.norm(obj_pos) > 5.0:
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

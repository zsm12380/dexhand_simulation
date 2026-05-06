import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class DexHandGraspEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        model_path: str,
        workspace_path: str = "workspace_tripod.npz",
        object_geom_name: str = "object_geom",
        object_body_name: str = None,
        frame_skip: int = 5,
        max_steps: int = 220,
        action_type: str = "delta",
        delta_scale: float = 0.002,
        # Tripod geometry. h is the real object thickness / grasp thickness.
        tripod_h: float = 0.015,
        tol_h: float = 0.00025,
        tol_sym: float = 0.0008,
        tol_base_height_orth: float = 0.04,
        tol_normal_plane: float = 0.04,
        tol_normal_height_align: float = 0.95,
        min_base_len: float = 0.003,
        max_tripod_trials: int = 20000,
        max_reset_attempts: int = 80,
        require_no_initial_contact: bool = True,
        object_center_mode: str = "between_thumb_and_base",
        debug_tripod: bool = True,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.step_count = 0
        self.action_type = action_type
        self.delta_scale = delta_scale

        self.tripod_h = float(tripod_h)
        self.tol_h = float(tol_h)
        self.tol_sym = float(tol_sym)
        self.tol_base_height_orth = float(tol_base_height_orth)
        self.tol_normal_plane = float(tol_normal_plane)
        self.tol_normal_height_align = float(tol_normal_height_align)
        self.min_base_len = float(min_base_len)
        self.max_tripod_trials = int(max_tripod_trials)
        self.max_reset_attempts = int(max_reset_attempts)
        self.require_no_initial_contact = bool(require_no_initial_contact)
        self.object_center_mode = object_center_mode
        self.debug_tripod = bool(debug_tripod)

        # object
        self.object_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, object_geom_name)
        if self.object_gid < 0:
            raise ValueError(f"找不到 object geom: {object_geom_name}")

        if object_body_name is not None:
            self.object_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
            if self.object_bid < 0:
                raise ValueError(f"找不到 object body: {object_body_name}")
        else:
            self.object_bid = int(self.model.geom_bodyid[self.object_gid])

        # finger geom groups
        self.thumb_gids = self._geom_ids(["th_distal_geom", "th_tip_geom"], "thumb")
        self.index_gids = self._geom_ids(["ff_distal_geom", "ff_tip_geom"], "index")
        self.middle_gids = self._geom_ids(["mf_distal_geom", "mf_tip_geom"], "middle")
        self.tripod_hand_gids = set(self.thumb_gids + self.index_gids + self.middle_gids)

        # sites
        self.tripod_site_names = ["th_tip_site", "ff_tip_site", "mf_tip_site"]
        self.tripod_ref_site_names = ["th_j1_ref_site", "ff_j1_ref_site", "mf_j1_ref_site"]
        self.palm_site_names = ["palm_contact_ff", "palm_contact_mf", "palm_contact_rf", "palm_contact_lf"]

        self.site_ids = {}
        for s in self.tripod_site_names + self.tripod_ref_site_names + self.palm_site_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s)
            if sid < 0:
                raise ValueError(f"找不到 site: {s}")
            self.site_ids[s] = sid

        # actuator map
        self.nu = self.model.nu
        self.act_name_to_id = {}
        for i in range(self.nu):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if n is not None:
                self.act_name_to_id[n] = i

        # lock RF/LF
        self.locked_act_names = ["RFJ4", "RFJ3", "RFJ2", "RFJ1", "LFJ4", "LFJ3", "LFJ2", "LFJ1"]
        self.locked_act_target = {}
        for n in self.locked_act_names:
            if n in self.act_name_to_id:
                aid = self.act_name_to_id[n]
                lo, hi = self.model.actuator_ctrlrange[aid]
                self.locked_act_target[aid] = float(np.clip(0.0, lo, hi))

        # load workspace
        ws = np.load(workspace_path)
        self.ws_th_pos = ws["th_pos"].astype(np.float64)
        self.ws_th_nrm = ws["th_nrm"].astype(np.float64)
        self.ws_ff_pos = ws["ff_pos"].astype(np.float64)
        self.ws_ff_nrm = ws["ff_nrm"].astype(np.float64)
        self.ws_mf_pos = ws["mf_pos"].astype(np.float64)
        self.ws_mf_nrm = ws["mf_nrm"].astype(np.float64)

        # phase
        self.phase = 1
        self.contact_streak = 0
        self.loss_streak = 0
        self.stabilize_steps = 0
        self.max_stabilize_steps = 10
        self.freeze_ctrl = None
        self.freeze_steps = 0
        self.max_freeze_steps = 15
        self.success_hold = False

        # history
        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        self.prev_ctrl = np.zeros(self.nu, dtype=np.float32)
        self.prev_geom_err = None
        self.prev_obj_speed = None
        self.last_contact_sum = 0
        self.target_tripod = None

        # spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)
        obs = self._reset_sim_and_get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)

        print("[DexHandGraspEnv] action_space:", self.action_space)
        print("[DexHandGraspEnv] observation_space:", self.observation_space)

    # ===== gym api =====
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

    # ===== reset/obs =====
    def _reset_sim_and_get_obs(self):
        mujoco.mj_resetData(self.model, self.data)

        self.phase = 1
        self.contact_streak = 0
        self.loss_streak = 0
        self.stabilize_steps = 0
        self.freeze_ctrl = None
        self.freeze_steps = 0
        self.success_hold = False
        self.last_contact_sum = 0
        self.step_count = 0

        self.prev_action[:] = 0.0
        self.prev_ctrl = self._default_ctrl().copy()
        self.data.ctrl[:] = self.prev_ctrl

        found_valid_reset = False
        last_sol = None
        last_contact_sum = None

        for attempt in range(self.max_reset_attempts):
            sol = self._solve_tripod_targets_from_workspace(
                h=self.tripod_h,
                tol_h=self.tol_h,
                tol_sym=self.tol_sym,
                tol_base_height_orth=self.tol_base_height_orth,
                tol_normal_plane=self.tol_normal_plane,
                tol_normal_height_align=self.tol_normal_height_align,
                min_base_len=self.min_base_len,
                max_trials=self.max_tripod_trials,
            )

            if sol is None:
                continue

            last_sol = sol
            self._place_object_from_tripod_solution(sol)
            self.target_tripod = sol
            mujoco.mj_forward(self.model, self.data)

            contact_sum = sum(self._get_tripod_contacts())
            last_contact_sum = contact_sum

            if (not self.require_no_initial_contact) or contact_sum == 0:
                found_valid_reset = True
                break

        if not found_valid_reset:
            if last_sol is not None:
                self._place_object_from_tripod_solution(last_sol)
                self.target_tripod = last_sol
            else:
                self._randomize_object_pose_if_possible()
                self.target_tripod = None
            mujoco.mj_forward(self.model, self.data)

        self.prev_geom_err = self._compute_geom_err_to_target()
        obj_linvel, _ = self._get_object_velocity()
        self.prev_obj_speed = float(np.linalg.norm(obj_linvel))

        if self.debug_tripod and self.target_tripod is not None:
            th_c, ff_c, mf_c = self._get_tripod_contacts()
            sol = self.target_tripod
            print(
                "[RESET]",
                f"ok={found_valid_reset}",
                f"contact=({th_c},{ff_c},{mf_c})",
                f"ncon={self.data.ncon}",
                f"geom_err={self.prev_geom_err:.8f}",
                f"h_err={sol.get('h_err', -1.0):.8f}",
                f"sym={sol.get('sym', -1.0):.8f}",
                f"orth={sol.get('base_height_orth_err', -1.0):.6f}",
                f"plane={sol.get('normal_plane_err', -1.0):.6f}",
                f"height_align={sol.get('normal_height_score', -1.0):.6f}",
                f"last_contact={last_contact_sum}",
            )

        return self._get_obs()

    def _get_obs(self):
        qpos = self.data.qpos.copy().astype(np.float32)
        qvel = self.data.qvel.copy().astype(np.float32)

        obj_pos = self._get_object_pos().astype(np.float32)
        obj_linvel, obj_angvel = self._get_object_velocity()

        # 用 fingertip site 和 reward 保持一致，不再用 j1_ref_site 做主要几何误差。
        th = self._get_site_pos("th_tip_site").astype(np.float32)
        ff = self._get_site_pos("ff_tip_site").astype(np.float32)
        mf = self._get_site_pos("mf_tip_site").astype(np.float32)

        if self.target_tripod is not None:
            pth = self.target_tripod["pth"].astype(np.float32)
            pff = self.target_tripod["pff"].astype(np.float32)
            pmf = self.target_tripod["pmf"].astype(np.float32)
        else:
            pth, pff, pmf = th.copy(), ff.copy(), mf.copy()

        th_c, ff_c, mf_c = self._get_tripod_contacts()
        contact_vec = np.array([th_c, ff_c, mf_c], dtype=np.float32)

        phase_vec = np.array([self.phase / 4.0], dtype=np.float32)

        obs = np.concatenate([
            qpos,
            qvel,
            obj_pos,
            obj_linvel.astype(np.float32),
            obj_angvel.astype(np.float32),
            th - pth,
            ff - pff,
            mf - pmf,
            contact_vec,
            phase_vec,
            self.prev_action.astype(np.float32),
            self.prev_ctrl.astype(np.float32),
        ], axis=0)
        return obs

    # ===== reward =====
    def _compute_reward(self, action):
        th_c, ff_c, mf_c = self._get_tripod_contacts()
        contact_sum = th_c + ff_c + mf_c
        self.last_contact_sum = contact_sum

        geom_err = self._compute_geom_err_to_target()
        geom_prog = 0.0 if self.prev_geom_err is None else (self.prev_geom_err - geom_err)

        normal_align = self._compute_normal_align_score()

        obj_linvel, obj_angvel = self._get_object_velocity()
        obj_speed = float(np.linalg.norm(obj_linvel))
        obj_ang_speed = float(np.linalg.norm(obj_angvel))
        speed_inc = 0.0 if self.prev_obj_speed is None else max(obj_speed - self.prev_obj_speed, 0.0)

        if contact_sum >= 2:
            self.contact_streak += 1
            self.loss_streak = 0
        else:
            self.contact_streak = 0
            self.loss_streak += 1

        prev_phase = self.phase

        # Phase 1: approach. 不再只靠 1 个 contact 立刻升级。
        if self.phase == 1:
            if self.step_count >= 4 and geom_err < 0.006 and contact_sum >= 1:
                self.phase = 2

        # Phase 2: establish two/three contacts.
        if self.phase == 2:
            if geom_err < 0.004 and contact_sum >= 2 and self.contact_streak >= 3:
                self.phase = 3
                self.stabilize_steps = 0

        # Phase 3: stabilize tripod grasp.
        if self.phase == 3:
            self.stabilize_steps += 1
            stable = (
                contact_sum >= 2 and
                normal_align > 0.55 and
                obj_speed < 0.035 and
                obj_ang_speed < 0.45 and
                self.stabilize_steps >= self.max_stabilize_steps
            )
            if stable:
                self.phase = 4
                self.freeze_ctrl = self.data.ctrl.copy()
                self.freeze_steps = 0

        if self.phase == 4:
            self.freeze_steps += 1

        if self.phase != prev_phase:
            print(
                f"[PHASE] {prev_phase}->{self.phase}, "
                f"step={self.step_count}, contact={contact_sum}, "
                f"geom_err={geom_err:.8f}, n_align={normal_align:.3f}"
            )

        # Reward terms.
        r_geom = -10.0 * geom_err + 16.0 * geom_prog
        r_contact = 1.2 * contact_sum + (2.0 if contact_sum >= 2 else 0.0) + (3.0 if contact_sum == 3 else 0.0)
        r_normal = 5.0 * normal_align
        r_stable = -1.4 * obj_speed - 0.22 * obj_ang_speed - 0.6 * speed_inc
        r_ctrl = -0.0015 * np.sum(np.square(action))
        r_smooth = -0.0030 * np.sum(np.square(action - self.prev_action))

        # 如果一开始就乱撞到物体，不要直接高奖励。
        early_contact_penalty = 0.0
        if self.phase == 1 and self.step_count < 4 and contact_sum > 0:
            early_contact_penalty = -1.0 * contact_sum

        if self.phase == 1:
            reward = r_geom + 0.25 * r_contact + 0.25 * r_normal + r_ctrl + r_smooth + early_contact_penalty
        elif self.phase == 2:
            reward = 0.9 * r_geom + 1.0 * r_contact + 0.8 * r_normal + 0.7 * r_stable + 1.2 * r_ctrl + 1.2 * r_smooth
        elif self.phase == 3:
            reward = 0.7 * r_geom + 1.2 * r_contact + 1.2 * r_normal + 1.0 * r_stable + 1.2 * r_ctrl + 1.2 * r_smooth + 0.8
        else:
            reward = 1.4 * r_contact + 1.6 * r_normal + 1.0 * r_stable + 1.2

        success = (
            self.phase == 4 and
            self.freeze_steps >= self.max_freeze_steps and
            contact_sum >= 2 and
            normal_align > 0.55
        )
        if success:
            reward += 100.0
            self.success_hold = True

        self.prev_geom_err = geom_err
        self.prev_obj_speed = obj_speed
        self.prev_action = action.copy()

        info = {
            "reward_total": float(reward),
            "phase": int(self.phase),
            "geom_err": float(geom_err),
            "geom_progress": float(geom_prog),
            "normal_align": float(normal_align),
            "obj_speed": float(obj_speed),
            "obj_ang_speed": float(obj_ang_speed),
            "contact_sum": int(contact_sum),
            "th_contact": int(th_c),
            "ff_contact": int(ff_c),
            "mf_contact": int(mf_c),
            "contact_streak": int(self.contact_streak),
            "loss_streak": int(self.loss_streak),
            "freeze_steps": int(self.freeze_steps),
            "success": bool(success),
        }
        return float(reward), info

    # ===== helpers =====
    def _geom_ids(self, names, tag):
        ids = []
        for n in names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            if gid < 0:
                raise ValueError(f"找不到 {tag} geom: {n}")
            ids.append(int(gid))
        return ids

    def _get_site_pos(self, site_name):
        return self.data.site_xpos[self.site_ids[site_name]].copy()

    def _get_object_pos(self):
        return self.data.xpos[self.object_bid].copy()

    def _get_object_velocity(self):
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_bid, vel, 0
        )
        angvel = vel[:3].copy()
        linvel = vel[3:].copy()
        return linvel, angvel

    def _geom_group_in_contact(self, geom_ids, object_gid):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == object_gid and g2 in geom_ids:
                return 1
            if g2 == object_gid and g1 in geom_ids:
                return 1
        return 0

    def _get_tripod_contacts(self):
        th = self._geom_group_in_contact(self.thumb_gids, self.object_gid)
        ff = self._geom_group_in_contact(self.index_gids, self.object_gid)
        mf = self._geom_group_in_contact(self.middle_gids, self.object_gid)
        return th, ff, mf

    def _normalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x / (np.linalg.norm(x) + 1e-9)

    def _solve_tripod_targets_from_workspace(
        self,
        h=0.015,
        tol_h=0.00025,
        tol_sym=0.0008,
        tol_base_height_orth=0.04,
        tol_normal_plane=0.04,
        tol_normal_height_align=0.95,
        min_base_len=0.003,
        max_trials=20000,
    ):
        """
        在三指 workspace 里找一个 thin-object tripod 几何：
        1. ff 和 mf 构成底边 base。
        2. thumb 到底边中点的连线是 height。
        3. ||height|| 严格接近物体厚度 h。
        4. height 和 base 近似垂直。
        5. 三个 workspace normal 都在三点形成的平面内。
        6. ff/mf normal 与 thumb normal 沿 height 方向相对，形成夹持。
        """
        best = None
        best_cost = 1e18

        n_th = len(self.ws_th_pos)
        n_ff = len(self.ws_ff_pos)
        n_mf = len(self.ws_mf_pos)

        for _ in range(max_trials):
            i_th = np.random.randint(0, n_th)
            i_ff = np.random.randint(0, n_ff)
            i_mf = np.random.randint(0, n_mf)

            pth = self.ws_th_pos[i_th]
            pff = self.ws_ff_pos[i_ff]
            pmf = self.ws_mf_pos[i_mf]

            nth = self._normalize(self.ws_th_nrm[i_th])
            nff = self._normalize(self.ws_ff_nrm[i_ff])
            nmf = self._normalize(self.ws_mf_nrm[i_mf])

            base_vec = pff - pmf
            base_len = float(np.linalg.norm(base_vec))
            if base_len < min_base_len:
                continue
            base_dir = base_vec / (base_len + 1e-9)

            mid = 0.5 * (pff + pmf)
            height_vec = pth - mid
            height_len = float(np.linalg.norm(height_vec))
            if height_len < 1e-9:
                continue
            height_dir = height_vec / (height_len + 1e-9)

            # 1. 厚度约束。h 就是物体厚度，不做大范围放宽。
            h_err = abs(height_len - h)
            if h_err > tol_h:
                continue

            # 2. 等腰/中垂线约束。这个约束会让 pth 更接近 ff-mf 的中垂线。
            l1 = float(np.linalg.norm(pth - pff))
            l2 = float(np.linalg.norm(pth - pmf))
            sym = abs(l1 - l2)
            if sym > tol_sym:
                continue

            # 3. base 与 height 应该垂直。
            base_height_orth_err = abs(float(np.dot(base_dir, height_dir)))
            if base_height_orth_err > tol_base_height_orth:
                continue

            # 4. 三点形成的面。
            plane_n = np.cross(base_dir, height_dir)
            plane_n_norm = float(np.linalg.norm(plane_n))
            if plane_n_norm < 1e-8:
                continue
            plane_n = plane_n / plane_n_norm

            # 5. 法向量应该和三点形成的面平行，也就是与 plane_n 垂直。
            normal_plane_err = (
                abs(float(np.dot(nth, plane_n))) +
                abs(float(np.dot(nff, plane_n))) +
                abs(float(np.dot(nmf, plane_n)))
            ) / 3.0
            if normal_plane_err > tol_normal_plane:
                continue

            # 6. 法向应沿 height 方向夹持。
            # candidate +1: ff/mf normal ~= +height, thumb normal ~= -height
            # candidate -1: ff/mf normal ~= -height, thumb normal ~= +height
            score_pos = (
                float(np.dot(nff, height_dir)) +
                float(np.dot(nmf, height_dir)) +
                float(np.dot(nth, -height_dir))
            ) / 3.0
            score_neg = (
                float(np.dot(nff, -height_dir)) +
                float(np.dot(nmf, -height_dir)) +
                float(np.dot(nth, height_dir))
            ) / 3.0

            if score_pos >= score_neg:
                normal_sign = 1.0
                normal_height_score = score_pos
                nff_tgt = height_dir.copy()
                nmf_tgt = height_dir.copy()
                nth_tgt = -height_dir.copy()
            else:
                normal_sign = -1.0
                normal_height_score = score_neg
                nff_tgt = -height_dir.copy()
                nmf_tgt = -height_dir.copy()
                nth_tgt = height_dir.copy()

            if normal_height_score < tol_normal_height_align:
                continue

            # 7. thumb normal 应该和 ff/mf normal 相反。
            opposition_err = 0.5 * (
                abs(float(np.dot(nth, nff)) + 1.0) +
                abs(float(np.dot(nth, nmf)) + 1.0)
            )

            cost = (
                80.0 * h_err +
                8.0 * sym +
                4.0 * base_height_orth_err +
                4.0 * normal_plane_err +
                3.0 * (1.0 - normal_height_score) +
                1.0 * opposition_err
            )

            if cost < best_cost:
                best_cost = cost
                best = dict(
                    pth=pth.copy(),
                    pff=pff.copy(),
                    pmf=pmf.copy(),
                    nth=nth.copy(),
                    nff=nff.copy(),
                    nmf=nmf.copy(),
                    mid=mid.copy(),
                    base_dir=base_dir.copy(),
                    height_dir=height_dir.copy(),
                    plane_n=plane_n.copy(),
                    nth_tgt=nth_tgt.copy(),
                    nff_tgt=nff_tgt.copy(),
                    nmf_tgt=nmf_tgt.copy(),
                    normal_sign=float(normal_sign),
                    h=float(height_len),
                    h_err=float(h_err),
                    sym=float(sym),
                    base_len=float(base_len),
                    base_height_orth_err=float(base_height_orth_err),
                    normal_plane_err=float(normal_plane_err),
                    normal_height_score=float(normal_height_score),
                    opposition_err=float(opposition_err),
                    cost=float(cost),
                )

        return best

    def _place_object_from_tripod_solution(self, sol):
        pth = sol["pth"]
        pff = sol["pff"]
        pmf = sol["pmf"]
        mid = 0.5 * (pff + pmf)

        base_dir = self._normalize(pff - pmf)
        height_dir = self._normalize(pth - mid)

        # 如果 h 是物体厚度，物体中心更合理地放在 thumb 面和 ff/mf 面之间，
        # 也就是 pth 与底边中点 mid 的中点。
        if self.object_center_mode == "between_thumb_and_base":
            center = 0.5 * (pth + mid)
        elif self.object_center_mode == "ff_mf_mid":
            center = mid.copy()
        elif self.object_center_mode == "tripod_centroid":
            center = (pth + pff + pmf) / 3.0
        else:
            raise ValueError(f"未知 object_center_mode: {self.object_center_mode}")

        ux = base_dir
        uy = height_dir
        uz = np.cross(ux, uy)
        uz = self._normalize(uz)
        uy = self._normalize(np.cross(uz, ux))

        R = np.column_stack([ux, uy, uz]).astype(np.float64)

        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, R.reshape(-1))

        jadr = self.model.body_jntadr[self.object_bid]
        jnum = self.model.body_jntnum[self.object_bid]
        if jnum <= 0:
            return

        jid = jadr
        if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            return

        qpos_adr = self.model.jnt_qposadr[jid]
        qvel_adr = self.model.jnt_dofadr[jid]

        self.data.qpos[qpos_adr:qpos_adr + 3] = center
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = quat
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    def _compute_geom_err_to_target(self):
        if self.target_tripod is None:
            return 0.0

        th = self._get_site_pos("th_tip_site")
        ff = self._get_site_pos("ff_tip_site")
        mf = self._get_site_pos("mf_tip_site")

        e_th = np.linalg.norm(th - self.target_tripod["pth"])
        e_ff = np.linalg.norm(ff - self.target_tripod["pff"])
        e_mf = np.linalg.norm(mf - self.target_tripod["pmf"])

        return float((e_th + e_ff + e_mf) / 3.0)

    def _contact_normal(self, contact):
        # MuJoCo contact.frame 的前 3 个数是 contact normal 方向。
        R_contact = np.array(contact.frame, dtype=np.float64).reshape(3, 3)
        n = R_contact[0, :].copy()
        return self._normalize(n)

    def _compute_normal_align_score(self):
        if self.target_tripod is None:
            return 0.0

        # 这里用找点时同一个几何目标：normal 与 height 平行。
        # contact normal 的正负方向受 geom1/geom2 影响，所以 reward 用 abs(dot)。
        height_dir = self.target_tripod.get("height_dir", None)
        if height_dir is None:
            pth = self.target_tripod["pth"]
            pff = self.target_tripod["pff"]
            pmf = self.target_tripod["pmf"]
            mid = 0.5 * (pff + pmf)
            height_dir = self._normalize(pth - mid)
        else:
            height_dir = self._normalize(height_dir)

        score_th = score_ff = score_mf = 0.0
        has_th = has_ff = has_mf = False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            if not (g1 == self.object_gid or g2 == self.object_gid):
                continue

            other = g2 if g1 == self.object_gid else g1
            n = self._contact_normal(c)
            s = abs(float(np.dot(n, height_dir)))

            if other in self.thumb_gids:
                has_th = True
                score_th = max(score_th, s)
            elif other in self.index_gids:
                has_ff = True
                score_ff = max(score_ff, s)
            elif other in self.middle_gids:
                has_mf = True
                score_mf = max(score_mf, s)

        s_th = score_th if has_th else 0.0
        s_ff = score_ff if has_ff else 0.0
        s_mf = score_mf if has_mf else 0.0

        return float(np.clip((s_th + s_ff + s_mf) / 3.0, 0.0, 1.0))

    # ===== control =====
    def _default_ctrl(self):
        ctrl = np.zeros(self.nu, dtype=np.float32)
        for i in range(self.nu):
            lo, hi = self.model.actuator_ctrlrange[i]
            ctrl[i] = 0.5 * (lo + hi)
        return ctrl

    def _action_to_ctrl(self, action):
        if self.phase == 4 and self.freeze_ctrl is not None:
            ctrl = self.freeze_ctrl.copy()
            for aid, t in self.locked_act_target.items():
                ctrl[aid] = t
            self.prev_ctrl = ctrl.copy()
            return ctrl

        if self.action_type == "absolute":
            ctrl = np.zeros(self.nu, dtype=np.float32)
            for i in range(self.nu):
                lo, hi = self.model.actuator_ctrlrange[i]
                ctrl[i] = lo + (action[i] + 1.0) * 0.5 * (hi - lo)

        elif self.action_type == "delta":
            ctrl = self.prev_ctrl.copy()
            if self.phase == 1:
                eff = self.delta_scale
            elif self.phase == 2:
                eff = self.delta_scale * 0.5
            elif self.phase == 3:
                eff = self.delta_scale * 0.25
            else:
                eff = 0.0

            for i in range(self.nu):
                lo, hi = self.model.actuator_ctrlrange[i]
                delta = eff * action[i] * (hi - lo)
                ctrl[i] = np.clip(ctrl[i] + delta, lo, hi)
        else:
            raise ValueError(f"未知 action_type: {self.action_type}")

        for aid, t in self.locked_act_target.items():
            ctrl[aid] = t

        self.prev_ctrl = ctrl.copy()
        return ctrl

    # ===== termination =====
    def _check_terminated(self):
        obj_pos = self._get_object_pos()
        if np.linalg.norm(obj_pos) > 5.0:
            return True

        if self.phase in [2, 3] and self.loss_streak >= 8:
            return True

        if self.phase == 4 and self.freeze_steps >= self.max_freeze_steps and self.last_contact_sum >= 2:
            return True

        if self.phase == 4 and self.loss_streak >= 4:
            return True

        return False

    # ===== fallback =====
    def _randomize_object_pose_if_possible(self):
        jadr = self.model.body_jntadr[self.object_bid]
        jnum = self.model.body_jntnum[self.object_bid]
        if jnum <= 0:
            return

        jid = jadr
        if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            return

        qpos_adr = self.model.jnt_qposadr[jid]
        qvel_adr = self.model.jnt_dofadr[jid]

        self.data.qpos[qpos_adr + 0] = np.random.uniform(-0.01, 0.05)
        self.data.qpos[qpos_adr + 1] = np.random.uniform(-0.02, 0.04)
        self.data.qpos[qpos_adr + 2] = np.random.uniform(0.35, 0.41)

        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

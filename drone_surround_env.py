"""
多无人机包围强化学习环境（6 阶段 Curriculum + 简化奖励）

核心设计：
- 6 个阶段，难度逐步增加
- 前 3 阶段只要求距离，后 3 阶段再加入角度要求
- 奖励为：距离高斯奖励 + 队形安全 + 边界惩罚 + 成功保持奖励 + 终局奖励
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DroneSurroundEnv(gym.Env):

    def __init__(self, max_steps=600,
                 randomize_reset: bool = True,
                 reset_center_range: float = 50.0,
                 init_radius_range=(12.0, 15.0),
                 world_limit: float = 200.0,
                 curriculum: bool = False):
        super(DroneSurroundEnv, self).__init__()

        # 无人机名称
        self.hunter_names = ["Drone1", "Drone2", "Drone3"]
        self.target_name = "Drone0"

        # 环境参数
        self.max_steps = max_steps
        self.current_step = 0
        self.episode_num = 0
        self.success_hold_steps = 0

        # 简化奖励权重
        self.w_dist = 5.0         # 距离奖励权重
        self.w_safety = 1.0       # 队形安全（防止过近）
        self.w_shape = 5.0        # 角度分散奖励权重（仅在非 easy_success 时生效）
        self.w_boundary = 1.0     # 边界惩罚权重


        # 终局奖励
        self.R_success = 1000.0
        self.R_fail = -100.0
        self.R_timeout = -20.0

        # 队形安全参数
        self.d_safe = 3.5

        self.last_action = None

        # 动作与观测空间（保持和 MAPPO 一致）
        # allow slightly higher max speed for hunters (±6 m/s)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32)

        # 连接 AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.is_initialized = False

        # 随机化与边界控制
        self.randomize_reset = randomize_reset
        self.reset_center_range = float(reset_center_range)
        self.init_radius_range = (float(init_radius_range[0]), float(init_radius_range[1]))
        self.world_limit = float(world_limit)

        # =============== 6 阶段 Curriculum 配置 ===============
        self.curriculum_enabled = curriculum

        # Stage 0:
        self.stage0_r_min = 4.0
        self.stage0_r_max = 10.0
        self.stage0_target_speed = 0.0
        self.stage0_success_threshold = 3
        self.stage0_max_steps = 350

        # Stage 1:
        self.stage1_r_min = 5.0
        self.stage1_r_max = 9.0
        self.stage1_target_speed = 0.5
        self.stage1_success_threshold = 4
        self.stage1_max_steps = 400

        # Stage 2:
        self.stage2_r_min = 4.0
        self.stage2_r_max = 8.0
        self.stage2_target_speed = 1.0
        self.stage2_success_threshold = 5
        self.stage2_max_steps = 450

        # Stage 3:
        self.stage3_r_min = 4.0
        self.stage3_r_max = 8.0
        self.stage3_target_speed = 1.5
        self.stage3_success_threshold = 6
        self.stage3_max_steps = 500
        self.stage3_angle_min_deg = 45.0

        # Stage 4:
        self.stage4_r_min = 5.0
        self.stage4_r_max = 7.0
        self.stage4_target_speed = 2.0
        self.stage4_success_threshold = 5
        self.stage4_max_steps = 550
        self.stage4_angle_min_deg = 75.0

        # Stage 5:
        self.stage5_r_min = 5.0
        self.stage5_r_max = 7.0
        self.stage5_target_speed = 2.0
        self.stage5_success_threshold = 6
        self.stage5_max_steps = 600
        self.stage5_angle_min_deg = 90.0

        # Stage 6:
        self.stage6_r_min = 5.0
        self.stage6_r_max = 7.0
        self.stage6_target_speed = 2.0
        self.stage6_success_threshold = 7
        self.stage6_max_steps = 650
        self.stage6_angle_min_deg = 120.0

        # easy_success = True 时只看距离；False 时再看角度分散
        self.easy_success = True
        # 当前角度阈值（弧度），只在 easy_success=False 时使用
        self.angle_min_diff = np.pi / 3  # 默认 60°，会在 _apply_curriculum_stage 中覆盖

        # 初始化阶段
        if self.curriculum_enabled:
            self.curriculum_stage = 0
            self._apply_curriculum_stage(0)
        else:
            # 不启用 curriculum 时，直接用最高难度（Stage 6）
            self.curriculum_stage = 6
            self._apply_curriculum_stage(6)

    # =====================================================
    # Curriculum 阶段参数应用
    # =====================================================
    def _apply_curriculum_stage(self, stage: int):
        """根据 stage 设置 r_min/r_max、target_speed、success_threshold 等参数"""
        if stage == 0:
            self.r_min = self.stage0_r_min
            self.r_max = self.stage0_r_max
            self.target_speed = self.stage0_target_speed
            self.success_threshold = self.stage0_success_threshold
            self.max_steps = self.stage0_max_steps
            self.easy_success = True
            self.angle_min_diff = np.pi / 3  # 不会用到
            # 在容易阶段不使用角度奖励
            self.w_shape = 0.0

        elif stage == 1:
            self.r_min = self.stage1_r_min
            self.r_max = self.stage1_r_max
            self.target_speed = self.stage1_target_speed
            self.success_threshold = self.stage1_success_threshold
            self.max_steps = self.stage1_max_steps
            self.easy_success = True
            self.angle_min_diff = np.pi / 3  # 不会用到
            self.w_shape = 0.0

        elif stage == 2:
            self.r_min = self.stage2_r_min
            self.r_max = self.stage2_r_max
            self.target_speed = self.stage2_target_speed
            self.success_threshold = self.stage2_success_threshold
            self.max_steps = self.stage2_max_steps
            self.easy_success = True
            self.angle_min_diff = np.pi / 3  # 不会用到
            self.w_shape = 0.0

        elif stage == 3:
            self.r_min = self.stage3_r_min
            self.r_max = self.stage3_r_max
            self.target_speed = self.stage3_target_speed
            self.success_threshold = self.stage3_success_threshold
            self.max_steps = self.stage3_max_steps
            self.easy_success = False
            self.angle_min_diff = np.deg2rad(self.stage3_angle_min_deg)
            # Stage 3: temporarily increase angle-shape weight to 7.0 as requested
            self.w_shape = 7.0

        elif stage == 4:
            self.r_min = self.stage4_r_min
            self.r_max = self.stage4_r_max
            self.target_speed = self.stage4_target_speed
            self.success_threshold = self.stage4_success_threshold
            self.max_steps = self.stage4_max_steps
            self.easy_success = False
            self.angle_min_diff = np.deg2rad(self.stage4_angle_min_deg)
            # later stages use default moderate angle weight
            self.w_shape = 7.0

        elif stage == 5:
            self.r_min = self.stage5_r_min
            self.r_max = self.stage5_r_max
            self.target_speed = self.stage5_target_speed
            self.success_threshold = self.stage5_success_threshold
            self.max_steps = self.stage5_max_steps
            self.easy_success = False
            self.angle_min_diff = np.deg2rad(self.stage5_angle_min_deg)
            self.w_shape = 10.0

        elif stage == 6:
            # Stage6: same as Stage5 but faster target_speed
            self.r_min = self.stage6_r_min
            self.r_max = self.stage6_r_max
            self.target_speed = self.stage6_target_speed
            self.success_threshold = self.stage6_success_threshold
            self.max_steps = self.stage6_max_steps
            self.easy_success = False
            self.angle_min_diff = np.deg2rad(self.stage6_angle_min_deg)
            self.w_shape = 5.0

        else:
            # Fallback to Stage 5 semantics if an out-of-range stage provided
            self._apply_curriculum_stage(5)

        print(f"[Env] Apply Stage {stage}: "
              f"radius[{self.r_min},{self.r_max}], speed={self.target_speed}, "
              f"threshold={self.success_threshold}, max_steps={self.max_steps}, "
              f"easy_success={self.easy_success}, angle_min_diff={np.rad2deg(self.angle_min_diff):.1f}°")

    # =====================================================
    # 成功条件：距离 + （可选）角度均匀
    # =====================================================
    def _check_success_conditions(self, hunter_positions, target_pos):
        """
        成功条件：
        1. 所有追击者距离目标在 [r_min, r_max]
        2. easy_success=True 时只看距离
        3. 否则再要求角度分散（最小夹角 > self.angle_min_diff）
        """
        distances = [np.linalg.norm(hp[:2] - target_pos[:2]) for hp in hunter_positions]
        radius_ok = all(self.r_min <= d <= self.r_max for d in distances)
        if not radius_ok:
            return False

        if self.easy_success:
            return True

        # 角度条件
        angles = []
        for hp in hunter_positions:
            dx = hp[0] - target_pos[0]
            dy = hp[1] - target_pos[1]
            angles.append(np.arctan2(dy, dx))

        angles_sorted = sorted(angles)
        diffs = []
        for i in range(3):
            if i < 2:
                diff = angles_sorted[i + 1] - angles_sorted[i]
            else:
                diff = angles_sorted[0] + 2 * np.pi - angles_sorted[2]
            diffs.append(diff)

        # 使用当前阶段设定的最小角度阈值
        min_angle_diff = self.angle_min_diff
        angle_ok = all(d > min_angle_diff for d in diffs)
        return angle_ok

    # =====================================================
    # Gym 接口：reset
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_num += 1
        self.success_hold_steps = 0

        self.client.reset()

        for name in self.hunter_names + [self.target_name]:
            self.client.enableApiControl(True, name)
            self.client.armDisarm(True, name)

        # 随机初始位置
        if self.randomize_reset:
            margin = 30.0
            center_x = float(np.random.uniform(-self.reset_center_range, self.reset_center_range))
            center_y = float(np.random.uniform(-self.reset_center_range, self.reset_center_range))
            center_x = np.clip(center_x, -self.world_limit + margin, self.world_limit - margin)
            center_y = np.clip(center_y, -self.world_limit + margin, self.world_limit - margin)
            radius = float(np.random.uniform(self.init_radius_range[0], self.init_radius_range[1]))
            theta = float(np.random.uniform(0.0, 2 * np.pi))
        else:
            center_x, center_y = 0.0, 0.0
            radius = 10.0
            theta = 0.0

        # 三机初始位置：等边三角形绕目标
        local_pts = np.array([
            [radius, 0.0],
            [-radius / 2, radius * 0.866],
            [-radius / 2, -radius * 0.866],
        ], dtype=np.float32)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        rot_pts = (R @ local_pts.T).T

        hunter_positions = [
            (center_x + rot_pts[0, 0], center_y + rot_pts[0, 1], -15.0),
            (center_x + rot_pts[1, 0], center_y + rot_pts[1, 1], -15.0),
            (center_x + rot_pts[2, 0], center_y + rot_pts[2, 1], -15.0),
        ]
        target_pos = (center_x, center_y, -15.0)

        for i, name in enumerate(self.hunter_names):
            pose = airsim.Pose()
            pose.position.x_val, pose.position.y_val, pose.position.z_val = hunter_positions[i]
            self.client.simSetVehiclePose(pose, True, name)

        pose = airsim.Pose()
        pose.position.x_val, pose.position.y_val, pose.position.z_val = target_pos
        self.client.simSetVehiclePose(pose, True, self.target_name)

        for name in self.hunter_names + [self.target_name]:
            self.client.enableApiControl(True, name)
            self.client.armDisarm(True, name)
            self.client.simGetCollisionInfo(name)

        self.last_avg_distance = None
        self.last_action = None
        self.success_hold_steps = 0

        self._move_target_randomly()
        self.is_initialized = True

        observation = self._get_observation()
        info = {}
        return observation, info

    # =====================================================
    # Gym 接口：step
    # =====================================================
    def step(self, action):
        self.current_step += 1

        # 追击者执行动作
        for i, name in enumerate(self.hunter_names):
            vx = float(action[i * 2])
            vy = float(action[i * 2 + 1])
            self.client.moveByVelocityZAsync(vx, vy, -15, 0.1, vehicle_name=name)

        # 目标移动
        self._move_target_randomly()

        # 获取观测 + 计算奖励
        observation = self._get_observation()
        reward, collision, out_of_bounds, info = self._compute_reward(action)

        terminated = False
        if info.get('success', False):
            terminated = True
            reward += self.R_success
        elif collision:
            terminated = True
            reward += self.R_fail
        elif out_of_bounds:
            terminated = True
            reward += self.R_fail

        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            if not terminated:
                reward += self.R_timeout
                info['termination_reason'] = 'timeout'

        self.last_action = action.copy()
        return observation, reward, terminated, truncated, info

    # =====================================================
    # 构造观测向量（39 维）
    # =====================================================
    def _get_observation(self):
        obs = []

        target_state = self.client.getMultirotorState(self.target_name)
        target_pos = np.array([
            target_state.kinematics_estimated.position.x_val,
            target_state.kinematics_estimated.position.y_val,
            target_state.kinematics_estimated.position.z_val
        ])

        hunter_states = []
        for name in self.hunter_names:
            state = self.client.getMultirotorState(name)
            pos = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val
            ])
            vel = np.array([
                state.kinematics_estimated.linear_velocity.x_val,
                state.kinematics_estimated.linear_velocity.y_val,
                state.kinematics_estimated.linear_velocity.z_val
            ])
            hunter_states.append((pos, vel))

        for i in range(3):
            my_pos, my_vel = hunter_states[i]

            # 自身位置 (2)
            obs.extend([my_pos[0], my_pos[1]])
            # 自身速度 (2)
            obs.extend([my_vel[0], my_vel[1]])
            # 相对目标位置 (2)
            rel_target_pos = np.array([target_pos[0] - my_pos[0], target_pos[1] - my_pos[1]])
            obs.extend(rel_target_pos)
            # 距离目标 (1)
            dist_to_target = np.linalg.norm(rel_target_pos)
            obs.append(dist_to_target)

            # 相对队友位置 (2 + 2)
            other_indices = [j for j in range(3) if j != i]
            for j in other_indices:
                other_pos, _ = hunter_states[j]
                rel_teammate_pos = [other_pos[0] - my_pos[0], other_pos[1] - my_pos[1]]
                obs.extend(rel_teammate_pos)

            # 到边界距离 (2)
            dist_to_boundary_x = self.world_limit - abs(my_pos[0])
            dist_to_boundary_y = self.world_limit - abs(my_pos[1])
            obs.extend([dist_to_boundary_x, dist_to_boundary_y])

        return np.array(obs, dtype=np.float32)

    # =====================================================
    # 目标运动逻辑：根据当前阶段的 target_speed
    # =====================================================
    def _move_target_randomly(self):
        speed = self.target_speed

        # 静止
        if speed <= 0.0:
            self.client.moveByVelocityZAsync(0.0, 0.0, -15, 0.1, vehicle_name=self.target_name)
            return

        target_state = self.client.getMultirotorState(self.target_name)
        target_pos_2d = np.array([
            target_state.kinematics_estimated.position.x_val,
            target_state.kinematics_estimated.position.y_val
        ])

        hunter_positions_2d = []
        for name in self.hunter_names:
            state = self.client.getMultirotorState(name)
            pos_2d = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val
            ])
            hunter_positions_2d.append(pos_2d)

        hunters_center_2d = np.mean(hunter_positions_2d, axis=0)

        # 基础逃跑方向：从猎手中心指向目标（即远离猎手）
        escape_dir = target_pos_2d - hunters_center_2d
        norm = np.linalg.norm(escape_dir)
        if norm > 1e-6:
            escape_dir = escape_dir / norm
        else:
            escape_dir = np.array([1.0, 0.0], dtype=np.float32)

        # 靠近边界时朝地图中心偏转
        dist_x = self.world_limit - abs(target_pos_2d[0])
        dist_y = self.world_limit - abs(target_pos_2d[1])
        if min(dist_x, dist_y) < 50:
            dir_center = -target_pos_2d
            center_norm = np.linalg.norm(dir_center)
            if center_norm > 1e-6:
                dir_center = dir_center / center_norm
                escape_dir = dir_center

        # 加一点噪声，避免完全规则
        noise = np.random.normal(0, 0.05, size=2)
        escape_dir = escape_dir + noise
        norm_final = np.linalg.norm(escape_dir)
        if norm_final > 1e-6:
            escape_dir = escape_dir / norm_final

        vx = float(escape_dir[0] * speed)
        vy = float(escape_dir[1] * speed)
        self.client.moveByVelocityZAsync(vx, vy, -15, 0.1, vehicle_name=self.target_name)

    # =====================================================
    # 奖励函数：距离 + 安全 + 边界 + 成功保持
    # =====================================================
    def _compute_reward(self, action):
        """
        R = w_dist * R_dist + w_safety * R_safety - w_boundary * boundary_penalty
        + 成功保持奖励（在环内多待几步）+ 终局奖励（在 step 外加）
        """
        reward = 0.0
        info = {}

        target_state = self.client.getMultirotorState(self.target_name)
        target_pos = np.array([
            target_state.kinematics_estimated.position.x_val,
            target_state.kinematics_estimated.position.y_val,
            target_state.kinematics_estimated.position.z_val
        ])

        hunter_positions = []
        for name in self.hunter_names:
            state = self.client.getMultirotorState(name)
            pos = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val
            ])
            hunter_positions.append(pos)

        distances_2d = []
        for pos in hunter_positions:
            dist_2d = np.sqrt((pos[0] - target_pos[0]) ** 2 + (pos[1] - target_pos[1]) ** 2)
            distances_2d.append(dist_2d)

        avg_distance = np.mean(distances_2d)
        info['avg_distance'] = float(avg_distance)
        info['distances_2d'] = [float(d) for d in distances_2d]

        # 1. 距离奖励：在 [r_min, r_max] 中心附近最高（高斯）
        target_r = 0.5 * (self.r_min + self.r_max)
        sigma = max(0.5 * (self.r_max - self.r_min), 1.0)
        R_dist = np.exp(-((avg_distance - target_r) ** 2) / (2 * sigma ** 2))
        # 映射到 [-1, 1]
        R_dist = (R_dist - 0.5) * 2.0
        R_dist = np.clip(R_dist, -1.0, 1.0)

        # 2b. 角度分散奖励（仅在非 easy_success 阶段启用，辅助性）
        R_shape = 0.0
        if not self.easy_success:
            angles = []
            for pos in hunter_positions:
                dx = pos[0] - target_pos[0]
                dy = pos[1] - target_pos[1]
                angles.append(np.arctan2(dy, dx))
            angles_sorted = sorted(angles)
            diffs = []
            for i in range(3):
                if i < 2:
                    diff = angles_sorted[i + 1] - angles_sorted[i]
                else:
                    diff = angles_sorted[0] + 2 * np.pi - angles_sorted[2]
                diffs.append(diff)
            min_angle = float(np.min(diffs))
            # formula: (min_angle / pi - 0.3), clip to [-1,1]
            R_shape = (min_angle / np.pi - 0.3)
            R_shape = np.clip(R_shape, -1.0, 1.0)
        else:
            R_shape = 0.0

        # 2. 安全奖励：猎手之间太近则惩罚
        R_safety = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                dij = np.linalg.norm(hunter_positions[i][:2] - hunter_positions[j][:2])
                if dij < self.d_safe:
                    R_safety -= (self.d_safe - dij)
        R_safety = np.clip(R_safety, -2.0, 0.0)

        # 总奖励（含角度分散辅助项）
        reward = self.w_dist * R_dist + self.w_safety * R_safety + self.w_shape * R_shape

        info['reward_components'] = {
            'dist': float(R_dist),
            'shape': float(R_shape),
            'safety': float(R_safety),
            'total_before_boundary': float(reward)
        }

        # 碰撞检测
        collision = False
        for name in self.hunter_names:
            collision_info = self.client.simGetCollisionInfo(name)
            if collision_info.has_collided:
                collision = True
                info['collision'] = True
                info['termination_reason'] = 'collision'
                reward = 0.0
                break

        # 出界 + 边界惩罚
        out_of_bounds = False
        if not collision:
            boundary_penalty = 0.0
            warning_distance = 50.0

            for name in self.hunter_names + [self.target_name]:
                state = self.client.getMultirotorState(name)
                pos = state.kinematics_estimated.position

                dist_to_boundary = min(
                    self.world_limit - abs(pos.x_val),
                    self.world_limit - abs(pos.y_val)
                )

                if dist_to_boundary < warning_distance:
                    ratio = (warning_distance - dist_to_boundary) / warning_distance
                    boundary_penalty += ratio * 5.0

                if abs(pos.x_val) > self.world_limit or abs(pos.y_val) > self.world_limit:
                    out_of_bounds = True
                    info['out_of_bounds'] = True
                    info['termination_reason'] = 'out_of_bounds'
                    reward = 0.0
                    break

            if not out_of_bounds:
                boundary_penalty = np.clip(boundary_penalty, 0.0, 20.0)
                reward -= self.w_boundary * boundary_penalty
                info['boundary_penalty'] = float(boundary_penalty)

        # 成功条件检查 + 保持奖励
        if not collision and not out_of_bounds:
            success_conditions_met = self._check_success_conditions(hunter_positions, target_pos)
            if success_conditions_met:
                self.success_hold_steps += 1
                hold_bonus = min(self.success_hold_steps * 1.0, 10.0)
                reward += hold_bonus
                info['success_hold_steps'] = self.success_hold_steps
                if self.success_hold_steps >= self.success_threshold:
                    info['success'] = True
                    info['termination_reason'] = 'success'
                else:
                    info['success'] = False
            else:
                self.success_hold_steps = 0
                info['success'] = False
        else:
            self.success_hold_steps = 0
            info['success'] = False

        if self.current_step >= self.max_steps and 'termination_reason' not in info:
            info['termination_reason'] = 'timeout'

        info['reward_total'] = float(reward)
        return reward, collision, out_of_bounds, info

    # =====================================================
    # 其它工具函数
    # =====================================================
    def get_positions(self):
        hunter_positions = []
        for name in self.hunter_names:
            state = self.client.getMultirotorState(name)
            pos = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val
            ])
            hunter_positions.append(pos)

        target_state = self.client.getMultirotorState(self.target_name)
        target_pos = np.array([
            target_state.kinematics_estimated.position.x_val,
            target_state.kinematics_estimated.position.y_val,
            target_state.kinematics_estimated.position.z_val
        ])

        return hunter_positions, target_pos

    def close(self):
        for name in self.hunter_names + [self.target_name]:
            self.client.enableApiControl(False, name)
            self.client.armDisarm(False, name)

    def render(self):
        pass

    # =====================================================
    # Curriculum 控制接口（给 MAPPO 用）
    # =====================================================
    def advance_curriculum(self):
        """升一个阶段（0~6）"""
        if not self.curriculum_enabled:
            return
        if self.curriculum_stage < 6:
            self.curriculum_stage += 1
            self._apply_curriculum_stage(self.curriculum_stage)
            print(f"[Curriculum] ↑ Advance to Stage {self.curriculum_stage}")

    def degrade_curriculum(self):
        """降一个阶段（0~6）"""
        if not self.curriculum_enabled:
            return
        if self.curriculum_stage > 0:
            self.curriculum_stage -= 1
            self._apply_curriculum_stage(self.curriculum_stage)
            print(f"[Curriculum] ↓ Degrade to Stage {self.curriculum_stage}")

    def set_curriculum_stage(self, stage: int):
        """外部直接设置阶段（用于训练/评估环境同步）"""
        if not self.curriculum_enabled:
            return
        if stage < 0 or stage > 6:
            return
        self.curriculum_stage = stage
        self._apply_curriculum_stage(stage)

    def get_curriculum_stage(self):
        return self.curriculum_stage

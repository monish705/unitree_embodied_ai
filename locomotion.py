"""
motor/locomotion.py
====================
G1 locomotion using Unitree's trained RL velocity policy.

Loads the ONNX checkpoint from unitree_rl_mjlab and runs inference
at 50Hz (step_dt=0.02) to produce walking joint targets.

Policy observation (96-dim):
  base_ang_vel (3) + projected_gravity (3) + velocity_commands (3) +
  joint_pos_rel (29) + joint_vel_rel (29) + last_action (29)

Policy action (29-dim):
  Joint position targets = action * scale + offset
"""
import numpy as np
import mujoco
import os
import yaml
import math
import sys
from typing import Optional, Tuple

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from motor.unitree_bridge import UnitreeBridge, G1_JOINT_NAMES


# Paths relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POLICY_DIR = os.path.join(
    _PROJECT_ROOT, "unitree_rl_mjlab", "deploy", "robots", "g1",
    "config", "policy", "velocity", "v0"
)
_ONNX_PATH = os.path.join(_POLICY_DIR, "exported", "policy.onnx")
_YAML_PATH = os.path.join(_POLICY_DIR, "params", "deploy.yaml")


class LocomotionController:
    """
    Locomotion using Unitree's trained RL policy (ONNX).

    Usage:
        loco = LocomotionController(bridge)
        loco.walk_to(target_x, target_y)
    """

    def __init__(self, bridge: UnitreeBridge):
        self.bridge = bridge
        self.model = bridge.model
        self.data = bridge.data

        # Load deploy config
        with open(_YAML_PATH, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.step_dt = self.cfg['step_dt']  # 0.02s = 50Hz policy
        self.sim_dt = self.model.opt.timestep   # usually 0.002s
        self.decimation = max(1, int(self.step_dt / self.sim_dt))

        # Joint mapping
        self.joint_ids = self.cfg['joint_ids_map']  # [0..28]
        self.n_joints = len(self.joint_ids)

        # PD gains from the policy
        self.stiffness = np.array(self.cfg['stiffness'], dtype=np.float64)
        self.damping = np.array(self.cfg['damping'], dtype=np.float64)

        # Default pose & action scaling
        self.default_pos = np.array(self.cfg['default_joint_pos'], dtype=np.float64)
        self.action_scale = np.array(self.cfg['actions']['JointPositionAction']['scale'],
                                      dtype=np.float64)
        self.action_offset = np.array(self.cfg['actions']['JointPositionAction']['offset'],
                                       dtype=np.float64)

        # Velocity command limits
        vel_ranges = self.cfg['commands']['base_velocity']['ranges']
        self.vx_range = vel_ranges['lin_vel_x']  # [-0.5, 1.0]
        self.vy_range = vel_ranges['lin_vel_y']  # [-0.5, 0.5]
        self.wz_range = vel_ranges['ang_vel_z']  # [-1.0, 1.0]

        # State for continuous simulation
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.step_counter = 0
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)

        # Load ONNX policy
        if HAS_ONNX and os.path.exists(_ONNX_PATH):
            self.session = ort.InferenceSession(
                _ONNX_PATH,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            print(f"[Loco] ✅ Loaded Unitree RL policy: {_ONNX_PATH}")
            print(f"       Input: {input_shape}, Output: {self.session.get_outputs()[0].shape}")
            print(f"       Step dt={self.step_dt}s, decimation={self.decimation}")
            self._has_policy = True

            # Override MuJoCo actuator PD gains to match training config
            self._apply_training_pd_gains()
        else:
            print(f"[Loco] ⚠️ ONNX policy not found at {_ONNX_PATH}")
            print(f"       Using fallback velocity controller")
            self._has_policy = False

    def update(self):
        """Continuous hook called by the bridge/server every physics step."""
        if self.step_counter % self.decimation == 0:
            if self._has_policy:
                obs = self._build_obs(self.cmd_vx, self.cmd_vy, self.cmd_wz)
                action = self.session.run([self.output_name], {self.input_name: obs})[0].flatten()
                self.last_action = action.astype(np.float32)
                targets = action * self.action_scale + self.action_offset
                for i, jid in enumerate(self.joint_ids):
                    self.data.ctrl[jid] = targets[i]
            else:
                self._fallback_inference()
        self.step_counter += 1

    def _fallback_inference(self):
        """Fallback: direct velocity injection."""
        cx, cy, cyaw = self._get_base_xytheta()
        cos_yaw = np.cos(cyaw)
        sin_yaw = np.sin(cyaw)
        self.data.qvel[0] = self.cmd_vx * cos_yaw - self.cmd_vy * sin_yaw
        self.data.qvel[1] = self.cmd_vx * sin_yaw + self.cmd_vy * cos_yaw
        self.data.qvel[2] = 0.0
        self.data.qvel[5] = self.cmd_wz

    def _apply_training_pd_gains(self):
        """
        Override the MuJoCo model's actuator gains to match the stiffness/damping
        from deploy.yaml. This is critical: the RL policy was trained with these
        specific PD parameters, and sim-to-sim transfer requires matching them.
        """
        for i, jid in enumerate(self.joint_ids):
            kp = self.stiffness[i]
            kd = self.damping[i]
            # MuJoCo position actuator:
            #   gainprm[0] = kp (proportional gain)
            #   biasprm[1] = -kp (position bias)
            #   biasprm[2] = -kd (velocity bias / damping)
            self.model.actuator_gainprm[jid, 0] = kp
            self.model.actuator_biasprm[jid, 1] = -kp
            self.model.actuator_biasprm[jid, 2] = -kd
        print(f"[Loco] Applied training PD gains to {len(self.joint_ids)} actuators")

    # ---- Observation Construction ----

    def _get_projected_gravity(self) -> np.ndarray:
        """Get gravity vector projected into the robot's base frame."""
        # Get pelvis rotation matrix
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        rot_mat = self.data.xmat[pelvis_id].reshape(3, 3)
        # World gravity = [0, 0, -9.81], project into body frame
        gravity_world = np.array([0, 0, -1.0])
        return (rot_mat.T @ gravity_world).astype(np.float32)

    def _get_base_ang_vel(self) -> np.ndarray:
        """Get base angular velocity in body frame."""
        imu = self.bridge.get_imu("pelvis")
        return imu['angular_velocity'].astype(np.float32)

    def _get_joint_pos_rel(self) -> np.ndarray:
        """Get joint positions relative to default pose."""
        all_pos = self.bridge.get_joint_positions()
        pos = np.array([all_pos.get(G1_JOINT_NAMES[i], 0.0)
                        for i in self.joint_ids], dtype=np.float32)
        return pos - self.default_pos.astype(np.float32)

    def _get_joint_vel(self) -> np.ndarray:
        """Get joint velocities."""
        all_vel = self.bridge.get_joint_velocities()
        return np.array([all_vel.get(G1_JOINT_NAMES[i], 0.0)
                         for i in self.joint_ids], dtype=np.float32)

    def _build_obs(self, cmd_vx: float, cmd_vy: float, cmd_wz: float) -> np.ndarray:
        """
        Build the 96-dim observation vector for the policy.
        Order: ang_vel(3) + gravity(3) + cmd(3) + joint_pos(29) + joint_vel(29) + last_action(29)
        """
        obs = np.concatenate([
            self._get_base_ang_vel(),           # 3
            self._get_projected_gravity(),       # 3
            np.array([cmd_vx, cmd_vy, cmd_wz], dtype=np.float32),  # 3
            self._get_joint_pos_rel(),           # 29
            self._get_joint_vel(),               # 29
            self.last_action,                    # 29
        ])
        return obs.reshape(1, -1).astype(np.float32)

    # ---- Policy Inference (internal, called only by update() hook) ----

    # ---- High-Level Commands ----

    def _get_base_xytheta(self) -> Tuple[float, float, float]:
        pos = self.bridge.get_base_pos()
        quat = self.bridge.get_base_quat()
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return pos[0], pos[1], yaw

    def walk_velocity(self, vx: float, vy: float, wz: float, duration_s: float = 1.0):
        """
        Send velocity command for a duration.
        """
        self.cmd_vx = np.clip(vx, self.vx_range[0], self.vx_range[1])
        self.cmd_vy = np.clip(vy, self.vy_range[0], self.vy_range[1])
        self.cmd_wz = np.clip(wz, self.wz_range[0], self.wz_range[1])

        # Step global simulation
        n_sim_steps = int(duration_s / self.sim_dt)
        for _ in range(n_sim_steps):
            self.bridge.step(1)
            
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
    
    def walk_to(self, target_x: float, target_y: float,
                 target_theta: Optional[float] = None,
                 tolerance: float = 0.3,
                 max_duration_s: float = 20.0) -> dict:
        """
        Walk to a target position using Artificial Potential Fields for navigation.
        This is the 'Control Layer' that handles obstacle avoidance.
        """
        target = np.array([target_x, target_y])
        positions = []
        n_max = int(max_duration_s / self.step_dt)
        kp_lin = 1.2
        kp_ang = 1.5

        # Stall detection state
        last_stall_check_pos = None
        stall_step = 0
        stall_penalty_dir = 1.0  # alternates +1/-1 to bias curl direction on escape

        for step in range(n_max):
            mujoco.mj_forward(self.model, self.data)
            cx, cy, cyaw = self._get_base_xytheta()
            current = np.array([cx, cy])
            dist = np.linalg.norm(target - current)
            positions.append(current.copy())

            if dist < tolerance:
                self.cmd_vx = 0.0
                self.cmd_vy = 0.0
                self.cmd_wz = 0.0
                print(f"[Loco] ✅ Arrived dist={dist:.3f}m")
                return {"success": True, "final_distance": dist, "steps": step, "positions": positions}

            # --- Stall detection: if <3cm moved in 100 steps, escape ---
            if step > 0 and step % 100 == 0:
                if last_stall_check_pos is not None:
                    moved = np.linalg.norm(current - last_stall_check_pos)
                    # ONLY stall if trying to move forward. Avoid false positives when intentionally turning in place.
                    if moved < 0.05 and self.cmd_vx > 0.05:
                        stall_step += 1
                        print(f"[Loco] ⚠️ Stall #{stall_step} at ({cx:.2f},{cy:.2f}), moved only {moved:.3f}m — escaping (dir={stall_penalty_dir:+.0f})")
                        # Back up briefly
                        self.cmd_vx = -0.3
                        self.cmd_wz = 0.0
                        for _ in range(int(0.8 / self.sim_dt)):
                            self.bridge.step(1)
                        # Turn opposite to curl bias so we try the other side
                        self.cmd_vx = 0.0
                        self.cmd_wz = 0.7 * stall_penalty_dir
                        for _ in range(int(0.6 / self.sim_dt)):
                            self.bridge.step(1)
                        stall_penalty_dir *= -1.0  # flip direction for next stall
                last_stall_check_pos = current.copy()

            # --- APF Navigation (Control Layer) ---
            K_att = 1.0
            K_rep = 0.3
            K_curl = 0.8     # Stronger curl to actually slide around walls
            D_thresh = 0.3   # Reduced from 0.6 — was overlapping in narrow gaps

            F_att_x = K_att * (target_x - cx)
            F_att_y = K_att * (target_y - cy)
            F_x, F_y = F_att_x, F_att_y

            for i in range(self.model.ngeom):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                if name and ("wall" in name or "block" in name):
                    g_pos = self.data.geom_xpos[i]
                    g_size = self.model.geom_size[i]
                    nx = np.clip(cx, g_pos[0]-g_size[0], g_pos[0]+g_size[0])
                    ny = np.clip(cy, g_pos[1]-g_size[1], g_pos[1]+g_size[1])
                    dx_g, dy_g = cx - nx, cy - ny
                    dist_g = math.hypot(dx_g, dy_g)

                    if dist_g < D_thresh and dist_g > 0.01:
                        rep_mag = K_rep * (1.0/dist_g - 1.0/D_thresh) / (dist_g**2)
                        F_x += rep_mag * (dx_g/dist_g)
                        F_y += rep_mag * (dy_g/dist_g)
                        # Vortex/Curl to slide around. Bias alternates on stall escape.
                        cross_p = dx_g * F_att_y - dy_g * F_att_x
                        curl_dir = (1.0 if cross_p > 0 else -1.0) * stall_penalty_dir
                        F_x += curl_dir * K_curl * rep_mag * (-dy_g/dist_g)
                        F_y += curl_dir * K_curl * rep_mag * (dx_g/dist_g)

            angle_to_target = math.atan2(F_y, F_x)
            angle_error = (angle_to_target - cyaw + math.pi) % (2*math.pi) - math.pi

            # KEY FIX: stop moving forward if needing to turn around drastically
            if abs(angle_error) > 0.8:
                # Needs to turn around (repelled by wall). Do NOT move forward. safely turn.
                self.cmd_vx = 0.0
                self.cmd_wz = float(np.clip(kp_ang * angle_error, -0.8, 0.8))
            elif abs(angle_error) > 0.6:
                # Moderate turn. Creep forward very slowly.
                self.cmd_vx = 0.1
                self.cmd_wz = float(np.clip(kp_ang * angle_error, -0.6, 0.6))
            elif abs(angle_error) > 0.2:
                # Medium: moderate turn + moderate speed
                self.cmd_vx = float(np.clip(kp_lin * dist * 0.5, 0.15, 0.5))
                self.cmd_wz = float(np.clip(kp_ang * angle_error, -0.5, 0.5))
            else:
                # Heading on target: full speed, small correction
                self.cmd_vx = float(np.clip(kp_lin * dist, 0.0, 0.8))
                self.cmd_wz = float(np.clip(kp_ang * angle_error, -0.3, 0.3))

            # Step simulation via bridge (which handles the continuous update hook)
            for _ in range(self.decimation):
                self.bridge.step(1)

            if self.bridge.get_base_pos()[2] < 0.5:
                print(f"[Loco] ❌ Fell at step {step}")
                return {"success": False, "final_distance": dist, "steps": step, "positions": positions}

            if step % 50 == 0:
                print(f"  [step {step}] pos=({cx:.3f},{cy:.3f}) dist={dist:.3f}m aerr={angle_error:.2f} cmd=({self.cmd_vx:.2f},{self.cmd_wz:.2f})")

        self.cmd_vx = 0.0
        self.cmd_wz = 0.0
        return {"success": False, "final_distance": dist, "steps": n_max, "positions": positions}

    def stand_still(self, duration_s: float = 2.0):
        """Hold zero velocity for a duration."""
        self.walk_velocity(0.0, 0.0, 0.0, duration_s)
        print(f"[Loco] Standing at {self.bridge.get_base_pos()}")


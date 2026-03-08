"""
motor/unitree_bridge.py
========================
Low-level interface between our pipeline and the G1's 29 joint actuators.
In simulation: reads/writes MuJoCo data directly.
On real robot: would use unitree_sdk2_python DDS interface (same API).
"""
import numpy as np
import mujoco
from typing import Dict, Optional, Tuple


# G1 29-DoF joint layout (maps joint name → actuator index)
G1_JOINT_NAMES = [
    # Left leg (0-5)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg (6-11)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (12-14)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (15-21)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (22-28)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Arm joint indices within the G1 actuator array
LEFT_ARM_INDICES = list(range(15, 22))
RIGHT_ARM_INDICES = list(range(22, 29))
LEFT_LEG_INDICES = list(range(0, 6))
RIGHT_LEG_INDICES = list(range(6, 12))
WAIST_INDICES = list(range(12, 15))

# End-effector body names
LEFT_HAND_BODY = "left_wrist_yaw_link"
RIGHT_HAND_BODY = "right_wrist_yaw_link"


class UnitreeBridge:
    """
    Abstraction layer for Unitree G1 motor control.
    
    In sim mode: directly interfaces with MuJoCo model/data.
    Designed so the same API can be swapped to unitree_sdk2_python for real robot.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # Build joint name → ID lookup
        self._joint_ids = {}
        self._actuator_ids = {}
        for i, name in enumerate(G1_JOINT_NAMES):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jid >= 0:
                self._joint_ids[name] = jid
            if aid >= 0:
                self._actuator_ids[name] = aid

        # Body IDs for end-effectors
        self._body_ids = {}
        for bname in [LEFT_HAND_BODY, RIGHT_HAND_BODY, "pelvis", "torso_link"]:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                self._body_ids[bname] = bid

        print(f"[Bridge] Initialised: {len(self._joint_ids)} joints, "
              f"{len(self._actuator_ids)} actuators, {len(self._body_ids)} bodies")
        self.pre_step_hooks = []

    def step(self, n_substeps: int = 1):
        """Step the MuJoCo simulation, running all pre-step hooks (e.g., GraspController.update)."""
        for _ in range(n_substeps):
            for hook in self.pre_step_hooks:
                hook()
            mujoco.mj_step(self.model, self.data)

    # ---- Joint Control ----

    def set_joint_positions(self, positions: Dict[str, float]):
        """Set target positions for named joints via position actuators."""
        for name, pos in positions.items():
            if name in self._actuator_ids:
                self.data.ctrl[self._actuator_ids[name]] = pos

    def set_ctrl_array(self, ctrl: np.ndarray):
        """Set all 29 actuator controls at once."""
        assert len(ctrl) == len(G1_JOINT_NAMES), f"Expected {len(G1_JOINT_NAMES)} controls, got {len(ctrl)}"
        self.data.ctrl[:len(ctrl)] = ctrl

    def get_joint_positions(self) -> Dict[str, float]:
        """Read current joint positions."""
        positions = {}
        for name, jid in self._joint_ids.items():
            # qpos index for this joint (account for freejoint at index 0)
            qadr = self.model.jnt_qposadr[jid]
            positions[name] = float(self.data.qpos[qadr])
        return positions

    def get_joint_velocities(self) -> Dict[str, float]:
        """Read current joint velocities."""
        velocities = {}
        for name, jid in self._joint_ids.items():
            vadr = self.model.jnt_dofadr[jid]
            velocities[name] = float(self.data.qvel[vadr])
        return velocities

    # ---- End-Effector ----

    def get_ee_pos(self, hand: str = "right") -> np.ndarray:
        """Get end-effector position in world frame."""
        body_name = RIGHT_HAND_BODY if hand == "right" else LEFT_HAND_BODY
        bid = self._body_ids.get(body_name)
        if bid is None:
            return np.zeros(3)
        return self.data.xpos[bid].copy()

    def get_ee_rot(self, hand: str = "right") -> np.ndarray:
        """Get end-effector rotation matrix (3x3) in world frame."""
        body_name = RIGHT_HAND_BODY if hand == "right" else LEFT_HAND_BODY
        bid = self._body_ids.get(body_name)
        if bid is None:
            return np.eye(3)
        return self.data.xmat[bid].reshape(3, 3).copy()

    # ---- Base / Pelvis ----

    def get_base_pos(self) -> np.ndarray:
        """Get pelvis position in world frame."""
        bid = self._body_ids.get("pelvis")
        if bid is None:
            return np.zeros(3)
        return self.data.xpos[bid].copy()

    def get_base_quat(self) -> np.ndarray:
        """Get pelvis quaternion (w, x, y, z)."""
        # Floating base qpos: first 3 = pos, next 4 = quat
        return self.data.qpos[3:7].copy()

    def get_base_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get base linear and angular velocity."""
        # Floating base qvel: first 3 = linear, next 3 = angular
        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()
        return lin_vel, ang_vel

    # ---- IMU ----

    def get_imu(self, location: str = "torso") -> Dict[str, np.ndarray]:
        """
        Read IMU sensor data.
        
        Args:
            location: "torso" or "pelvis"
        
        Returns:
            dict with 'angular_velocity' (3,) and 'linear_acceleration' (3,)
        """
        if location == "torso":
            gyro_name = "imu-torso-angular-velocity"
            acc_name = "imu-torso-linear-acceleration"
        else:
            gyro_name = "imu-pelvis-angular-velocity"
            acc_name = "imu-pelvis-linear-acceleration"

        result = {}
        gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, gyro_name)
        acc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, acc_name)

        if gyro_id >= 0:
            adr = self.model.sensor_adr[gyro_id]
            dim = self.model.sensor_dim[gyro_id]
            result["angular_velocity"] = self.data.sensordata[adr:adr+dim].copy()
        else:
            result["angular_velocity"] = np.zeros(3)

        if acc_id >= 0:
            adr = self.model.sensor_adr[acc_id]
            dim = self.model.sensor_dim[acc_id]
            result["linear_acceleration"] = self.data.sensordata[adr:adr+dim].copy()
        else:
            result["linear_acceleration"] = np.zeros(3)

        return result

    # ---- Keyframes ----

    def reset_to_stand(self):
        """
        Reset robot to the 'stand' keyframe defined in the G1 MJCF.
        
        IMPORTANT: The keyframe resets ALL qpos (including freejoint objects).
        We save and restore non-robot qpos to preserve object positions.
        Handles scenes where objects add extra qpos beyond the keyframe size.
        """
        # Save all qpos first (objects have freejoints that will be zeroed)
        saved_qpos = self.data.qpos.copy()
        
        robot_nq = 7 + 29  # floating_base(7) + 29 hinge joints
        
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id >= 0:
            try:
                mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
                # Restore qpos for non-robot joints (freejoints on objects)
                if len(saved_qpos) > robot_nq:
                    self.data.qpos[robot_nq:] = saved_qpos[robot_nq:]
                print("[Bridge] Reset to 'stand' keyframe (objects preserved)")
            except Exception:
                # Keyframe qpos size doesn't match model (scene has extra objects)
                # Manually apply the stand pose to robot joints only
                self._manual_stand(saved_qpos)
        else:
            # No keyframe found — apply manually
            self._manual_stand(saved_qpos)
    
    def _manual_stand(self, saved_qpos):
        """Manually set the robot to standing pose without using keyframe."""
        # Floating base: pos(3) + quat(4)
        self.data.qpos[0:3] = [0, 0, 0.79]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        # Legs: 12 joints, all zero
        self.data.qpos[7:19] = 0
        # Waist: 3 joints, all zero
        self.data.qpos[19:22] = 0
        # Left arm: [0.2, 0.2, 0, 1.28, 0, 0, 0]
        self.data.qpos[22:29] = [0.2, 0.2, 0, 1.28, 0, 0, 0]
        # Right arm: [0.2, -0.2, 0, 1.28, 0, 0, 0]
        self.data.qpos[29:36] = [0.2, -0.2, 0, 1.28, 0, 0, 0]
        # Preserve object qpos
        if len(saved_qpos) > 36:
            self.data.qpos[36:] = saved_qpos[36:]
        # Set ctrl to match
        self.data.ctrl[:] = 0
        self.data.ctrl[15:22] = [0.2, 0.2, 0, 1.28, 0, 0, 0]
        self.data.ctrl[22:29] = [0.2, -0.2, 0, 1.28, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        print("[Bridge] Reset to manual stand pose (objects preserved)")

    # ---- Utility ----

    def get_arm_joint_names(self, hand: str = "right") -> list:
        """Get the joint names for the specified arm."""
        indices = RIGHT_ARM_INDICES if hand == "right" else LEFT_ARM_INDICES
        return [G1_JOINT_NAMES[i] for i in indices]

    def get_arm_joint_positions(self, hand: str = "right") -> np.ndarray:
        """Get current joint angles for the specified arm."""
        names = self.get_arm_joint_names(hand)
        all_pos = self.get_joint_positions()
        return np.array([all_pos.get(n, 0.0) for n in names])

    def get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """Get joint position limits."""
        limits = {}
        for name, jid in self._joint_ids.items():
            if self.model.jnt_limited[jid]:
                lo = float(self.model.jnt_range[jid, 0])
                hi = float(self.model.jnt_range[jid, 1])
                limits[name] = (lo, hi)
        return limits

    def print_state(self):
        """Print a summary of the robot state."""
        base_pos = self.get_base_pos()
        r_ee = self.get_ee_pos("right")
        l_ee = self.get_ee_pos("left")
        imu = self.get_imu("torso")
        print(f"\n[Robot State]")
        print(f"  Base:     ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})")
        print(f"  R. Hand:  ({r_ee[0]:.3f}, {r_ee[1]:.3f}, {r_ee[2]:.3f})")
        print(f"  L. Hand:  ({l_ee[0]:.3f}, {l_ee[1]:.3f}, {l_ee[2]:.3f})")
        print(f"  IMU gyro: {imu['angular_velocity']}")

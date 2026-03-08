"""
motor/semantic_actions.py
==========================
ACTION GROUNDING LAYER — The missing piece between VLM semantic intent and motor commands.

This module solves the Perception-Action Semantic Gap:
  - VLMs cannot think in metric space (they hallucinate coordinates)
  - π0/RT-2 style: VLM does semantic reasoning, a separate layer does spatial math
  - This layer uses MuJoCo ground-truth (sim) or depth sensors (real) to convert
    semantic verbs ("walk toward the red mug") into grounded motor commands

The VLM should ONLY see methods from this class — never raw arm_move_to(x,y,z).
"""
import math
import numpy as np
import mujoco
from typing import Optional
from motor.unitree_bridge import UnitreeBridge, RIGHT_HAND_BODY, LEFT_HAND_BODY
from motor.locomotion import LocomotionController
from motor.arm_controller import ArmController
from motor.grasp_controller import GraspController


class SemanticActions:
    """Semantic action interface for VLM-driven robot control.
    
    Every method here is a grounded semantic verb that the VLM can call.
    No method exposes raw coordinates — all spatial math is handled internally
    using MuJoCo ground-truth state.
    """

    def __init__(self, bridge: UnitreeBridge,
                 locomotion: LocomotionController,
                 arm: ArmController,
                 grasp: GraspController):
        self.bridge = bridge
        self.model = bridge.model
        self.data = bridge.data
        self.loco = locomotion
        self.arm = arm
        self.grasp = grasp

        # Build a map of named scene objects (non-robot bodies with geoms)
        self._scene_objects = {}
        self._build_scene_map()

    # ------------------------------------------------------------------
    # Internal: Scene understanding via MuJoCo ground truth
    # ------------------------------------------------------------------

    def _build_scene_map(self):
        """Index all named non-robot bodies for semantic reference."""
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name not in ("world", "pelvis") and not name.endswith("_link"):
                self._scene_objects[name] = i

        # Also index named geoms that belong to worldbody (walls, floors, targets)
        self._scene_geoms = {}
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name not in ("floor",):
                body_id = self.model.geom_bodyid[i]
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name == "world" or body_name in self._scene_objects:
                    self._scene_geoms[name] = i

        n_obj = len(self._scene_objects)
        n_geom = len(self._scene_geoms)
        print(f"[SemanticActions] Scene map: {n_obj} objects, {n_geom} named geoms")

    def _get_robot_pos(self):
        """Get robot base (x, y) position in world frame."""
        return self.bridge.get_base_pos()[:2]

    def _get_robot_heading(self):
        """Get robot yaw angle in radians."""
        quat = self.bridge.get_base_quat()
        # Extract yaw from quaternion (w, x, y, z)
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _get_object_pos(self, name: str) -> Optional[np.ndarray]:
        """Get world position of a named object or geom."""
        mujoco.mj_forward(self.model, self.data)
        if name in self._scene_objects:
            return self.data.xpos[self._scene_objects[name]].copy()
        if name in self._scene_geoms:
            return self.data.geom_xpos[self._scene_geoms[name]].copy()
        return None

    def _get_navigation_target(self, name: str, stand_off: float = 0.25) -> Optional[np.ndarray]:
        """Get a reachable navigation target for a named object or geom.

        For solid box geoms like the green target zone, navigate to the nearest
        face just outside the box instead of the geom center or a corner.
        """
        target_pos = self._get_object_pos(name)
        if target_pos is None:
            return None

        if name in self._scene_geoms:
            gid = self._scene_geoms[name]
            geom_type = self.model.geom_type[gid]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                robot_pos = self._get_robot_pos()
                size = self.model.geom_size[gid]
                min_x = target_pos[0] - size[0] - stand_off
                max_x = target_pos[0] + size[0] + stand_off
                min_y = target_pos[1] - size[1] - stand_off
                max_y = target_pos[1] + size[1] + stand_off

                nav_pos = target_pos.copy()
                dx_out = 0.0
                if robot_pos[0] < min_x:
                    dx_out = robot_pos[0] - min_x
                elif robot_pos[0] > max_x:
                    dx_out = robot_pos[0] - max_x

                dy_out = 0.0
                if robot_pos[1] < min_y:
                    dy_out = robot_pos[1] - min_y
                elif robot_pos[1] > max_y:
                    dy_out = robot_pos[1] - max_y

                if min_x < robot_pos[0] < max_x and min_y < robot_pos[1] < max_y:
                    # If already inside the inflated bounds, keep current target center.
                    return target_pos

                if abs(dy_out) >= abs(dx_out):
                    nav_pos[0] = target_pos[0]
                    nav_pos[1] = min_y if robot_pos[1] < target_pos[1] else max_y
                else:
                    nav_pos[0] = min_x if robot_pos[0] < target_pos[0] else max_x
                    nav_pos[1] = target_pos[1]
                return nav_pos

        return target_pos

    def _distance_to_object_surface_xy(self, name: str, point_xy: np.ndarray) -> Optional[float]:
        if name not in self._scene_geoms:
            return None

        gid = self._scene_geoms[name]
        pos = self.data.geom_xpos[gid]
        size = self.model.geom_size[gid]
        geom_type = self.model.geom_type[gid]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            dx = max(abs(point_xy[0] - pos[0]) - size[0], 0.0)
            dy = max(abs(point_xy[1] - pos[1]) - size[1], 0.0)
            return float(np.hypot(dx, dy))

        radius = size[0]
        return float(max(np.hypot(point_xy[0] - pos[0], point_xy[1] - pos[1]) - radius, 0.0))

    def _micro_approach_target(self, target_xy: np.ndarray,
                               object_name: str,
                               surface_threshold: float = 0.1,
                               max_iters: int = 12):
        """Final short-range approach using direct velocity commands.

        This avoids the APF controller getting stuck oscillating against the
        green target box during the last few centimeters.
        """
        last_surface_dist = None
        worsening_steps = 0

        for _ in range(max_iters):
            robot_pos = self._get_robot_pos()
            heading = self._get_robot_heading()
            vec = target_xy - robot_pos
            dist = float(np.hypot(vec[0], vec[1]))
            surface_dist = self._distance_to_object_surface_xy(object_name, robot_pos)

            if surface_dist is not None and surface_dist <= surface_threshold:
                break
            if dist <= 0.03:
                break

            angle = math.atan2(vec[1], vec[0]) - heading
            angle = (angle + math.pi) % (2 * math.pi) - math.pi

            if abs(angle) > 0.12:
                self.loco.walk_velocity(
                    vx=0.0,
                    vy=0.0,
                    wz=float(np.clip(angle * 1.5, -0.5, 0.5)),
                    duration_s=0.25,
                )
            else:
                self.loco.walk_velocity(
                    vx=float(np.clip(0.08 + dist * 0.5, 0.08, 0.18)),
                    vy=0.0,
                    wz=float(np.clip(angle * 0.8, -0.15, 0.15)),
                    duration_s=0.25,
                )

            if surface_dist is not None and last_surface_dist is not None:
                if surface_dist >= last_surface_dist - 0.01:
                    worsening_steps += 1
                else:
                    worsening_steps = 0
            last_surface_dist = surface_dist

            if worsening_steps >= 3:
                break

    def _is_path_blocked(self, start_pos: np.ndarray, end_pos: np.ndarray) -> Optional[str]:
        """Returns the name of the first obstacle blocking the straight-line path, or None."""
        obstacles = []
        for name, gid in self._scene_geoms.items():
            if "wall" in name or "block" in name:
                pos = self.data.geom_xpos[gid]
                size = self.model.geom_size[gid]
                obstacles.append((name, pos[0]-size[0], pos[0]+size[0], pos[1]-size[1], pos[1]+size[1]))
                
        # Sample points along the segment (inflation = 0.2m for robot width)
        for t in np.linspace(0.1, 0.9, 15):
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            
            for obs_name, mx, Mx, my, My in obstacles:
                if (mx - 0.2) <= x <= (Mx + 0.2) and (my - 0.2) <= y <= (My + 0.2):
                    return obs_name
        return None

    def _relative_description(self, target_pos: np.ndarray, target_name: str = "") -> str:
        """Describe a target position relative to the robot in natural language."""
        robot_pos = self._get_robot_pos()
        heading = self._get_robot_heading()

        # Vector from robot to target in world frame
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        dist = np.sqrt(dx**2 + dy**2)

        # Angle to target in world frame, then relative to robot heading
        angle_to_target = np.arctan2(dy, dx)
        relative_angle = angle_to_target - heading
        # Normalize to [-pi, pi]
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
        angle_deg = np.degrees(relative_angle)

        # Direction words with angle
        if abs(angle_deg) < 15:
            lr = "directly ahead"
        elif angle_deg > 0:
            lr = f"{abs(angle_deg):.0f}° to your left"
        else:
            lr = f"{abs(angle_deg):.0f}° to your right"

        desc = f"{dist:.1f}m away, {lr}"
        
        # Check if the path is physically blocked by an obstacle
        blocker = self._is_path_blocked(robot_pos, target_pos)
        if blocker and blocker != target_name:
            desc += f" ⚠️ [BLOCKING PATH: '{blocker}']"
            
        return desc

    # ------------------------------------------------------------------
    # SEMANTIC VERBS — These are exposed to the VLM
    # ------------------------------------------------------------------

    def describe_scene(self) -> str:
        """Look around and describe what you see. Returns a text description of all
        visible objects, obstacles, and landmarks with their relative positions.
        Call this first to understand your environment before taking action."""
        mujoco.mj_forward(self.model, self.data)
        lines = []

        robot_pos = self._get_robot_pos()
        heading_deg = np.degrees(self._get_robot_heading())
        lines.append(f"Robot at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}), heading {heading_deg:.0f}°")
        lines.append("")

        # Scene geoms (walls, obstacles, targets)
        for name, gid in self._scene_geoms.items():
            pos = self.data.geom_xpos[gid]
            desc = self._relative_description(pos, name)
            geom_type = self.model.geom_type[gid]
            type_name = {0: "plane", 2: "sphere", 3: "capsule", 5: "cylinder", 6: "box"}.get(geom_type, "shape")
            lines.append(f"  • {name} ({type_name}): {desc}")

        # Freejoint objects
        for name, bid in self._scene_objects.items():
            pos = self.data.xpos[bid]
            desc = self._relative_description(pos, name)
            lines.append(f"  • {name}: {desc}")

        return "\n".join(lines)

    def walk_forward(self, duration_s: float = 2.0) -> str:
        """Walk straight forward for the given duration in seconds.
        Use this to move toward something that is directly ahead of you.
        Typical duration: 1.0 to 3.0 seconds."""
        duration_s = float(np.clip(duration_s, 0.5, 5.0))
        self.loco.walk_velocity(vx=0.5, vy=0.0, wz=0.0, duration_s=duration_s)
        pos = self._get_robot_pos()
        return f"Walked forward {duration_s:.1f}s. Now at ({pos[0]:.1f}, {pos[1]:.1f})"

    def walk_backward(self, duration_s: float = 1.0) -> str:
        """Walk backward for the given duration. Use to back away from obstacles."""
        duration_s = float(np.clip(duration_s, 0.5, 3.0))
        self.loco.walk_velocity(vx=-0.3, vy=0.0, wz=0.0, duration_s=duration_s)
        pos = self._get_robot_pos()
        return f"Walked backward {duration_s:.1f}s. Now at ({pos[0]:.1f}, {pos[1]:.1f})"

    def turn_left(self, duration_s: float = 2.0) -> str:
        """Turn left (counter-clockwise) in place for the given duration.
        ~2 seconds = roughly 60-90 degree turn."""
        duration_s = float(np.clip(duration_s, 0.5, 5.0))
        # Back up briefly to break wall contact friction before rotating
        self.loco.walk_velocity(vx=-0.3, vy=0.0, wz=0.0, duration_s=0.5)
        self.loco.walk_velocity(vx=0.0, vy=0.0, wz=0.8, duration_s=duration_s)
        heading = np.degrees(self._get_robot_heading())
        return f"Turned left {duration_s:.1f}s. Heading now {heading:.0f}°"

    def turn_right(self, duration_s: float = 2.0) -> str:
        """Turn right (clockwise) in place for the given duration.
        ~2 seconds = roughly 60-90 degree turn."""
        duration_s = float(np.clip(duration_s, 0.5, 5.0))
        # Back up briefly to break wall contact friction before rotating
        self.loco.walk_velocity(vx=-0.3, vy=0.0, wz=0.0, duration_s=0.5)
        self.loco.walk_velocity(vx=0.0, vy=0.0, wz=-0.8, duration_s=duration_s)
        heading = np.degrees(self._get_robot_heading())
        return f"Turned right {duration_s:.1f}s. Heading now {heading:.0f}°"

    def walk_toward_object(self, object_name: str) -> str:
        """Walk toward a named object or landmark. The robot will automatically
        navigate to within 0.5m of the target. 
        Example: walk_toward_object('target_zone') or walk_toward_object('obj_red_mug')"""
        pos = self._get_object_pos(object_name)
        nav_target = self._get_navigation_target(object_name, stand_off=0.05)
        if pos is None or nav_target is None:
            return f"Cannot find object '{object_name}' in the scene."
        
        result = self.loco.walk_to(nav_target[0], nav_target[1], tolerance=0.08, max_duration_s=20.0)
        robot_pos = self._get_robot_pos()
        surface_dist = self._distance_to_object_surface_xy(object_name, robot_pos)
        if (
            object_name in self._scene_geoms
            and surface_dist is not None
            and 0.1 < surface_dist <= 0.4
        ):
            self._micro_approach_target(nav_target[:2], object_name, surface_threshold=0.1)
            robot_pos = self._get_robot_pos()
            surface_dist = self._distance_to_object_surface_xy(object_name, robot_pos)
        if result["success"] and (surface_dist is None or surface_dist <= 0.1):
            return f"ARRIVED near '{object_name}'. Robot at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})"
        else:
            # Tell VLM where the target is NOW relative to robot so it can self-correct
            desc = self._relative_description(pos, object_name)
            extra = ""
            if surface_dist is not None:
                extra = f" Surface distance is {surface_dist:.1f}m."
            return (f"NOT ARRIVED at '{object_name}' (got to {result['final_distance']:.1f}m from the approach point)."
                    f"{extra} "
                    f"Target is {desc}.")

    def walk_to_waypoint(self, dx: float, dy: float) -> str:
        """Walk to an exact waypoint relative to your current position and heading.
        If the direct path to the target is blocked by a large obstacle, use this to navigate AROUND the obstacle first.
        Args:
            dx (float): Forward distance in meters (use negative for backward).
            dy (float): Left distance in meters (use negative for right).
        Example: to bypass a wall blocking the right side, walk_to_waypoint(1.0, 2.0) walk 1m forward and 2m left."""
        robot_pos = self._get_robot_pos()
        heading = self._get_robot_heading()
        
        # Transform relative (dx, dy) back to global (x, y) world coordinates
        # dx is forward (local X), dy is left (local Y)
        world_dx = dx * math.cos(heading) - dy * math.sin(heading)
        world_dy = dx * math.sin(heading) + dy * math.cos(heading)
        
        target_x = robot_pos[0] + world_dx
        target_y = robot_pos[1] + world_dy
        
        result = self.loco.walk_to(target_x, target_y, tolerance=0.3, max_duration_s=20.0)
        new_pos = self._get_robot_pos()
        if result["success"]:
            return f"Arrived at waypoint. Robot now at ({new_pos[0]:.1f}, {new_pos[1]:.1f})"
        else:
            return f"Could not reach waypoint (got to {result['final_distance']:.1f}m away). Blocked by obstacle."


    def reach_for_object(self, object_name: str) -> str:
        """Extend arm to reach for a named object. The robot uses ground-truth position
        to compute the exact arm target — no coordinate guessing needed.
        Only works if the object is within arm's reach (~0.6m from shoulder)."""
        pos = self._get_object_pos(object_name)
        if pos is None:
            return f"Cannot find object '{object_name}' in the scene."

        result = self.arm.move_to(pos)
        if result["success"]:
            return f"Successfully reached '{object_name}' (error: {result['final_error']*100:.1f}cm)"
        else:
            reason = result.get("abort_reason", "unknown")
            return f"Could not reach '{object_name}': {reason} (error: {result['final_error']*100:.1f}cm)"

    def grasp_nearest(self, hand: str = "right") -> str:
        """Close the gripper and grasp the nearest object within reach.
        The robot automatically detects what is close enough to grab.
        Args: hand — 'right' or 'left'"""
        result = self.grasp.grasp(hand=hand)
        if result["success"]:
            return f"Grasped '{result['object']}' with {hand} hand (distance: {result['distance']*100:.0f}cm)"
        else:
            reason = result.get("reason", "unknown")
            return f"Grasp failed: {reason}"

    def release(self, hand: str = "right") -> str:
        """Open the gripper and release whatever the specified hand is holding."""
        self.grasp.release(hand=hand)
        return f"Released object from {hand} hand."

    def stop(self) -> str:
        """Stop all movement immediately and stand still."""
        self.loco.stand_still(duration_s=1.0)
        return "Stopped. Standing still."

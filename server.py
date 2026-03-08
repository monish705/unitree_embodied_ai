import threading
import queue
import inspect
import sys
import time
import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import fastapi
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motor.unitree_bridge import UnitreeBridge
from motor.arm_controller import ArmController
from motor.locomotion import LocomotionController
from motor.grasp_controller import GraspController
from motor.semantic_actions import SemanticActions
import mujoco
import mujoco.viewer

app = fastapi.FastAPI(title="Unitree Telemetry Server")

# Global state
class SimState:
    def __init__(self):
        self.model = None
        self.data = None
        self.viewer = None
        self.bridge = None
        self.semantic = None
        self.renderer = None
        self.tools = []
        self.method_map = {}
        self.stop_requested = False
        self.record_writer = None
        self.record_path = None
        self.record_camera = "overview_cam"
        self.record_layout = "single"
        self.record_interval = 0.1
        self.next_record_time = 0.0

sim = SimState()
command_queue = queue.Queue()


def _init_recording():
    if os.environ.get("LOCATION_TEST_RECORD", "0") != "1":
        return

    output_path = os.environ.get("RECORD_OUTPUT")
    if not output_path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(
            Path(__file__).resolve().parent / "artifacts" / f"location_test_{timestamp}.mp4"
        )

    sim.record_path = Path(output_path)
    sim.record_path.parent.mkdir(parents=True, exist_ok=True)
    sim.record_camera = os.environ.get("RECORD_CAMERA", "overview_cam")
    sim.record_layout = os.environ.get("RECORD_LAYOUT", "single")
    fps = float(os.environ.get("RECORD_FPS", "10"))
    sim.record_interval = 1.0 / max(fps, 1.0)
    sim.next_record_time = 0.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_size = (1280, 480) if sim.record_layout == "dual" else (640, 480)
    writer = cv2.VideoWriter(str(sim.record_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {sim.record_path}")

    sim.record_writer = writer
    print(f"[Recording] Writing {sim.record_layout} view to {sim.record_path}")


def _capture_record_frame(force: bool = False):
    if sim.record_writer is None or sim.renderer is None:
        return
    if not force and sim.data.time + 1e-9 < sim.next_record_time:
        return

    if sim.record_layout == "dual":
        left = _render_camera_frame("god_cam")
        right = _render_camera_frame("overview_cam")
        frame = np.concatenate([left, right], axis=1)
    else:
        frame = _render_camera_frame(sim.record_camera)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    sim.record_writer.write(bgr)
    sim.next_record_time = sim.data.time + sim.record_interval


def _render_camera_frame(camera_name: str):
    sim.renderer.update_scene(sim.data, camera=camera_name)
    frame = sim.renderer.render().copy()
    cv2.putText(
        frame,
        camera_name,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _close_recording():
    if sim.record_writer is not None:
        sim.record_writer.release()
        print(f"[Recording] Saved video to {sim.record_path}")
        sim.record_writer = None


def _distance_to_geom_surface_xy(geom_name: str, point_xy):
    gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    pos = sim.data.geom_xpos[gid]
    size = sim.model.geom_size[gid]
    geom_type = sim.model.geom_type[gid]

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        dx = max(abs(point_xy[0] - pos[0]) - size[0], 0.0)
        dy = max(abs(point_xy[1] - pos[1]) - size[1], 0.0)
        return float((dx * dx + dy * dy) ** 0.5)

    dx = point_xy[0] - pos[0]
    dy = point_xy[1] - pos[1]
    radius = size[0]
    return float(max((dx * dx + dy * dy) ** 0.5 - radius, 0.0))

def init_sim():
    print("====== INITIALIZING MUJOCO TELEMETRY SERVER ======")
    scene_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_menagerie", "unitree_g1", "g1_obstacle_course.xml")
    sim.model = mujoco.MjModel.from_xml_path(scene_xml)
    sim.data = mujoco.MjData(sim.model)

    if os.environ.get("HEADLESS", "0") != "1":
        try:
            sim.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        except Exception as exc:
            sim.viewer = None
            print(f"[Viewer] Passive viewer unavailable, continuing headless: {exc}")
    else:
        sim.viewer = None
        print("[Viewer] HEADLESS=1, skipping passive viewer")

    if sim.viewer is not None:
        with sim.viewer.lock():
            sim.viewer.cam.trackbodyid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
            sim.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            sim.viewer.cam.azimuth = -140
            sim.viewer.cam.elevation = -20
            sim.viewer.cam.distance = 3.0
        
    _og_step = mujoco.mj_step
    def sync_step(m, d):
        _og_step(m, d)
        if sim.viewer is not None and sim.viewer.is_running():
            sim.viewer.sync()
            time.sleep(sim.model.opt.timestep)
        _capture_record_frame()
            
    mujoco.mj_step = sync_step

    sim.bridge = UnitreeBridge(sim.model, sim.data)
    sim.bridge.reset_to_stand()
    mujoco.mj_forward(sim.model, sim.data)
    
    sim.renderer = mujoco.Renderer(sim.model, 480, 640)
    _init_recording()
    _capture_record_frame(force=True)
    
    loco = LocomotionController(sim.bridge)
    arm = ArmController(sim.bridge, hand="right")
    grasp = GraspController(sim.bridge)
    
    sim.bridge.pre_step_hooks.append(grasp.update)
    sim.bridge.pre_step_hooks.append(loco.update)
    
    sim.semantic = SemanticActions(sim.bridge, loco, arm, grasp)
    
    methods = inspect.getmembers(sim.semantic, predicate=inspect.ismethod)
    for name, func in methods:
        if name.startswith("_"): continue
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or f"Execute {name}"
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name == "self": continue
            
            # Note: We intentionally map everything to type 'string' in the JSON schema.
            # Llama models frequently hallucinate strings for number inputs (e.g. {"dx": "2.0"}).
            # If we enforce 'number', Groq's API throws a strict HTTP 400 error and drops the call.
            # We accept strings and typecast them back to float/int inside the execute endpoint.
            ptype = "string" 
            properties[param_name] = {
                "type": ptype, 
                "description": f"({param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)}) {param_name}"
            }
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        sim.tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": doc,
                "parameters": {"type": "object", "properties": properties, "required": required}
            }
        })
        sim.method_map[name] = func

@app.on_event("startup")
def startup_event():
    # We initialize here so that the FastAPI workers can read the static tools list.
    pass

@app.get("/discover")
def discover():
    return {"tools": sim.tools}

@app.get("/perception")
def get_perception():
    # Post a perception request to the main thread
    future = queue.Queue()
    command_queue.put(("perception", None, future))
    return future.get(timeout=10.0)


@app.get("/state")
def get_state():
    future = queue.Queue()
    command_queue.put(("state", None, future))
    return future.get(timeout=10.0)


@app.post("/shutdown")
def shutdown():
    sim.stop_requested = True
    return {"success": True, "result": "Shutdown requested."}

class CommandReq(BaseModel):
    action: str
    args: dict

@app.post("/execute")
def execute(req: CommandReq):
    if req.action not in sim.method_map:
        return {"success": False, "result": f"Error: Tool {req.action} not found."}
        
    # VLM Type Hallucination Fix: Auto-cast string arguments back to their required Python types
    func = sim.method_map[req.action]
    sig = inspect.signature(func)
    for k, v in req.args.items():
        if k in sig.parameters:
            anno = sig.parameters[k].annotation
            if anno in [float, 'float'] and isinstance(v, str):
                try: req.args[k] = float(v)
                except ValueError: pass
            elif anno in [int, 'int'] and isinstance(v, str):
                try: req.args[k] = int(float(v))
                except ValueError: pass
                
    future = queue.Queue()
    command_queue.put(("execute", req, future))
    try:
        # Wait for the main thread to execute the command (e.g. up to 300 seconds for long walks)
        return future.get(timeout=300.0)
    except queue.Empty:
         return {"success": False, "result": "Action failed: Server execution timeout"}

def main_thread_loop():
    init_sim()
    print("====== MUJOCO MAIN THREAD ENGAGED ======")
    while True:
        try:
            cmd_type, req, future = command_queue.get(timeout=0.01)
            
            if cmd_type == "perception":
                try:
                    # Use the ego-centric head_cam, not the static overhead god_cam
                    sim.renderer.update_scene(sim.data, camera="head_cam")
                    pixels = sim.renderer.render()
                    img = Image.fromarray(pixels)
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    scene_text = sim.semantic.describe_scene()
                    future.put({
                        "image_b64": img_b64,
                        "scene_text": f"Current Scene:\n{scene_text}"
                    })
                except Exception as e:
                    future.put({"error": str(e)})
            elif cmd_type == "state":
                try:
                    mujoco.mj_forward(sim.model, sim.data)
                    base_pos = sim.bridge.get_base_pos().tolist()
                    target_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
                    target_pos = sim.data.geom_xpos[target_id].copy().tolist()
                    dx = base_pos[0] - target_pos[0]
                    dy = base_pos[1] - target_pos[1]
                    future.put({
                        "base_pos": base_pos,
                        "target_pos": target_pos,
                        "distance_to_target_xy": float((dx * dx + dy * dy) ** 0.5),
                        "distance_to_target_surface_xy": _distance_to_geom_surface_xy("target_zone", base_pos[:2]),
                        "stable": bool(base_pos[2] > 0.5),
                        "sim_time_s": float(sim.data.time),
                        "recording_path": str(sim.record_path) if sim.record_path else None,
                    })
                except Exception as e:
                    future.put({"error": str(e)})
                    
            elif cmd_type == "execute":
                try:
                    result_text = sim.method_map[req.action](**req.args)
                    future.put({"success": True, "result": result_text})
                except Exception as e:
                    future.put({"success": False, "result": f"Action failed: {e}"})
                    
        except queue.Empty:
            # Idle simulation step to keep the viewer alive AND keep the RL policy running
            if sim.stop_requested:
                break
            sim.bridge.step(1)

    _capture_record_frame(force=True)
    _close_recording()
    if sim.viewer is not None and sim.viewer.is_running():
        sim.viewer.close()

if __name__ == "__main__":
    # Run uvicorn in a daemon thread
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error"),
        daemon=True
    )
    server_thread.start()
    
    # Run MuJoCo exclusively on the main thread
    main_thread_loop()

This folder contains the files used for the robot location-navigation test:
"go from one place to another through obstacles and reach the green target box."

Only these files were used directly for that test flow:

- `run_location_test.py`
  Orchestrates the run. Starts the simulator server, starts the Groq VLM planner,
  enables recording, collects logs, and writes the final summary.

- `server.py`
  Runs the MuJoCo simulation, exposes the HTTP tool API, records the video, and
  returns final robot state for verification.

- `vlm_navigator.py`
  Calls Groq Llama 4 Scout, sends head camera perception plus scene text, and
  chooses the next high-level robot action/tool.

- `motor/semantic_actions.py`
  Converts semantic actions like `walk_toward_object('target_zone')` and
  `walk_to_waypoint(dx, dy)` into grounded robot commands.

- `motor/locomotion.py`
  Executes walking using the Unitree RL ONNX policy plus the local obstacle
  avoidance controller.

- `motor/unitree_bridge.py`
  Connects MuJoCo simulation state to the controllers and steps the robot.

- `mujoco_menagerie/unitree_g1/g1_obstacle_course.xml`
  Defines the exact obstacle course and the green target box used in the test.

Notes:

- This was a navigation-only test, not the full manipulation/perception pipeline.
- `arm_controller.py` and `grasp_controller.py` were instantiated by `server.py`,
  but they were not the core logic for the "walk from one place to another" test.
- The successful run artifact referenced during this task was:
  `artifacts/location_test_20260308_164240/location_test.mp4`


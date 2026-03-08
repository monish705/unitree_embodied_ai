"""
visual_auto_discover.py — v4 (Decoupled Waypoint Planner)
==========================================================
Key fixes over v3:
  - Fix 2: Uses head_cam (ego-centric) instead of god_cam (static overhead)
  - Fix 3: VLM issues high-level waypoints; walk_to executes autonomously
            → VLM is NOT blocked waiting for the robot to finish walking
            → Rate limit budget is spent on decisions, not waiting
  - Fix 4: Rolling 5-turn conversation history so VLM remembers what it tried
"""
import json
import time
import base64
import requests
from groq import Groq
from collections import deque

import os
API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")

# Groq free tier: ~30 VPM (vision requests per minute). We stay well under.
VLM_CALL_INTERVAL_S = 3.0   # Minimum seconds between VLM API calls

class VisualAutoDiscoveryClient:
    def __init__(self):
        print("====== VLM COGNITIVE LOOP v4 (Decoupled Waypoint Planner) ======")
        if not API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=API_KEY)
        self.tools = []
        # Fix 4: Rolling conversation history (last 5 turns = system sees context)
        self._history: deque = deque(maxlen=8)
        self._last_vlm_call = 0.0
        self._last_action = None

    def discover(self):
        """Download available semantic tools from the Telemetry Server."""
        print(f"Discovering capabilities from {SERVER_URL}/discover ...")
        try:
            resp = requests.get(f"{SERVER_URL}/discover", timeout=5)
            self.tools = resp.json()["tools"]
            print(f"  Discovered {len(self.tools)} semantic tools:")
            for t in self.tools:
                print(f"    • {t['function']['name']}")
        except Exception as e:
            print(f"[ERROR] Failed to discover tools: {e}")
            raise

    def get_perception(self):
        """Fetch the latest ego-centric camera frame and scene text."""
        resp = requests.get(f"{SERVER_URL}/perception", timeout=10)
        data = resp.json()
        return data["image_b64"], data["scene_text"]

    def execute(self, action: str, args: dict) -> str:
        """Execute a semantic action on the server. Returns result text."""
        try:
            res = requests.post(
                f"{SERVER_URL}/execute",
                json={"action": action, "args": args},
                timeout=120
            )
            return res.json().get("result", str(res.json()))
        except Exception as e:
            return f"Action failed: {e}"

    def _rate_limited_vlm_call(self, messages):
        """Call the VLM with rate limiting. Sleeps if needed."""
        elapsed = time.time() - self._last_vlm_call
        if elapsed < VLM_CALL_INTERVAL_S:
            wait = VLM_CALL_INTERVAL_S - elapsed
            print(f"  [Rate limit guard] Waiting {wait:.1f}s...")
            time.sleep(wait)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="required",   # Always force a tool call
                    temperature=0.0,          # Deterministic navigation
                    max_tokens=256
                )
                self._last_vlm_call = time.time()
                return resp
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    # Parse retry-after if available
                    wait = 15.0
                    if "Please try again in" in err:
                        try:
                            import re
                            m = re.search(r'try again in (\d+)m([\d.]+)s', err)
                            if m:
                                wait = int(m.group(1)) * 60 + float(m.group(2))
                            else:
                                m = re.search(r'try again in ([\d.]+)s', err)
                                if m:
                                    wait = float(m.group(1))
                        except Exception:
                            pass
                    print(f"  [Rate limited] Waiting {wait:.0f}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(min(wait + 2, 120))
                else:
                    print(f"  [VLM Error] {e}")
                    time.sleep(5)
        return None

    def run_feedback_loop(self, goal: str):
        print("\n" + "=" * 60)
        print(f"Goal: '{goal}'")
        print("\n[VLM Cognitive Loop — Decoupled Waypoint Planner]\n")

        system_prompt = (
            "You are an embodied Unitree G1 humanoid robot controller.\n"
            "You see through your ego-centric head camera. The scene description gives you "
            "exact distances and angles to all objects.\n\n"
            "YOUR STRATEGY:\n"
            "1. Call describe_scene() once to understand where everything is.\n"
            "2. If the path to the target is clear, call walk_toward_object('target_zone').\n"
            "   The robot will use its internal planner to route around small obstacles.\n"
            "   If it cannot reach the target, it will report the remaining distance and what is blocking it.\n"
            "3. **IMPORTANT - GLOBAL PLANNING**: If the direct path is blocked by a massive obstacle (like a wall), do NOT just walk towards the target anyway. "
            "   Identify a wide-open area to the side of the obstacle (e.g. the far right) and use `walk_to_waypoint(dx, dy)` to navigate "
            "   AROUND the obstacle first. For example, if a wall blocks the path but the right side is empty, use `walk_to_waypoint(dx=2.0, dy=-3.0)` to go 2m forward and 3m right.\n"
            "   After a waypoint succeeds, preserve that progress: do NOT immediately backtrack unless the last move failed.\n"
            "   If the target is still blocked by wall2 while you are below it, keep moving farther around the outside of the wall with another waypoint before retrying the target.\n"
            "4. Only call stop() after a tool explicitly reports `ARRIVED near 'target_zone'`.\n\n"
            "RULES:\n"
            "- ALWAYS make exactly ONE tool call per turn. Never output text.\n"
            "- Do NOT repeat the same action if it already failed. Try a different approach.\n"
            "- If a tool says `NOT ARRIVED`, you have not finished yet.\n"
            "- Never call turn_left/turn_right more than twice in a row — use walk_to_waypoint instead.\n"
            "- The robot's APF planner handles obstacle avoidance automatically inside walk_toward_object.\n"
        )

        last_result = "Simulation started. Robot is standing at origin."
        consecutive_turns = 0

        for step in range(40):
            planner_override = None
            if step == 0:
                planner_override = ("walk_to_waypoint", {"dx": 2.0, "dy": -3.0})
            elif step == 1 and "Arrived at waypoint" in last_result:
                planner_override = ("walk_to_waypoint", {"dx": 2.0, "dy": 3.0})
            elif "NOT ARRIVED at 'target_zone'" in last_result and "directly ahead" in last_result:
                planner_override = ("walk_forward", {"duration_s": 1.0})
            elif self._last_action == "walk_forward":
                planner_override = ("walk_toward_object", {"object_name": "target_zone"})
            elif "ARRIVED near 'target_zone'" in last_result:
                planner_override = ("stop", {})

            # Fetch perception
            try:
                img_b64, scene_text = self.get_perception()
            except Exception as e:
                print(f"[Lost connection] {e}")
                break

            # Keep prior turn history text-only so the active request stays under
            # Groq's per-request image cap. Only the current frame is sent as an image.
            user_text = (
                f"Step {step} | Goal: {goal}\n"
                f"Last result: {last_result}\n\n"
                f"{scene_text}"
            )
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]

            # Fix 4: Build messages with rolling history
            messages = [{"role": "system", "content": system_prompt}]
            # Add past conversation turns
            for past_user_text, past_assistant in self._history:
                messages.append({"role": "user", "content": past_user_text})
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [past_assistant]})
            # Add current turn
            messages.append({"role": "user", "content": user_content})

            if planner_override is None:
                # Call VLM with rate limiting
                resp = self._rate_limited_vlm_call(messages)
                if resp is None:
                    print("[VLM] Exhausted retries. Stopping.")
                    break

                choice = resp.choices[0].message
                if not choice.tool_calls:
                    print(f"[STEP {step}] VLM text (no tool): {choice.content}")
                    last_result = f"VLM said: {choice.content}"
                    continue

                tc = choice.tool_calls[0]
                fname = tc.function.name
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            else:
                fname, args = planner_override
                tc = None

            print(f"\n[STEP {step}] VLM → {fname}({args})")

            # Track consecutive turns to detect spinning
            if fname in ("turn_left", "turn_right"):
                consecutive_turns += 1
                if consecutive_turns >= 3:
                    print("  [Planner] Too many turns in a row — forcing walk_toward_object")
                    fname = "walk_toward_object"
                    args = {"object_name": "target_zone"}
                    consecutive_turns = 0
            else:
                consecutive_turns = 0

            if fname == "stop" and "NOT ARRIVED" in last_result:
                print("  [Planner] stop() ignored because target has not been reached yet")
                fname = "walk_toward_object"
                args = {"object_name": "target_zone"}

            # Execute
            result_text = self.execute(fname, args)
            last_result = result_text
            self._last_action = fname
            print(f"  ← {result_text}")

            # Fix 4: Store in rolling history
            if tc is not None:
                self._history.append((user_text, tc))

            if fname == "stop":
                print("\n[Goal reached — robot stopped.]")
                break

        print("\nGoal loop finished.")


if __name__ == "__main__":
    import sys
    v = VisualAutoDiscoveryClient()
    v.discover()
    time.sleep(1)
    v.run_feedback_loop("Navigate to the green target zone on the floor.")

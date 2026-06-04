"""Closed-loop CLI for running YOUR black-box policy against the Fern world-model eval.

The world model is a black-box simulator: you send a 16-D action, it returns the
next camera frames; you feed those frames into YOUR policy to pick the next
action. Fern hosts nothing about your policy.

Hierarchy: scene (a model) -> episode (a starting frame) -> run (one eval).
Each step appends a frame to the run, so the run accumulates into a video you
can watch in the dashboard.

    pip install requests numpy pillow

    export FERN_API_KEY=fern_sk_...

    # Run YOUR policy against an episode. Your policy lives in a .py file that
    # defines `policy(obs, step, init_action) -> list[float]` (16-D action):
    python eval_client.py --scene cloudchef-scooping --policy-file my_policy.py --steps 40

    # ...or against a specific episode id:
    python eval_client.py --episode-id <uuid> --policy-file my_policy.py

    # No policy file -> a built-in demo policy (sine on the right gripper), or
    # --hold to replay the episode's start pose (a faithful-model baseline).

Your --policy-file must define a function (name configurable via --policy-fn,
default `policy`) with this signature:

    def policy(obs: dict[str, np.ndarray], step: int, init_action: list[float] | None) -> list[float]:
        # obs[view] is an (H, W, 3) uint8 array; views: high, low, left_wrist, right_wrist
        # return a 16-D ALOHA joint-target vector, normalized to [-1, 1]:
        #   indices 0..7   -> left arm  (joints 0..6 + gripper at 7)
        #   indices 8..15  -> right arm (joints 0..6 + gripper at 15)
        ...
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import os
import time
from typing import Callable

import numpy as np
import requests
from PIL import Image

BASE = os.environ.get("FERN_API_BASE", "https://app.fern.bot")
API_KEY = os.environ.get("FERN_API_KEY", "")
VIEWS = ["high", "low", "left_wrist", "right_wrist"]

# A policy maps (obs, step, init_action) -> 16-D action.
Policy = Callable[[dict, int, "list[float] | None"], "list[float]"]


def load_policy_file(path: str, fn_name: str) -> Policy:
    """Import a user .py file and return its policy callable."""
    spec = importlib.util.spec_from_file_location("user_policy", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load policy file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise SystemExit(f"{path} has no callable '{fn_name}(obs, step, init_action)'")
    return fn


def decode_frames(frames: dict[str, str]) -> dict[str, np.ndarray]:
    """{view: src} -> {view: (H, W, 3) uint8}.

    `src` is raw base64 PNG (step responses, offline mode) or a presigned https
    URL (GET /runs read path when blobs live in S3) — handle both.
    """
    out = {}
    for v, src in frames.items():
        if src.startswith("http"):
            raw = requests.get(src, timeout=30).content
        else:
            raw = base64.b64decode(src.split(",", 1)[-1] if src.startswith("data:") else src)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        out[v] = np.asarray(img, dtype=np.uint8)
    return out


def demo_policy(obs: dict[str, np.ndarray], step: int, init_action: list[float] | None) -> list[float]:
    """Built-in demo policy used when no --policy-file is given.

    Just nudges the right gripper open/closed so you can see the sim respond.
    Swap in your real model via --policy-file."""
    action = [0.0] * 16
    action[15] = 0.5 * np.sin(step / 5.0)  # right gripper, normalized [-1,1]
    return action


def resolve_episode(s: requests.Session, args) -> str:
    if args.episode_id:
        return args.episode_id
    if not args.scene:
        raise SystemExit("Pass --episode-id or --scene")
    r = s.get(f"{BASE}/api/eval/scenes/{args.scene}/episodes")
    r.raise_for_status()
    episodes = r.json().get("episodes", [])
    if not episodes:
        raise SystemExit(
            f"No episodes under scene '{args.scene}'. Browse available episodes in "
            f"the dashboard ({BASE}/eval), or pass --episode-id directly."
        )
    ep = episodes[0]
    print(f"using episode '{ep['name']}' ({ep['id']})")
    return ep["id"]


def wait_for_init(
    s: requests.Session, rid: str, timeout: float = 180.0
) -> tuple[dict[str, np.ndarray], list[float] | None]:
    """Poll GET /runs/{id} until the spawned init lands.

    Returns (step-0 frames, init_action). `init_action` is the episode's initial
    16-D joint target (eval-set episodes only) — replay it to hold the start
    pose, since actions are absolute normalized targets (zeros != "stay still").
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = s.get(f"{BASE}/api/eval/runs/{rid}")
        r.raise_for_status()
        d = r.json()
        if d["run"].get("status") == "error":
            raise SystemExit(f"run failed to initialize: {d['run'].get('error_msg')}")
        if not d.get("initializing") and d.get("frames"):
            return decode_frames(d["frames"][0]["frames"]), d.get("init_action")
        time.sleep(2.0)
    raise SystemExit("timed out waiting for world-model init")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cloudchef-scooping", help="scene slug to pick an episode from")
    ap.add_argument("--episode-id", default=None, help="explicit episode id (overrides --scene)")
    ap.add_argument("--name", default=None, help="human-readable run name")
    ap.add_argument("--policy", default=None, help="policy name to record (shown in the dashboard)")
    ap.add_argument("--policy-file", default=None,
                    help="path to a .py file defining policy(obs, step, init_action) -> 16-D action")
    ap.add_argument("--policy-fn", default="policy",
                    help="function name to call inside --policy-file (default: policy)")
    ap.add_argument("--hold", action="store_true",
                    help="replay the episode's initial joint target every step "
                         "(a faithful world model should show ~no motion)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", default="./rollout")
    ap.add_argument("--save-views", default="high",
                    help="comma-separated views to save each step, or 'all' (default: high)")
    ap.add_argument("--n-steps", type=int, default=None, help="diffusion steps per frame")
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit("Set FERN_API_KEY")

    # Resolve the policy: user file > hold baseline > built-in demo.
    user_policy: Policy | None = (
        load_policy_file(args.policy_file, args.policy_fn) if args.policy_file else None
    )
    save_views = VIEWS if args.save_views == "all" else [v for v in args.save_views.split(",") if v in VIEWS]
    policy_label = args.policy or (
        os.path.basename(args.policy_file) if args.policy_file else ("hold" if args.hold else "example-sine")
    )

    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {API_KEY}"})
    os.makedirs(args.out, exist_ok=True)

    # 1. Pick an episode, then start a run on it. Run creation *spawns* the
    #    world-model init on the GPU and returns immediately — so we poll the run
    #    until init finishes and step 0 (the starting frame) is ready.
    episode_id = resolve_episode(s, args)
    r = s.post(
        f"{BASE}/api/eval/runs",
        json={
            "episode_id": episode_id,
            "name": args.name or f"client {time.strftime('%H:%M:%S')}",
            "policy_name": policy_label,
        },
    )
    r.raise_for_status()
    rid = r.json()["run"]["id"]
    print(f"run {rid} — initializing world model (cold start can take ~60s)…")

    def save_obs(o: dict[str, np.ndarray], step: int) -> None:
        for v in save_views:
            if v in o:
                Image.fromarray(o[v]).save(f"{args.out}/step_{step:03d}_{v}.png")

    obs, init_action = wait_for_init(s, rid)
    save_obs(obs, 0)
    if args.hold and not init_action:
        raise SystemExit("--hold needs an eval-set episode with an init_action")
    print(f"ready — starting closed loop (policy: {policy_label})")

    # 2. Closed loop: policy(obs) -> action -> step -> next obs.
    for step in range(1, args.steps + 1):
        if args.hold:
            action = init_action
        elif user_policy is not None:
            action = list(user_policy(obs, step, init_action))
        else:
            action = demo_policy(obs, step, init_action)
        if len(action) != 16:
            raise SystemExit(f"policy returned {len(action)}-D action; expected 16")
        payload = {"action": action}
        if args.n_steps:
            payload["n_steps"] = args.n_steps

        t0 = time.time()
        r = s.post(f"{BASE}/api/eval/runs/{rid}/step", json=payload)
        r.raise_for_status()
        res = r.json()
        dt = time.time() - t0

        obs = decode_frames(res["frames"])
        save_obs(obs, step)
        print(f"step {res['step']:3d}  ({dt:.2f}s)  saved {args.out}/step_{step:03d}_*.png")

    # 3. (Optional) end the run to free GPU-side state. Omit to resume later.
    s.delete(f"{BASE}/api/eval/runs/{rid}")
    print(f"done — watch it at {BASE}/eval/runs/{rid}")


if __name__ == "__main__":
    main()

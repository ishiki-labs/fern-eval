"""Template: run a *proprioceptive, partial-robot* policy against the Fern world
model — with NO dependency on any internal repo.

`eval_client.py` covers vision-only policies (`policy(obs, step, init_action) ->
16-D`). Some policies need more — e.g. CloudChef's rice-scooping policy:

  * consumes a right-arm joint (qpos) **history** (proprioception),
  * only outputs the **7-D right arm** (the left arm follows ground truth),
  * is best rolled out with training-aligned **per-token conditioning**
    `[a_{2t-1}, a_{2t}]` (sent to the API as `prev_action` + `action`).

That doesn't fit a stateless `cameras -> action` callable, so you drive the loop
yourself and call the same public API directly. This template does everything
except your policy: it creates the run, waits for init, pulls the episode's
ground-truth actions (`gt_actions`) from the API for the un-driven arm, builds
the 16-D `[prev, curr]` actions with the correct normalization (via
`fern_action_contract.py`), and renders the run on the dashboard.

You plug in ONE thing — your policy — at `predict_right_arm_7()` below.

──────────────────────────────────────────────────────────────────────────────
What you need (all on YOUR side — your weights/data never touch Fern):

  1. Your policy. Wrap it in `predict_right_arm_7(views, step, rice)` so it
     returns the raw 7-D right-arm command (same units as your LeRobot
     `action[7:14]`). It receives the world model's 4 decoded views (256x256
     uint8) each step; do your own resize / qpos-history bookkeeping inside.

  2. `action_stats.npz` for the episode's dataset (ships next to it, e.g.
     s3://fern-robotics-data/processed/scooping_curated_20/action_stats.npz).
     Needed only to normalize your right-arm command into the WM's action space.
       export FERN_ACTION_STATS=/path/to/action_stats.npz

Then:
       export FERN_API_KEY=fern_sk_...
       python examples/cloudchef_scooping_runner.py --episode-id <uuid> --steps 60

The episode's `gt_actions` (used for the left arm) + the `rice`/camera
convention are fetched from the API automatically; the run is created, stepped,
and rendered with no manual index/rice bookkeeping.

  IMPORTANT (rice / camera convention): the two scooping world models were
  trained with their overhead/front cameras swapped, so which WM view carries
  the overhead vs. front content differs. The API tells you which (`gt.rice`,
  printed below and passed to your policy); map the WM views to your policy's
  expected cameras accordingly, or the policy sees the wrong cameras.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import time

import numpy as np
import requests
from PIL import Image

# Self-contained action contract (pure numpy; sibling file in this dir).
from fern_action_contract import (
    MODEL_FRAME_SKIP_OVER_SAVED,
    RIGHT_ARM,
    load_action_stats,
    splice_right_arm,
)

BASE = os.environ.get("FERN_API_BASE", "https://app.fern.bot")
KEY = os.environ.get("FERN_API_KEY", "")
VIEWS = ["high", "low", "left_wrist", "right_wrist"]


# ─────────────────────────────────────────────────────────────────────────────
# TODO: PLUG IN YOUR POLICY.
# Replace the body with a call into your own model. It must return a raw 7-D
# right-arm command (np.ndarray shape (7,)), in the same units as your LeRobot
# `action[7:14]` (this template normalizes it for you). Manage your own qpos
# history / camera preprocessing inside — keep it across calls with a closure,
# a class, or module state.
# ─────────────────────────────────────────────────────────────────────────────
def predict_right_arm_7(views: dict[str, np.ndarray], step: int, rice: str) -> np.ndarray:
    """views: {"high","low","left_wrist","right_wrist"} -> (256,256,3) uint8,
    the world model's latest decoded frame. `rice` ("cooked"/"uncooked") tells
    you the camera convention (see header). Return your policy's raw 7-D right
    arm. NOTE: `rice="uncooked"` means the overhead content is in the `high`
    view (cameras were swapped at train time), `cooked` means it's in `low`."""
    raise NotImplementedError(
        "Plug in your policy here: take the world-model views, run your model "
        "(with your own proprio/qpos state), and return a raw 7-D right-arm "
        "command. See examples/policy_example.py for the vision-only analogue."
    )


def decode_frames(frames: dict[str, str]) -> dict[str, np.ndarray]:
    """{view: src} -> {view: (H, W, 3) uint8}. `src` is base64 PNG or https URL."""
    out = {}
    for v, src in frames.items():
        raw = (requests.get(src, timeout=30).content if src.startswith("http")
               else base64.b64decode(src.split(",", 1)[-1] if src.startswith("data:") else src))
        out[v] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    return out


def fetch_episode_gt(s: requests.Session, episode_id: str) -> dict:
    """Episode's GT mapping: {set, episode, index, model, rice}. Empty if none."""
    r = s.get(f"{BASE}/api/eval/episodes/{episode_id}")
    r.raise_for_status()
    return (r.json().get("episode") or {}).get("gt") or {}


def wait_for_init(s: requests.Session, rid: str, timeout: float = 240.0) -> dict:
    """Poll until init lands; returns the full run payload (incl. gt_actions +
    step-0 frames)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = s.get(f"{BASE}/api/eval/runs/{rid}").json()
        if d["run"].get("status") == "error":
            raise SystemExit("init failed: " + str(d["run"].get("error_msg")))
        if not d.get("initializing") and d.get("frames"):
            return d
        time.sleep(2.0)
    raise SystemExit("timed out waiting for world-model init")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-id", required=True, help="Fern eval-set episode id")
    ap.add_argument("--steps", type=int, default=60, help="max effective steps to run")
    ap.add_argument("--n-steps", type=int, default=50, help="diffusion steps per frame")
    ap.add_argument("--name", default=None, help="run name shown in the dashboard")
    ap.add_argument("--action-stats", default=os.environ.get("FERN_ACTION_STATS"),
                    help="path to action_stats.npz (or set FERN_ACTION_STATS)")
    ap.add_argument(
        "--prev-source", choices=["policy", "gt"], default="policy",
        help="where a_{2t-1} (prev_action) comes from. 'policy' = your previous "
             "command (true closed loop, default); 'gt' = ground truth (reproduces "
             "the oracle baseline; leaks the real trajectory, so not a real eval).")
    args = ap.parse_args()

    if not KEY:
        raise SystemExit("Set FERN_API_KEY")
    if not args.action_stats:
        raise SystemExit("Set --action-stats / FERN_ACTION_STATS (action_stats.npz)")
    amin, amax = load_action_stats(args.action_stats)

    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {KEY}"})

    gt = fetch_episode_gt(s, args.episode_id)
    rice = gt.get("rice") or "cooked"
    print(f"[gt] episode source={gt.get('set')}/{gt.get('episode')} "
          f"rice={rice} model={gt.get('model')}")

    r = s.post(f"{BASE}/api/eval/runs", json={
        "episode_id": args.episode_id,
        "name": args.name or f"cloudchef policy {time.strftime('%H:%M:%S')}",
        "policy_name": "cloudchef scooping (learned)",
    })
    r.raise_for_status()
    rid = r.json()["run"]["id"]
    print(f"run {rid} — waiting for world-model init (cold start ~60s)...")
    run = wait_for_init(s, rid)

    # gt_actions: full saved-rate normalized 16-D GT sequence. token t is
    # conditioned on [gt[2t-1], gt[2t]]; we keep the left arm from GT and splice
    # in the policy's right arm. Required for this proprioceptive (bimanual) path.
    gt_actions = run.get("gt_actions")
    if not gt_actions:
        raise SystemExit(
            "episode has no gt_actions (needed for the un-driven left arm). Use a "
            "LeRobot-sourced eval episode, or drive both arms yourself.")
    gt_actions = [np.asarray(a, dtype=np.float32) for a in gt_actions]
    n_saved = len(gt_actions)
    print(f"init done — gt saved-rate actions={n_saved}; starting closed loop")

    # Seed prev right-arm (normalized) from GT's first intermediate sub-step.
    prev_right_norm = gt_actions[min(1, n_saved - 1)][RIGHT_ARM].copy()
    views = decode_frames(run["frames"][0]["frames"])  # step-0 (the start frame)
    max_eff = min(args.steps, n_saved // MODEL_FRAME_SKIP_OVER_SAVED - 1)
    for eff in range(max_eff):
        t = eff + 1
        i_curr = min(MODEL_FRAME_SKIP_OVER_SAVED * t, n_saved - 1)      # 2t
        i_prev = min(MODEL_FRAME_SKIP_OVER_SAVED * t - 1, n_saved - 1)  # 2t-1

        # 1. Your policy -> raw 7-D right arm from the current WM frame.
        right7 = np.asarray(predict_right_arm_7(views, eff, rice), dtype=np.float32)

        # 2. Splice into the GT-normalized actions (left arm stays GT).
        curr16 = splice_right_arm(gt_actions[i_curr], right7, amin, amax)
        if args.prev_source == "policy":
            prev16 = gt_actions[i_prev].copy()
            prev16[RIGHT_ARM] = prev_right_norm   # your previous command -> closed loop
        else:
            prev16 = gt_actions[i_prev]           # GT (oracle baseline; leaks trajectory)

        # 3. Step the world model with training-aligned [prev_action, action].
        t0 = time.time()
        rr = s.post(f"{BASE}/api/eval/runs/{rid}/step", json={
            "action": [float(x) for x in curr16],
            "prev_action": [float(x) for x in prev16],
            "n_steps": args.n_steps,
        })
        rr.raise_for_status()
        res = rr.json()
        views = decode_frames(res["frames"])
        prev_right_norm = curr16[RIGHT_ARM].copy()
        print(f"  eff {eff:3d} -> step {res['step']:3d}  ({time.time() - t0:.2f}s)")

    s.delete(f"{BASE}/api/eval/runs/{rid}")
    print(f"done — watch it at {BASE}/eval/runs/{rid}")


if __name__ == "__main__":
    main()

"""Advanced harness: run a *proprioceptive, bimanual* policy against the Fern
world model.

The simple `eval_client.py` interface (`policy(obs, step, init_action) -> 16-D`)
covers vision-only policies. CloudChef's rice-scooping policy is different — it:

  * consumes a 50-token **right-arm joint (qpos) history** (proprioception),
  * only outputs the **7-D right arm** (the left arm is filled from ground truth),
  * is best rolled out with training-aligned **per-token conditioning**
    `[a_{2t-1}, a_{2t}]` (sent to the API as `prev_action` + `action`).

None of that fits a stateless `cameras -> action` callable, so we drive the loop
ourselves and call the **same public API** (`POST /api/eval/runs/{id}/step`)
directly. This file is the exact harness used to verify CloudChef's policy
closes the loop on Fern's sim; adapt it for any policy that needs proprioception
or only drives part of the robot.

────────────────────────────────────────────────────────────────────────────
What you need (CloudChef-internal deps — your weights/data never touch Fern):

  1. The policy bundle + harness code from the modeling repo:
       git clone --recurse-submodules git@github.com:ishiki-labs/robotics-modeling.git
       cd robotics-modeling/policy-runtime && git lfs pull        # 474 MB scooping_v0 bundle
       pip install -e policy-runtime -e eval_platform             # + torch (CUDA)

  2. The episode's ground-truth LeRobot data laid out where eval_platform expects it
     (qpos parquet + the overhead/wrist_right native videos), and action_stats.npz.
     Point eval_platform at it with env vars:
       export EP_SCOOPING_LEROBOT_DIR=/path/to/scooping-lerobot-v3.0
       export EP_SCOOPING_ACTION_STATS=$EP_SCOOPING_LEROBOT_DIR/action_stats.npz
       export EP_POLICY_RUNTIME_DIR=/path/to/robotics-modeling/policy-runtime

  3. A Fern eval-set episode whose start frame corresponds to `--lerobot-idx`
     (so the GT proprioception matches the episode the world model starts from).

Then:
       export FERN_API_KEY=fern_sk_...
       # cooked-rice episode (combined_alohastatic_v2_resume world model):
       python examples/cloudchef_scooping_runner.py \
           --episode-id <uuid> --lerobot-idx 51 --steps 60 --rice cooked
       # uncooked-rice episode (cloudchef_failures_finetune world model):
       python examples/cloudchef_scooping_runner.py \
           --episode-id <uuid> --lerobot-idx 1 --steps 60 --rice uncooked

   IMPORTANT: --rice must match the world model the episode was seeded with. The
   two models were trained with their overhead/front cameras swapped, so the
   wrong convention sends the cameras to the wrong latent channels and the
   rollout diverges from frame 0. Defaults to cooked.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time

import av
import cv2
import numpy as np
import requests
from PIL import Image

# CloudChef-internal harness (see deps above). These imports require
# robotics-modeling/{eval_platform,policy-runtime} on PYTHONPATH.
from eval_platform.policy.wrapper import ScoopingPolicy, PolicyConfig
from eval_platform.paths import SCOOPING_POLICY_BUNDLE, SCOOPING_LEROBOT_DIR

BASE = os.environ.get("FERN_API_BASE", "https://app.fern.bot")
KEY = os.environ.get("FERN_API_KEY", "")
VIEWS = ["high", "low", "left_wrist", "right_wrist"]
# DFoT world-model view name <- LeRobot native video key, used to build the
# step-0 frame. CloudChef's two scooping world models were trained with
# DIFFERENT camera->view conventions (see, in the modeling repo,
# scripts/convert_scooping_4view.py vs scripts/convert_cloudchef_failures.py):
#
#   cooked   (combined_alohastatic_v2_resume): front->high,    overhead->low
#   uncooked (cloudchef_failures_finetune):    overhead->high,  front->low
#
# The convention MUST match how the target episode's world model was trained,
# otherwise the two cameras land in the wrong latent channels and the rollout
# diverges immediately. Select it with --rice (see main()).
INIT_VIEW_SRC_BY_RICE = {
    "cooked": {"high": "front", "low": "overhead",
               "left_wrist": "wrist_left", "right_wrist": "wrist_right"},
    "uncooked": {"high": "overhead", "low": "front",
                 "left_wrist": "wrist_left", "right_wrist": "wrist_right"},
}


def decode_frames(frames: dict[str, str]) -> dict[str, np.ndarray]:
    """{view: src} -> {view: (H, W, 3) uint8}. `src` is base64 PNG or https URL."""
    out = {}
    for v, src in frames.items():
        raw = (requests.get(src, timeout=30).content if src.startswith("http")
               else base64.b64decode(src.split(",", 1)[-1] if src.startswith("data:") else src))
        out[v] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    return out


def init_views(lerobot_idx: int, init_view_src: dict[str, str]) -> dict[str, np.ndarray]:
    """Build the world model's step-0 frame (4 views, 256x256) from the
    episode's native LeRobot videos. ``init_view_src`` maps each DFoT view name
    to the LeRobot native video key for the chosen --rice convention."""
    root = str(SCOOPING_LEROBOT_DIR)
    out = {}
    for dfot_view, key in init_view_src.items():
        path = f"{root}/videos/observation.images.{key}/chunk-000/file-{lerobot_idx:03d}.mp4"
        container = av.open(path)
        img = None
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            break
        container.close()
        out[dfot_view] = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return out


def wait_for_init(s: requests.Session, rid: str, timeout: float = 240.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = s.get(f"{BASE}/api/eval/runs/{rid}").json()
        if d["run"].get("status") == "error":
            raise SystemExit("init failed: " + str(d["run"].get("error_msg")))
        if not d.get("initializing") and d.get("frames"):
            return
        time.sleep(2.0)
    raise SystemExit("timed out waiting for world-model init")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-id", required=True,
                    help="Fern eval-set episode whose start frame matches --lerobot-idx")
    ap.add_argument("--lerobot-idx", type=int, required=True,
                    help="source LeRobot episode index (provides GT qpos + native videos)")
    ap.add_argument("--steps", type=int, default=60, help="max effective steps to run")
    ap.add_argument("--n-steps", type=int, default=50, help="diffusion steps per frame")
    ap.add_argument("--device", default="cuda", help="torch device for the policy bundle")
    ap.add_argument("--execution", default="first_only", choices=["first_only", "halfchunk"],
                    help="policy chunk strategy (see ScoopingPolicy)")
    ap.add_argument("--name", default=None, help="run name shown in the dashboard")
    ap.add_argument(
        "--rice", choices=["cooked", "uncooked"], default="cooked",
        help="camera/view convention, which MUST match the episode's world model: "
             "'cooked' = combined_alohastatic_v2_resume (front->high, overhead->low); "
             "'uncooked' = cloudchef_failures_finetune (overhead->high, front->low). "
             "The uncooked-rice rollouts were trained with the cameras swapped, so "
             "picking the wrong one makes the world model diverge from step 0.")
    args = ap.parse_args()

    if not KEY:
        raise SystemExit("Set FERN_API_KEY")

    # `policy_mode` lives in the modeling repo's deploy/ dir; allow pointing at it.
    deploy_dir = os.environ.get("EP_DEPLOY_DIR")
    if deploy_dir:
        sys.path.insert(0, deploy_dir)
    import policy_mode as pm  # noqa: E402

    init_view_src = INIT_VIEW_SRC_BY_RICE[args.rice]

    # The policy consumes the real `overhead` + `wrist_right` cameras. Which WM
    # view carries the overhead content depends on the rice convention: it's the
    # `low` view for cooked, but the `high` view for uncooked (cameras swapped at
    # train time). eval_platform.camera_map hardcodes the cooked mapping, so for
    # uncooked we override the mapping policy_mode reads (no edit to eval_platform).
    if args.rice == "uncooked":
        from eval_platform.camera_map import CameraMapping
        pm.SCOOPING_POLICY_CAMERAS = (
            CameraMapping(policy_name="overhead",
                          lerobot_video_key="observation.images.overhead",
                          dfot_view_name="high"),
            CameraMapping(policy_name="wrist_right",
                          lerobot_video_key="observation.images.wrist_right",
                          dfot_view_name="right_wrist"),
        )
        print("[harness] uncooked convention: policy 'overhead' <- WM 'high', "
              "step-0 high<-overhead / low<-front")

    print("loading CloudChef scooping_v0 policy on", args.device, "...")
    policy = ScoopingPolicy(PolicyConfig(bundle_path=SCOOPING_POLICY_BUNDLE,
                                         device=args.device, execution=args.execution))
    state = pm.build_policy_episode_state(f"sc_{args.lerobot_idx:05d}", policy)
    n_saved = state.wm_actions_saved.shape[0]
    print(f"episode GT loaded: T_native={state.qpos_native.shape[0]} saved_tokens={n_saved}")

    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {KEY}"})

    r = s.post(f"{BASE}/api/eval/runs", json={
        "episode_id": args.episode_id,
        "name": args.name or f"cloudchef policy {time.strftime('%H:%M:%S')}",
        "policy_name": "cloudchef scooping_v0 (learned)",
    })
    r.raise_for_status()
    rid = r.json()["run"]["id"]
    print(f"run {rid} — waiting for world-model init (cold start ~60s)...")
    wait_for_init(s, rid)
    print("init done — starting closed loop")

    decoded_views_now = init_views(args.lerobot_idx, init_view_src)
    max_eff = min(args.steps, n_saved // pm.MODEL_FRAME_SKIP_OVER_SAVED - 1)
    for eff in range(max_eff):
        # 1. Build the policy's inputs (qpos history + now/t-1 cameras) from the
        #    latest world-model frame, then run the policy -> 7-D right arm.
        qpos_stack, current_qpos, cam_inputs = pm.policy_step_inputs(
            state, policy, decoded_views_now, eff)
        a7 = policy.predict_action(qpos_stack=qpos_stack, current_qpos=current_qpos,
                                   camera_frames_uint8=cam_inputs, frame_index=eff)
        # 2. Merge with GT left arm + normalize -> the 16-D action for the next token.
        curr16 = pm.apply_policy_action_to_dfot16(state, a7, eff)
        prev16 = state.wm_actions_saved[
            min(pm.MODEL_FRAME_SKIP_OVER_SAVED * (eff + 1) - 1, n_saved - 1)]
        # 3. Step the world model with training-aligned [prev_action, action].
        t0 = time.time()
        rr = s.post(f"{BASE}/api/eval/runs/{rid}/step", json={
            "action": [float(x) for x in curr16],
            "prev_action": [float(x) for x in prev16],
            "n_steps": args.n_steps,
        })
        rr.raise_for_status()
        res = rr.json()
        decoded_views_now = decode_frames(res["frames"])
        print(f"  eff {eff:3d} -> step {res['step']:3d}  ({time.time() - t0:.2f}s)")

    s.delete(f"{BASE}/api/eval/runs/{rid}")
    print(f"done — watch it at {BASE}/eval/runs/{rid}")


if __name__ == "__main__":
    main()

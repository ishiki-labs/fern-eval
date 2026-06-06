"""Self-contained Fern world-model action contract — pure numpy, no deps.

This module reproduces the only non-obvious parts you need to talk to the Fern
`/step` API: the exact action **normalization** and the **rate** constants. The
reference harness (`cloudchef_scooping_runner.py`) uses it; you can too if you
write your own loop. Pure numpy, no torch, no internal repos.

You still bring your own policy, its camera preprocessing, and your GT data; this
just guarantees the 16-D action you POST matches what the world model was trained
on. Get the layout or normalization wrong and the rollout diverges from step 0.

──────────────────────────────────────────────────────────────────────────────
The 16-D action layout (what the world model was actually trained on)

The WM is conditioned on a 14-D LeRobot action padded to 16-D with zeros, then
per-dim normalized to [-1, 1] using the dataset's `action_stats.npz`:

    index 0..6   -> LEFT arm   (6 joints + gripper)      from LeRobot action[0:7]
    index 7..13  -> RIGHT arm  (6 joints + gripper)      from LeRobot action[7:14]
    index 14,15  -> zero pad   (normalized, ~constant)

  NOTE: this is the `lerobot14_to_dfot16` layout — NOT a duplicate-gripper
  "0..7 left / 8..15 right" layout. The grippers are at index 6 (left) and 13
  (right); 14/15 are pads. Normalization uses amin/amax from action_stats.npz —
  raw joint values will NOT be in [-1, 1] until you normalize.

──────────────────────────────────────────────────────────────────────────────
The rate (why MODEL_FRAME_SKIP_OVER_SAVED matters)

Training preprocessing subsamples native 30 FPS by FRAME_STRIDE=3 -> 10 FPS
("saved" rate), then the model is trained with frame_skip=2 over that. So one WM
token = MODEL_FRAME_SKIP_OVER_SAVED (2) saved-rate steps = EFF_TO_NATIVE (6)
native frames (200 ms, 5 FPS effective). Per `/step` you advance ONE token, and
the training-aligned conditioning for token t is the pair [a_{2t-1}, a_{2t}]:

    POST /step  { "action": a_2t (16-D), "prev_action": a_{2t-1} (16-D), "n_steps": 50 }

For a TRUE closed-loop eval, BOTH a_{2t-1} and a_{2t} must come from your policy
(see the note in `closed_loop_pair` below). Sourcing a_{2t-1} from GT leaks the
real trajectory each step and flatters the policy.
"""
from __future__ import annotations

import numpy as np

# 16-D padded action ("DFoT-16"): 14-D LeRobot action + 2 zero pads.
ACTION_DIM = 16
LEFT_ARM = slice(0, 7)    # 6 joints + gripper
RIGHT_ARM = slice(7, 14)  # 6 joints + gripper

# Rate constants (keep in sync with the WM training preprocessing).
FRAME_STRIDE = 3                 # native 30 FPS -> 10 FPS saved rate
MODEL_FRAME_SKIP_OVER_SAVED = 2  # WM tokens per 2 saved-rate steps
EFF_TO_NATIVE = FRAME_STRIDE * MODEL_FRAME_SKIP_OVER_SAVED  # = 6 native frames / token


def load_action_stats(stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (amin, amax) as (16,) float32 from `action_stats.npz`.

    Both vectors are required to reproduce the WM's training-time normalization.
    The file ships next to each eval dataset (e.g.
    s3://fern-robotics-data/processed/scooping_curated_20/action_stats.npz)."""
    d = np.load(stats_path)
    amin = np.asarray(d["amin"], dtype=np.float32).reshape(-1)
    amax = np.asarray(d["amax"], dtype=np.float32).reshape(-1)
    if amin.shape != (ACTION_DIM,) or amax.shape != (ACTION_DIM,):
        raise ValueError(f"expected (16,) stats, got amin={amin.shape} amax={amax.shape}")
    return amin, amax


def lerobot14_to_dfot16(
    action_14: np.ndarray, amin: np.ndarray, amax: np.ndarray
) -> np.ndarray:
    """14-D raw LeRobot action -> normalized 16-D action the WM expects.

    Pads to 16-D with zeros at indices 14,15, then normalizes every dim to
    [-1, 1] with the per-dim training amin/amax. Accepts (...,14) arrays.
    Exact reproduction of `_normalize_actions` in convert_scooping_4view.py."""
    if action_14.shape[-1] != 14:
        raise ValueError(f"want last dim 14, got {action_14.shape}")
    pad = np.zeros(action_14.shape[:-1] + (ACTION_DIM - 14,), dtype=np.float32)
    raw16 = np.concatenate([action_14.astype(np.float32), pad], axis=-1)
    rng = amax - amin
    safe = np.where(rng < 1e-4, 1.0, rng)
    offs = np.where(rng < 1e-4, (amin + amax) / 2.0, amin)
    norm = 2.0 * (raw16 - offs) / safe - 1.0
    return np.clip(norm, -1.0, 1.0).astype(np.float32)


def build_wm_action_16d(
    gt_action_14: np.ndarray,
    right_arm_7: np.ndarray,
    amin: np.ndarray,
    amax: np.ndarray,
) -> np.ndarray:
    """Merge your policy's 7-DoF right-arm output with the GT left arm, then
    normalize -> the 16-D action to POST as `action`/`prev_action`.

    `gt_action_14` is the raw 14-D LeRobot action at the matching timestep (its
    left-arm slice [0:7] is kept so the un-driven arm follows ground truth);
    `right_arm_7` is your policy's raw right-arm command (LeRobot units, NOT
    pre-normalized). Indices 7:14 are overwritten with it."""
    if gt_action_14.shape[-1] != 14:
        raise ValueError(f"gt_action_14 must be 14-D, got {gt_action_14.shape}")
    if right_arm_7.shape != (7,):
        raise ValueError(f"right_arm_7 must be (7,), got {right_arm_7.shape}")
    merged = gt_action_14.astype(np.float32).copy()
    merged[7:14] = right_arm_7
    return lerobot14_to_dfot16(merged, amin, amax)


def normalize_right_arm_7(
    right_7_raw: np.ndarray, amin: np.ndarray, amax: np.ndarray
) -> np.ndarray:
    """Normalize a raw 7-D right-arm command to [-1, 1] using the right-arm
    slice of the training stats. Per-dim, independent of the other dims."""
    if right_7_raw.shape != (7,):
        raise ValueError(f"right_7_raw must be (7,), got {right_7_raw.shape}")
    lo, hi = amin[RIGHT_ARM], amax[RIGHT_ARM]
    rng = hi - lo
    safe = np.where(rng < 1e-4, 1.0, rng)
    offs = np.where(rng < 1e-4, (lo + hi) / 2.0, lo)
    return np.clip(2.0 * (right_7_raw.astype(np.float32) - offs) / safe - 1.0,
                   -1.0, 1.0).astype(np.float32)


def splice_right_arm(
    gt_action_16_norm: np.ndarray,
    right_7_raw: np.ndarray,
    amin: np.ndarray,
    amax: np.ndarray,
) -> np.ndarray:
    """Overwrite the right-arm dims (7:14) of an ALREADY-normalized 16-D action
    with your policy's raw 7-D command (normalized in place).

    Use this with the API's `gt_actions` (which are already normalized, saved
    rate): copy `gt_actions[k]`, splice in your right-arm command, POST it. The
    left arm + pads stay at their GT-normalized values, so the un-driven arm
    follows ground truth — exactly what `apply_policy_action_to_dfot16` does,
    but with zero internal deps."""
    out = np.asarray(gt_action_16_norm, dtype=np.float32).copy()
    if out.shape != (ACTION_DIM,):
        raise ValueError(f"gt_action_16_norm must be (16,), got {out.shape}")
    out[RIGHT_ARM] = normalize_right_arm_7(np.asarray(right_7_raw, np.float32), amin, amax)
    return out


def closed_loop_pair(
    curr_right_7: np.ndarray,
    prev_right_7: np.ndarray,
    gt_curr_14: np.ndarray,
    gt_prev_14: np.ndarray,
    amin: np.ndarray,
    amax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (action=a_2t, prev_action=a_{2t-1}) for one `/step`, both 16-D.

    For a faithful CLOSED-LOOP eval, source BOTH right-arm commands from YOUR
    policy — `curr_right_7` is the policy's command for this token and
    `prev_right_7` is its command for the intermediate sub-step (e.g. the 2nd
    entry of the previous token's predicted action chunk, since the policy runs
    at the saved rate while the WM steps every 2 saved steps).

    Don't pull `prev_right_7` from ground truth in a policy eval: that injects a
    GT correction each step and is no longer closed-loop. (The `oracle` baseline
    deliberately uses GT for both — that's a different thing.)

    The left arm in both pairs comes from GT (`gt_*_14`), since this policy only
    drives the right arm and the left arm is meant to hold its real trajectory."""
    a_curr = build_wm_action_16d(gt_curr_14, curr_right_7, amin, amax)
    a_prev = build_wm_action_16d(gt_prev_14, prev_right_7, amin, amax)
    return a_curr, a_prev


if __name__ == "__main__":
    # Tiny smoke test with synthetic stats (no real data needed).
    amin = np.full(ACTION_DIM, -1.5, dtype=np.float32)
    amax = np.full(ACTION_DIM, 1.5, dtype=np.float32)
    gt14 = np.zeros(14, dtype=np.float32)
    right7 = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    a16 = build_wm_action_16d(gt14, right7, amin, amax)
    assert a16.shape == (ACTION_DIM,) and a16.min() >= -1 and a16.max() <= 1
    print("16-D action:", np.round(a16, 3))
    print("left arm:", np.round(a16[LEFT_ARM], 3), "right arm:", np.round(a16[RIGHT_ARM], 3))
    print(f"rate: 1 /step = {MODEL_FRAME_SKIP_OVER_SAVED} saved steps = {EFF_TO_NATIVE} native frames")

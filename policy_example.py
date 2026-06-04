"""Example black-box policy for the Fern eval CLI.

Copy this file, drop in your real model, and run:

    python eval_client.py --scene cloudchef-scooping --policy-file policy_example.py --steps 40

Your function must be named `policy` (or pass --policy-fn NAME) with the signature
below. Fern never sees your model — it only receives the 16-D action you return.
"""
from __future__ import annotations

import numpy as np


def policy(
    obs: dict[str, np.ndarray],
    step: int,
    init_action: list[float] | None,
) -> list[float]:
    """Return the next 16-D ALOHA action, normalized to [-1, 1].

    Args:
        obs: current camera frames. obs[view] is an (H, W, 3) uint8 array.
             views = "high", "low", "left_wrist", "right_wrist".
        step: 1-based step index within the run.
        init_action: the episode's starting joint target (16-D) for eval-set
            episodes, else None. Actions are ABSOLUTE targets, so this is the
            pose that holds the start ("zeros" is mid-range, not "stay still").

    Layout: indices 0..7 = left arm (joints 0..6 + gripper at 7),
            indices 8..15 = right arm (joints 0..6 + gripper at 15).
    """
    # --- replace everything below with your model's forward pass ---
    # e.g.  action = my_model.predict(obs)
    action = list(init_action) if init_action else [0.0] * 16

    # Toy behavior: slowly close the right gripper so you can see motion.
    action[15] = float(np.clip(-1.0 + step * 0.05, -1.0, 1.0))
    return action

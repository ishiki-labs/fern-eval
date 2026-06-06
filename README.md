# Fern Eval — run your robot policy against a learned world model

Evaluate your robot policy against Fern's hosted **world model** — no real
hardware required. The world model is a **black-box simulator**: you send a
16‑D action, it returns the next camera frames; you feed those frames into
**your** policy to pick the next action, and repeat (closed loop). Fern hosts
nothing about your policy — your weights never leave your machine.

```
your policy  ──action──▶  Fern world model  ──next frames──▶  your policy  ──▶ …
```

- **Base URL:** `https://app.fern.bot`
- **Dashboard:** every run shows up at `https://app.fern.bot/eval` as a video
- **Live API reference:** https://app.fern.bot/eval/docs

---

## 1. Install

Requires Python 3.9+.

```bash
git clone https://github.com/ishiki-labs/fern-eval.git
cd fern-eval
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## 2. Get an API key

You authenticate with a bearer token (`fern_sk_…`). Request one from your Fern
contact (founders@fern.bot) or generate it in the dashboard. Then export it:

```bash
export FERN_API_KEY=fern_sk_xxxxxxxxxxxxxxxxxxxx
```

## 3. Quick start

Run the included demo policy against a built‑in scene:

```bash
python eval_client.py --scene cloudchef-scooping --steps 40
```

You'll see step‑by‑step latency printed, frames saved under `./rollout/`, and a
link to watch the run in the dashboard.

## 4. Plug in your own policy

Your policy lives in a plain `.py` file that defines a `policy(...)` function.
Copy [`policy_example.py`](./policy_example.py) and drop in your model:

```python
# my_policy.py
import numpy as np

def policy(obs: dict[str, np.ndarray], step: int, init_action: list[float] | None) -> list[float]:
    # obs[view] is an (H, W, 3) uint8 array; views: high, low, left_wrist, right_wrist
    # return a 16-D ALOHA joint-target vector, normalized to [-1, 1]
    action = list(init_action) if init_action else [0.0] * 16
    # ... your model's forward pass ...
    return action
```

Then point the CLI at it:

```bash
python eval_client.py --scene cloudchef-scooping --policy-file my_policy.py --steps 40
```

> Use a different function name with `--policy-fn NAME`.

### Advanced: proprioceptive / bimanual policies

The `policy(obs, step, init_action)` interface above is for **vision‑only**
policies (cameras in, 16‑D action out). Some policies need more — e.g.
CloudChef's rice‑scooping policy consumes a **joint (qpos) history**, only
drives **one arm** (the other is filled from ground truth), and is rolled out
with training‑aligned **per‑token conditioning** `[a_{2t-1}, a_{2t}]`.

That doesn't fit a stateless callable, so you drive the loop yourself and call
`POST /api/eval/runs/{id}/step` directly — same API, you just build a richer
16‑D action each step and pass `prev_action` for faithful conditioning.
[`examples/cloudchef_scooping_runner.py`](./examples/cloudchef_scooping_runner.py)
is a worked template for exactly this. It is **fully self‑contained** — no
internal repo, no native‑video reads — and you plug in just your policy:

```bash
pip install -r examples/requirements.txt           # requests, numpy, pillow
export FERN_API_KEY=fern_sk_...
export FERN_ACTION_STATS=/path/to/action_stats.npz # for right-arm normalization
# Edit predict_right_arm_7() in the file to call your model, then:
python examples/cloudchef_scooping_runner.py --episode-id <uuid> --steps 60
```

The template creates the run, waits for init, pulls the episode's ground‑truth
actions (`gt_actions`) from the API for the **un‑driven left arm**, splices in
your policy's 7‑D right‑arm command (normalized via `action_stats.npz`), and
steps the world model with training‑aligned `[prev_action, action]`. The only
external file you supply is `action_stats.npz` (ships next to each dataset).

**Episode → ground-truth mapping.** Each eval‑set episode was seeded from a real
recorded episode, and the API exposes that mapping so the client only needs the
episode id. `GET /api/eval/episodes/{id}` returns a `gt` block:

```json
{ "gt": { "set": "scooping-lerobot-v3.0", "episode": "lerobot_051",
          "index": 51, "model": "combined_alohastatic_v2_resume", "rice": "cooked" } }
```

The template reads `gt.rice` (camera convention) from this automatically and
passes it to your policy; you don't pass an index because the left‑arm GT comes
straight from `gt_actions` on the run.

The `prev_action` field on `POST /step` is the key primitive: when set, the
world model conditions the new token on `[prev_action, action]` and steps one
token at a time (instead of the default 4‑frame chunk). `GET /api/eval/runs/{id}`
returns `gt_actions` (the full saved‑rate ground truth, normalized 16‑D) so you
can index `[gt_actions[2t-1], gt_actions[2t]]` per token. For a **true closed
loop**, source the right arm of *both* from your policy (the template tracks your
previous command for `prev_action`; `--prev-source gt` reproduces the oracle
baseline, which leaks the real trajectory and is not a real eval).

**rice (camera convention):** CloudChef's two scooping world models were trained
with their `overhead`/`front` cameras **swapped** — cooked
(`combined_alohastatic_v2_resume`) uses `front→high, overhead→low`, while
uncooked (`cloudchef_failures_finetune`) uses `overhead→high, front→low`. Your
policy must map the world‑model views to its expected cameras using `gt.rice`
(passed to `predict_right_arm_7(views, step, rice)`), or it sees the wrong
cameras. The API‑resolved `gt.rice` is authoritative.

### The action contract, self-contained

The template builds actions via
[`examples/fern_action_contract.py`](./examples/fern_action_contract.py) —
pure numpy, no internal deps. Useful if you'd rather write your own loop:

- `load_action_stats(path)` — load `(amin, amax)` from `action_stats.npz`.
- `lerobot14_to_dfot16(action_14, ...)` — exact training‑time normalization of a
  raw 14‑D LeRobot action → normalized 16‑D.
- `splice_right_arm(gt_action_16, right_7, ...)` — overwrite the right‑arm dims of
  an already‑normalized GT action (e.g. from `gt_actions`) with your raw 7‑D
  command. What the template uses each step.
- `build_wm_action_16d(gt_14, right_7, ...)` / `closed_loop_pair(...)` — same idea
  from raw 14‑D GT if you have it instead of the API's normalized `gt_actions`.
- Constants `FRAME_STRIDE=3`, `MODEL_FRAME_SKIP_OVER_SAVED=2`, `EFF_TO_NATIVE=6`.

## 5. The action contract

Each action is a **16‑D vector, normalized to `[-1, 1]`** — a 14‑D LeRobot ALOHA
action padded to 16‑D, then per‑dim normalized via the dataset's
`action_stats.npz`:

```
indices 0..6   -> left arm   (6 joints + gripper at index 6)
indices 7..13  -> right arm  (6 joints + gripper at index 13)
indices 14,15  -> zero pad
```

Normalization uses each dim's `amin`/`amax`, so **raw joint values are not in
`[-1, 1]` until you normalize** (see `fern_action_contract.lerobot14_to_dfot16`).
These are **absolute targets, not deltas** — so all‑zeros is **not** "stay
still" (it commands every joint to its mid‑range pose). To hold the start pose,
replay the episode's `init_action` (returned by the run/episode reads for
eval‑set episodes; passed to your `policy()` as the `init_action` argument).
Replaying it is a good fidelity baseline — a faithful world model shows ~no
motion. The CLI's `--hold` flag does exactly this.

Cameras are 4 views (`high`, `low`, `left_wrist`, `right_wrist`), 256×256 RGB.

## 6. Concepts

```
scene      a customer model (a "folder"), e.g. cloudchef-scooping
  episode    a starting frame under a scene
    run        one eval on an episode; each step appends a frame -> a video
```

Runs are **pausable/resumable**: every `step` is one inference call and the
run's frames are stored server‑side, so you can stop and pick up later — or
watch the run live in the dashboard at `/eval/runs/{id}`.

## 7. CLI reference

```
python eval_client.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--scene` | `cloudchef-scooping` | Scene slug to pick an episode from |
| `--episode-id` | — | Explicit episode id (overrides `--scene`) |
| `--policy-file` | — | Path to a `.py` file defining `policy(obs, step, init_action)` |
| `--policy-fn` | `policy` | Function name to call inside `--policy-file` |
| `--policy` | auto | Policy name recorded/shown in the dashboard |
| `--hold` | off | Replay the episode's initial joint target every step (fidelity baseline) |
| `--steps` | `40` | Number of closed‑loop steps to run |
| `--out` | `./rollout` | Directory to save frames into |
| `--save-views` | `high` | Comma‑separated views to save each step, or `all` |
| `--n-steps` | server default | Diffusion steps per frame (lower = faster/noisier) |

Environment variables: `FERN_API_KEY` (required), `FERN_API_BASE` (defaults to
`https://app.fern.bot`).

## 8. API endpoints (under the hood)

The CLI is a thin wrapper over a small REST API. Full reference:
**https://app.fern.bot/eval/docs**.

```
GET    /api/eval/scenes                       -> your scenes (models)
GET    /api/eval/scenes/{scene}/episodes      -> episodes (starting frames)
POST   /api/eval/runs                         -> start a run (async init; poll for step 0)
POST   /api/eval/runs/{id}/step               -> next frames  (repeat)
GET    /api/eval/runs/{id}                    -> run + all frames (the video)
DELETE /api/eval/runs/{id}                    -> end + free GPU state
```

All requests require `Authorization: Bearer <FERN_API_KEY>`. Run creation
**spawns** the world‑model init on the GPU and returns immediately (a cold start
can take ~60s), so the CLI polls `GET /runs/{id}` until step 0 is ready before
stepping.

## 9. Troubleshooting

- **`Set FERN_API_KEY`** — export your key (see step 2).
- **`401`** — missing / invalid / revoked key.
- **`No episodes under scene '…'`** — pick a different scene, or pass
  `--episode-id` directly. Browse available episodes at `https://app.fern.bot/eval`.
- **Run seems stuck on init** — the GPU may be cold‑starting (~60s) or busy;
  the CLI polls for up to 3 minutes.
- **`404` mid‑run** — the GPU‑side state for that run expired; start a new run.

## License

[MIT](./LICENSE) © Ishiki Labs

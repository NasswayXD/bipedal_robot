# bipedal_robot

Custom 10-joint biped robot — RL locomotion policy trained in Isaac Lab, deployed in Isaac Sim.

---

## Repository Structure

```
bipedal_robot/
├── my_biped/               # Training (Isaac Lab / RSL-RL)
│   ├── __init__.py             # Gym environment registration
│   ├── my_biped_env_cfg.py     # Environment, rewards, events
│   └── my_biped_robot_cfg.py   # Robot articulation, PD gains, default pose
│
├── biped_policy/           # Sim deployment (Isaac Sim, no ROS)
│   ├── biped_policy.py         # PolicyController subclass — runs the policy
│   ├── biped_scene.py          # Scene setup + keyboard control
│   └── biped_env.yaml          # Sim parameters (must match training exactly)
│
└── final_test/Assem2/      # Robot URDF and USD files
```

---

## Robot

- **10 joints**, 5 per leg: `hip_one` (yaw) → `hip_two` (roll) → `hip_leg` (pitch) → `knee_leg` → `ankle_leg`
- **Actuators:** ST3215 servos
- **Observation space:** 42 dimensions
- **Action space:** 10 joint position targets (deltas around default pose)
- **Default pose:** knees bent — `hip_leg=±0.4`, `knee=∓0.95`, `ankle=±0.55`

---

## Training

### Install
```bash
pip install -e my_biped/
```

### Train from scratch
```bash
~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task MyBiped-v0 --num_envs 4096
```

### Resume a run
```bash
~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task MyBiped-v0 --resume --load_run 2026-03-24_01-56-27
```

### Play / visualize a checkpoint
```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task MyBiped-v0 --load_run 2026-03-24_13-02-02 --num_envs 16
```

### Export policy to TorchScript
```bash
python scripts/reinforcement_learning/rsl_rl/export.py --task MyBiped-v0 --checkpoint logs/rsl_rl/my_biped_flat/<run>/model.pt
```

> **Known exporter bug:** `exporter.py` references `self.normalizer` but checkpoints save it as `obs_normalizer`, so normalizer stats are silently dropped on export.
> **Fix:** rename every `self.normalizer` → `self.obs_normalizer` in the exporter before running.

### Sanity check an exported policy
Run this to verify the exported policy produces reasonable outputs before deploying:
```bash
python3 - <<'EOF'
import torch, io, glob
runs = sorted(glob.glob('/home/nassway/IsaacLab/logs/rsl_rl/my_biped_flat/*/exported/policy.pt'))
for r in runs[-3:]:
    with open(r, 'rb') as f:
        policy = torch.jit.load(io.BytesIO(f.read()))
    obs = torch.zeros(1, 42)
    obs[0, 8] = -1.0   # set gravity_z = -1 (upright robot)
    out = policy(obs)
    print(f"{r.split('my_biped_flat/')[1][:19]}: max={out.abs().max().item():.3f}")
EOF
```
A healthy policy should output `max` values in the range ~0.1–1.5. Values above ~5.0 mean the normalizer was not exported correctly.

---

## Sim Deployment

### Setup
1. Update the three paths at the top of `biped_policy/biped_scene.py`:
```python
POLICY_PATH   = "/path/to/exported/policy.pt"
ENV_PATH      = "/path/to/biped_policy/biped_env.yaml"
URDF_USD_PATH = "/path/to/final_test/Assem2/Assem2.usd"
```

2. Open Isaac Sim → Script Editor → paste and run:
```python
import sys
sys.path.insert(0, "/path/to/biped_policy/")
exec(open("/path/to/biped_policy/biped_scene.py").read())
```

### Controls
| Key | Command |
|-----|---------|
| ↑ / Numpad 8 | Walk forward |
| ↓ / Numpad 2 | Walk backward |
| ← / Numpad 4 | Turn left |
| → / Numpad 6 | Turn right |

---

## Key Notes

**`biped_env.yaml` is the contract between training and deployment.** PD gains, default joint positions, and decimation must match `my_biped_robot_cfg.py` exactly. Any drift here silently degrades the policy.

**`empirical_normalization = True`** must be set in the PPO runner config. Without it the policy outputs high-magnitude actions for any input, causing immediate instability on the first step.

**Spawn height:** Isaac Sim and Isaac Lab handle ground contact slightly differently. If the robot spawns floating, lower `position=np.array([0.0, 0.0, Z])` in `biped_scene.py` by ~0.02–0.05 m.

**Policy runs at 50Hz** — physics at 200Hz (dt=0.005), decimation=4.

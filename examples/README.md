# Baseline examples

These programs are consumers of the public `dsle` API. None of their policy or
learning code is imported by the environment package.

| File | Baseline | Paper defaults |
|---|---|---|
| `random_agent.py` | Uniform random | 100 evaluation episodes |
| `expert_agent.py` | Reactive heal/advance/attack rules | Heal below 40% HP |
| `ppo.py` | PPO with Nature CNN | 128-step rollout, 4 epochs, batch 32, 100k steps |
| `dqn.py` | DQN with Nature CNN | 100k replay, batch 32, 100k steps |
| `scope.py` | SCOPE optimized by CMA-ES | k=100, p=90, sigma=0.5, 40 generations |
| `scope_parallel.py` | SCOPE/CMA-ES on an API-managed instance pool | Same SCOPE defaults |

Install the optional dependencies before PPO, DQN, or SCOPE:

```bash
python -m pip install -e '.[examples]'
```

Run examples inside the NVIDIA runtime container so they share `/proc`, X11,
and Wine with the selected game instance:

```bash
# Core image: random and expert
./scripts/build.sh

# Research image: also installs CUDA PyTorch, SciPy, and CMA-ES
./scripts/build.sh --research
```

```bash
./scripts/exec.sh python3 examples/random_agent.py --boss asylum_demon --instance dsr-1
./scripts/exec.sh python3 examples/expert_agent.py --boss asylum_demon --instance dsr-1
./scripts/exec.sh python3 examples/ppo.py train --boss asylum_demon --instance dsr-1 --total-timesteps 100000
./scripts/exec.sh python3 examples/dqn.py train --boss asylum_demon --instance dsr-1 --total-timesteps 100000
./scripts/exec.sh python3 examples/scope.py train \
  --boss asylum_demon \
  --instances dsr-1,dsr-2 \
  --output /var/lib/dsle/results/scope-asylum
```

Inside the image, default JSONL metrics, checkpoints, and SCOPE artifacts are
written beneath `$DSLE_OUTPUT_DIR` (`/var/lib/dsle/results`). That directory is
in the persistent host output bind and survives container removal.

Seeds affect agent sampling, initialization, and optimization. They cannot seed
Dark Souls' separate internal RNG.

## API-managed parallel SCOPE

`scope_parallel.py` is a separate host-side training example. It creates one
owned container from a prebuilt image, starts `--num-instances N` games inside
it, and reuses those N public environment proxies for the complete run. The
game installation is always mounted read-only; it is never copied into the
image. Install the examples extra on the host, build the runtime image once,
then run:

```bash
python examples/scope_parallel.py \
  --boss asylum_demon \
  --game-dir /path/to/DARK-SOULS-REMASTERED \
  --image dsle-runtime:0.1.0 \
  --num-instances 4 \
  --population 10 \
  --output-dir runs/scope-asylum
```

Each candidate begins with `env.reset()`. If the population is smaller than N,
surplus instances stay idle; if it is larger, candidates run in batches while
result order remains aligned with CMA-ES. Normal completion and failures close
the proxies and remove the owned container, but keep the output directory.
Metrics, best weights, optimizer state, and periodic checkpoints are stored in
`OUTPUT/results/scope-parallel/`.

Resume from the latest optimizer state with:

```bash
python examples/scope_parallel.py \
  --game-dir /path/to/DARK-SOULS-REMASTERED \
  --output-dir runs/scope-asylum \
  --resume-checkpoint runs/scope-asylum/results/scope-parallel/scope-state-latest.npz
```

Resume restores the persisted global best, so a worse later generation cannot
replace the existing `best.npy` artifact. Checkpoints are bounded, data-only
NumPy archives containing the generation, mean, and step size; they never
deserialize Python objects. CMA-ES restarts from those values, so its covariance
and evolution paths begin fresh rather than being restored exactly.

The existing `scope.py` remains the compact serial/manual-attachment baseline
for execution inside an already-running container.

## Recorded boss showcase

`boss_showcase.py` is a host-side API example rather than an agent baseline. It
starts one owned VNC-enabled instance, reuses it across every supported regular
boss, passively monitors each fight for 10 seconds or until death, returns to a
verified menu, and records a 3-second menu gap before the next scenario. The
whole run is written as one 30 FPS constant-frame-rate MP4 plus a JSON cut
manifest.

Use the repository launcher so the live source tree is importable:

```bash
DSLE_PYTHON=.venv/bin/python ./scripts/record-boss-showcase.sh \
  --game-dir Dark.Souls.Remastered.v1.04 \
  --output-dir runs/boss-showcase
```

OpenCV comes from the `runtime` optional dependency group. The attached game
directory remains read-only, and normal completion or failure removes only the
owned container while preserving the prebuilt image and output directory.

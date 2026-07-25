# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research project **Adaptive-TERL** (target: ICML 2027): replacing TERL's hand-tuned two-stage schedule with an adaptive controller (SGSA — Stagnation-Gated Softmax Allocation). Two trees:

- `asterl/` + `train.py` + `launch.py` + `configs/` + `tests/` — the active codebase (gymnasium + MuJoCo -v5, Python 3.11 conda env `asterl`).
- `TERL-main/` — the original paper's code (gym 0.23 + mujoco210, old API). **Read-only reference**; its `learning_curves/` holds the paper's recorded TensorBoard data — don't modify anything here.

The full research/experiment plan (phases, compute budget, laptop-vs-server split) lives at `~/.claude/plans/develop-an-adaptive-scheduler-iridescent-pony.md`.

## Commands

```bash
conda run -n asterl python train.py --algo aterl --env Hopper-v5 --seed 0   # single run (algo: td3|terl|aterl)
conda run -n asterl python train.py --algo terl --env Hopper-v5 --seed 0 --set ratio=0.1 --wandb online
conda run -n asterl python launch.py --algo terl,td3 --env Hopper-v5,HalfCheetah-v5 --seeds 0-4 --workers 5
conda run -n asterl python -m pytest tests/ -x -q                            # test suite
conda run -n asterl python -m pytest tests/test_determinism.py -q            # slowest tests (real short runs)
```

Run dirs are deterministic (`runs_v2/<env>/<algo>/seed<N>/`) — rerunning the same command **resumes from checkpoint**; a `DONE` marker means finished (use `--fresh` to redo). `launch.py` pins `OMP_NUM_THREADS=1` per worker: the workload is CPU-bound, parallelism comes from concurrent runs (~5-6 on this laptop).

## Architecture (asterl)

- `asterl/algos/terl.py` — `PopulationTrainerBase` (shared rollout/PSO/eval/checkpoint plumbing) + `TERLTrainer`, a faithful de-globalized port of `TERL-main/TERL.py` with the fixed `ratio` stage switch. `PopulationState` holds what used to be module globals. PSO is generic over `actor.state_dict()` (no hardcoded layer names).
- `asterl/algos/aterl.py` — `ATERLTrainer`: same population machinery, but rollout/gradient allocation and the PSO interval come from the controller each round instead of the hard stage switch and `extra_idx` heuristic.
- `asterl/controller/` — `signals.py` (stagnation gate g, rank-normalized fitness/Δ windows, behavioral diversity) and `allocator.py` (`SGSAController`: softmax-τ gated by g; `FixedStageController`: recovers TERL's schedule through the same interface — used for fallback and regression testing).
- `asterl/common/` — buffer (float32, checkpointable), evaluator (gymnasium 5-tuple; **bootstrap mask = terminated only**), config (dataclass + YAML layering + `--set k=v`), logger (always writes `metrics.jsonl`; **W&B online + TensorBoard are the project defaults** via configs/default.yaml — use `--wandb disabled` for throwaway runs), checkpoint (atomic write), seeding (full RNG capture incl. env RNGs → bit-exact resume on CPU).
- `configs/env/*.yaml` — per-env quirks ported from TERL (Swimmer discount 0.999, HalfCheetah stable_eval_times 1, Pendulum/LunarLander fitness_eval_times 5).

Invariants to preserve: total gradient steps per env step = 1 (UTD=1, compute-fairness across all methods); every stored transition counts toward `max_timesteps`; checkpoint only at round boundaries (that's what makes resume exact); controller internals (g, τ, p_i, diversity) are logged every round — they are the paper's analysis figures.

## Configuration

The repo root `.env` holds `WANDB_API_KEY` (the user's Weights & Biases API key). The current code does not load it — `TERL.py` only logs to TensorBoard — so any W&B integration must read it explicitly (e.g. `python-dotenv` or sourcing it before launch). Never print or commit its contents.

## Architecture

- **`TERL.py`** — entry point and the TERL algorithm itself. Module-level globals (argparse, `Parameters`, stage/fitness bookkeeping lists) are shared with the `Agent` class via `global` statements; the training state is deliberately not encapsulated, so edits to `Agent.train()` usually require touching those globals too.
  - `Parameters` hardcodes per-environment settings: `max_timesteps`, `max_episode_steps`, `stable_eval_times` (extra evaluations for high-variance envs), and the exploration `ratio` (0.25).
  - **Two-stage scheme**: stage 1 (exploration) runs until `timesteps >= max_timesteps * ratio`. A population of `pop_size` independent TD3 agents is evaluated; each trains its own actor/critic. Stage 2 (exploitation) redirects all gradient steps to the current best individual (`best_idx`), and better individuals overwrite the best's actor.
  - **PSO step** (`Agent.pso()`): non-best actors' weights are moved toward the global best (`gbest`) with a velocity term stored in `self.V`, layer by layer (`l1`/`l2`/`l3` weight and bias tensors — coupled to the exact `Actor` layer names in `TD3.py`). Update frequency: every 1e4 learned steps in stage 1, 1e3 in stage 2.
  - All individuals share one replay buffer; the reported score comes from a separate `test_individual` actor snapshot evaluated every 5e3 timesteps.
- **`TD3.py`** — standard TD3 (Fujimoto et al.) with `Actor`, twin-`Critic`, and the `TD3` trainer class. Swimmer-v2 uses `discount=0.999`; everything else uses defaults.
- **`utils.py`** — numpy-backed `ReplayBuffer` (1e6 capacity).

The README notes the default exploration ratio (0.25) is not optimal per environment (e.g. HalfCheetah does better with a smaller value).

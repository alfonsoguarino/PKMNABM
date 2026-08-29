"""
reward_study.py — Sensitivity of the learned policy to the reward weights and
to the discount factor gamma.

Motivation
----------
The paper argues that the reward weights are the sandbox's *explicit normative
stance*, and that gamma encodes the decision-maker's time horizon. This script
turns that claim into evidence: it re-trains the super-agent under a set of
alternative "regulatory personalities" and under several values of gamma, and
records how the learned policy shifts.

Each condition trains a fresh DDQN and evaluates its greedy policy, exactly as
in rl/train.py. The script then writes two tidy summary CSVs — one for the
reward-weight scenarios, one for the gamma sweep — plus, for every condition,
the full per-episode test log and the action distribution, so the figures can
be regenerated later.

Design choices
--------------
* We do NOT change the market model, only the reward (a SimConfig field) and
  gamma (a DQNConfig field). This isolates the effect of *objectives* and
  *patience* from any change in market mechanics.
* Every condition is trained from the same seed so that differences are due to
  the reward/gamma, not to random initialisation. For error bars, pass more
  seeds via --seeds; the summary then reports mean and std across seeds.
* Episodes default to a smaller number than the headline run (the point here is
  relative comparison across conditions, not a single best policy), but this is
  configurable.

Usage
-----
    # full study (recommended), ~a few hours on CPU depending on --episodes
    python -m rl.reward_study --episodes 120 --seeds 3 --out results/reward_study

    # quick smoke test
    python -m rl.reward_study --episodes 20 --seeds 1 --out results/reward_study_quick
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json

import numpy as np
import pandas as pd

from config import SimConfig
from pokesim.model import ACTIONS, N_ACTIONS
from rl.ddqn import DQNConfig
from rl.train import train


# ---------------------------------------------------------------------- #
# 1. The "regulatory personalities": alternative reward-weight scenarios #
# ---------------------------------------------------------------------- #
# Each scenario overrides a subset of the reward weights. The names are the
# ones that will appear in the figures and in the paper. Keep them short.
#
# baseline        : the weights used in the headline experiment
# price_hawk      : cares almost only about price stability (bubble + collapse)
# access_first    : prioritises serving collectors above all
# profit_seeker   : prioritises manufacturer margin (the "greedy firm")
# anti_bubble     : extreme aversion to bubbles only
# balanced_soft   : all weights equal and mild
REWARD_SCENARIOS = {
    "baseline": {},
    "price_hawk":    dict(w_bubble=3.0, w_collapse=3.0, w_access=0.3, w_margin=0.3),
    "access_first":  dict(w_access=3.0, w_afford=2.0, w_margin=0.3),
    "profit_seeker": dict(w_margin=3.0, w_bubble=0.3, w_collapse=0.3, w_access=0.3),
    "anti_bubble":   dict(w_bubble=4.0, w_collapse=0.5, w_margin=0.5, w_access=0.5),
    "balanced_soft": dict(w_access=1.0, w_margin=1.0, w_unsold=1.0,
                          w_afford=1.0, w_bubble=1.0, w_collapse=1.0),
}

# ---------------------------------------------------------------------- #
# 2. The gamma sweep: time horizon of the decision-maker                 #
# ---------------------------------------------------------------------- #
# Low gamma  = myopic (values immediate reward, e.g. this quarter's margin).
# High gamma = patient (values long-run outcomes, e.g. brand equity).
GAMMA_VALUES = [0.0, 0.30, 0.618, 0.85, 0.95, 0.99]


# ---------------------------------------------------------------------- #
def _summarise(out_dir: str) -> dict:
    """Read the test log and action distribution train() just wrote."""
    te = pd.read_csv(os.path.join(out_dir, "test_log.csv"))
    ad = pd.read_csv(os.path.join(out_dir, "action_distribution.csv"))
    share = dict(zip(ad["action"], ad["share"]))
    return {
        "reward": te["reward"].mean(),
        "premium": te["mean_premium"].mean(),
        "premium_std": te["mean_premium"].std(),
        "peak_premium": te["peak_premium"].mean(),
        "access": te["mean_access"].mean(),
        "final_msrp": te["final_msrp"].mean(),
        "final_capacity": te["final_capacity"].mean()
                          if "final_capacity" in te else np.nan,
        "final_interval": te["final_interval"].mean()
                          if "final_interval" in te else np.nan,
        # a few salient action shares, for the "what did it learn" story
        **{f"act_{a}": share.get(a, 0.0) for a in
           ("cadence_slower", "cadence_faster", "capacity_up", "capacity_down",
            "appeal_up", "appeal_down", "restock_more", "restock_less",
            "msrp_up", "msrp_down", "no_op")},
    }


def run_reward_scenarios(base_cfg, base_dqn, episodes, test_episodes,
                         seeds, out_root):
    rows = []
    for name, overrides in REWARD_SCENARIOS.items():
        for seed in seeds:
            cfg = base_cfg.with_(**overrides)
            out_dir = os.path.join(out_root, f"reward_{name}_seed{seed}")
            print(f"\n{'#'*60}\n# REWARD SCENARIO: {name}  (seed {seed})  {overrides}\n{'#'*60}")
            train(cfg, base_dqn, episodes, test_episodes, out_dir,
                  seed=seed, verbose=False)
            row = {"scenario": name, "seed": seed, **overrides,
                   **_summarise(out_dir)}
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "reward_scenarios_raw.csv"), index=False)
    # aggregate across seeds
    agg = df.groupby("scenario", as_index=False).agg(
        reward=("reward", "mean"), premium=("premium", "mean"),
        premium_sd=("premium", "std"), access=("access", "mean"),
        access_sd=("access", "std"), final_msrp=("final_msrp", "mean"),
        act_cadence_slower=("act_cadence_slower", "mean"),
        act_capacity_down=("act_capacity_down", "mean"),
        act_capacity_up=("act_capacity_up", "mean"),
        act_appeal_down=("act_appeal_down", "mean"),
        act_restock_more=("act_restock_more", "mean"),
        act_msrp_down=("act_msrp_down", "mean"),
    )
    agg.to_csv(os.path.join(out_root, "reward_scenarios_agg.csv"), index=False)
    print("\n=== REWARD SCENARIOS (aggregated) ===")
    print(agg.to_string(index=False))
    return agg


def run_gamma_sweep(base_cfg, base_dqn, episodes, test_episodes,
                    seeds, out_root):
    rows = []
    for gamma in GAMMA_VALUES:
        for seed in seeds:
            dqn = DQNConfig(**{**base_dqn.__dict__, "gamma": gamma})
            out_dir = os.path.join(out_root, f"gamma_{gamma}_seed{seed}")
            print(f"\n{'#'*60}\n# GAMMA = {gamma}  (seed {seed})\n{'#'*60}")
            train(base_cfg, dqn, episodes, test_episodes, out_dir,
                  seed=seed, verbose=False)
            rows.append({"gamma": gamma, "seed": seed, **_summarise(out_dir)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "gamma_sweep_raw.csv"), index=False)
    agg = df.groupby("gamma", as_index=False).agg(
        reward=("reward", "mean"), premium=("premium", "mean"),
        premium_sd=("premium", "std"), access=("access", "mean"),
        final_capacity=("final_capacity", "mean"),
        final_interval=("final_interval", "mean"),
        act_cadence_slower=("act_cadence_slower", "mean"),
        act_capacity_up=("act_capacity_up", "mean"),
        act_capacity_down=("act_capacity_down", "mean"),
        act_restock_more=("act_restock_more", "mean"),
    )
    agg.to_csv(os.path.join(out_root, "gamma_sweep_agg.csv"), index=False)
    print("\n=== GAMMA SWEEP (aggregated) ===")
    print(agg.to_string(index=False))
    return agg


# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Reward-weight and gamma sensitivity study")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--test-episodes", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of random seeds per condition (for error bars)")
    ap.add_argument("--agents", type=int, default=250)
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--out", type=str, default="results/reward_study")
    ap.add_argument("--only", choices=["reward", "gamma", "both"], default="both")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(42, 42 + args.seeds))

    base_cfg = SimConfig(n_agents=args.agents, years=args.years, sa_delay=2,
                         use_super_agent=True, seed=42)
    base_dqn = DQNConfig()

    # record exactly what was run, for reproducibility
    with open(os.path.join(args.out, "study_config.json"), "w") as fh:
        json.dump({"episodes": args.episodes, "test_episodes": args.test_episodes,
                   "seeds": seeds, "agents": args.agents, "years": args.years,
                   "reward_scenarios": REWARD_SCENARIOS,
                   "gamma_values": GAMMA_VALUES}, fh, indent=2)

    if args.only in ("reward", "both"):
        run_reward_scenarios(base_cfg, base_dqn, args.episodes,
                             args.test_episodes, seeds, args.out)
    if args.only in ("gamma", "both"):
        run_gamma_sweep(base_cfg, base_dqn, args.episodes,
                        args.test_episodes, seeds, args.out)

    print(f"\nDone. Results in {args.out}")


if __name__ == "__main__":
    main()

"""
gamma_optimal.py — Additional experiment for reviewer comment #5.

Retrains the DDQN super-agent at the discount factor the gamma-sweep flagged as
optimal (gamma = 0.85) and at the value used in the main paper (gamma = 0.618),
over several training seeds, then evaluates each greedy policy against the same
baselines used in Table 2. Produces:

    results/gamma_opt/summary.csv        # per (gamma, seed) DDQN + baseline means
    results/gamma_opt/gamma_compare.pdf  # 2-panel bar chart (premium, access)

The point is to answer, apples-to-apples with the rest of the paper: does moving
the deployed policy to gamma = 0.85 actually improve the premium/accessibility
balance of Table 2, or does the sweep's optimum not transfer to the full
baseline comparison?

USAGE (from the project root, the folder that contains config/, pokesim/, rl/):

    python -m experiments.gamma_optimal \
        --episodes 150 --test-episodes 40 --seeds 42 43 44 \
        --out results/gamma_opt

Notes
-----
* Uses SimConfig() defaults, i.e. the configuration reported in the paper
  (N = 300, horizon = 5 years, investor/flipper share 0.25, supply 0.50). If
  Table 2 in your submission was produced with a different config (e.g. the
  train.py CLI defaults of 250 agents / 8 years), pass the matching flags so the
  comparison is fair.
* Reuses rl.train.train() unchanged: for each (gamma, seed) it writes a full run
  folder, then we read back test_log.csv and baseline_log.csv and aggregate.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SimConfig
from rl.ddqn import DQNConfig
from rl.train import train


GAMMAS = [0.618, 0.85]          # main-paper value vs sweep optimum
BASELINES_TO_SHOW = ["no_op", "heuristic"]   # the informative reference points


def run(args) -> pd.DataFrame:
    rows = []
    for gamma in GAMMAS:
        for seed in args.seeds:
            out_dir = os.path.join(args.out, f"gamma_{gamma}_seed{seed}")
            cfg = SimConfig(
                n_agents=args.agents,
                years=args.years,
                sa_delay=args.sa_delay,
                investor_share=args.investor_share,
                fan_flipper_share=args.fan_flipper_share,
                supply_ratio=args.supply_ratio,
                network_type=args.network,
                seed=seed,
            )
            dqn_cfg = DQNConfig(gamma=gamma, double=True)

            print(f"\n### gamma={gamma}  seed={seed} ###")
            train(cfg, dqn_cfg, episodes=args.episodes,
                  test_episodes=args.test_episodes, out_dir=out_dir,
                  seed=seed, verbose=False)

            test = pd.read_csv(os.path.join(out_dir, "test_log.csv"))
            base = pd.read_csv(os.path.join(out_dir, "baseline_log.csv"))

            rows.append({
                "gamma": gamma, "seed": seed, "policy": f"DDQN(gamma={gamma})",
                "premium": test["mean_premium"].mean(),
                "access": test["mean_access"].mean(),
                "reward": test["reward"].mean(),
            })
            for name in BASELINES_TO_SHOW:
                b = base[base.policy == name]
                rows.append({
                    "gamma": gamma, "seed": seed, "policy": name,
                    "premium": b["mean_premium"].mean(),
                    "access": b["mean_access"].mean(),
                    "reward": b["reward"].mean(),
                })

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    return df


def plot(df: pd.DataFrame, out: str) -> None:
    # aggregate across seeds
    order = [f"DDQN(gamma={GAMMAS[0]})", f"DDQN(gamma={GAMMAS[1]})"] + BASELINES_TO_SHOW
    agg = (df.groupby("policy")[["premium", "access"]]
             .agg(["mean", "std"]).reindex(order))

    labels = ["DDQN\n$\\gamma$=0.618", "DDQN\n$\\gamma$=0.85", "no-op", "heuristic"]
    x = np.arange(len(order))
    greys = ["#222222", "#5a5a5a", "#9a9a9a", "#c9c9c9"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    for ax, metric, ylab, hline in [
        (ax1, "premium", "mean resale premium $V$", 1.0),
        (ax2, "access", "collector accessibility", None),
    ]:
        means = agg[(metric, "mean")].values
        stds = np.nan_to_num(agg[(metric, "std")].values)
        ax.bar(x, means, yerr=stds, capsize=4, color=greys,
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)
        if hline is not None:
            ax.axhline(hline, ls=":", lw=0.9, color="black")
            ax.text(len(x) - 0.5, hline * 1.02, "list price", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "gamma_compare.pdf"))
    fig.savefig(os.path.join(out, "gamma_compare.png"), dpi=130)
    print("wrote", os.path.join(out, "gamma_compare.pdf"))

    # console summary
    print("\n=== mean over seeds ===")
    for pol in order:
        m = df[df.policy == pol]
        print(f"{pol:>20}: premium={m['premium'].mean():.3f}"
              f"  access={m['access'].mean():.3f}"
              f"  reward={m['reward'].mean():.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--test-episodes", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str, default="results/gamma_opt")
    ap.add_argument("--agents", type=int, default=300)
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--sa-delay", type=int, default=2)
    ap.add_argument("--investor-share", type=float, default=0.25)
    ap.add_argument("--fan-flipper-share", type=float, default=0.25)
    ap.add_argument("--supply-ratio", type=float, default=0.50)
    ap.add_argument("--network", type=str, default="erdos_renyi")
    args = ap.parse_args()

    df = run(args)
    plot(df, args.out)


if __name__ == "__main__":
    main()

"""
train.py — Training e valutazione del super-agente.

Replica la struttura di `deepq_simulation.py`: N episodi di training con
epsilon-greedy decrescente, poi M episodi di test con policy greedy sulla rete
target. In piu' salva su CSV tutto cio' che serve per le figure e confronta
sistematicamente la policy appresa con tre baseline non apprese.

Uso:
    python -m rl.train --episodes 120 --test-episodes 30 --out results/rl
"""

from __future__ import annotations

import os
import sys

# Rende lo script eseguibile sia come modulo (python -m experiments.sweeps) sia
# come file (python experiments/sweeps.py) da qualsiasi cartella: aggiunge la
# root del progetto -- la cartella che contiene i package `config`, `pokesim`,
# `rl` -- a sys.path se non c'e' gia'. Senza questo, lanciare il file
# direttamente mette in path solo la sua cartella e `pokesim` non si trova.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from config import SimConfig
from pokesim.model import ACTIONS, N_ACTIONS
from rl.ddqn import DDQNAgent, DQNConfig, set_global_seed
from rl.env import TPCSandboxEnv


# ---------------------------------------------------------------------- #
# Policy di confronto (non apprese)                                      #
# ---------------------------------------------------------------------- #
def policy_noop(obs):
    """Nessun intervento: il mercato lasciato a se stesso."""
    return 0


def policy_random(obs):
    return np.random.randint(N_ACTIONS)


def policy_heuristic(obs):
    """
    Euristica esperta: la regola che un responsabile di prodotto applicherebbe
    a mano leggendo la stessa dashboard che vede il super-agente.

        invenduto alto o marca in erosione -> riduci la capacita'
        premium in bolla                   -> set meno appetibili
        premium elevato                    -> rallenta le uscite
        mercato calmo                      -> ristampa cio' che tira (margine)

    E' il confronto onesto per la policy appresa: se il DDQN non batte questa,
    il tier data-driven non sta aggiungendo nulla che non si sapesse gia'.
    """
    premium, hype, unsold, access, capture, inf_hype, brand, appeal = obs
    if unsold > 0.18 or brand < 0.85:
        return 2                       # capacity_down
    if premium > 0.35:
        return 10                      # appeal_down
    if premium > 0.20:
        return 6                       # cadence_slower
    return 7                           # restock_more


def policy_capacity(obs):
    """Espansione continua della capacita': la leva singola piu' forte."""
    return 1


BASELINES = {"no_op": policy_noop,
             "random": policy_random,
             "capacity_only": policy_capacity,
             "heuristic": policy_heuristic}


# ---------------------------------------------------------------------- #
def train(cfg: SimConfig, dqn_cfg: DQNConfig, episodes: int,
          test_episodes: int, out_dir: str, seed: int = 42, verbose: bool = True):

    os.makedirs(out_dir, exist_ok=True)
    set_global_seed(seed)

    env = TPCSandboxEnv(cfg)
    agent = DDQNAgent(state_dim=env.observation_space.shape[0],
                      n_actions=env.action_space.n, cfg=dqn_cfg)

    train_rows, t0 = [], time.time()

    # ------------------------- TRAINING ------------------------------- #
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total_r, n_steps = False, 0.0, 0
        premiums, accesses = [], []

        while not done:
            a = agent.act(obs)
            obs2, r, done, _, info = env.step(a)
            agent.remember(obs, a, r, obs2, float(done))
            agent.learn()
            obs = obs2
            total_r += r
            n_steps += 1
            premiums.append(info["premium"])
            accesses.append(info["access"])

        agent.decay_epsilon(ep)
        train_rows.append({
            "phase": "train", "episode": ep, "reward": total_r,
            "steps": n_steps, "epsilon": agent.epsilon,
            "final_premium": premiums[-1], "mean_premium": float(np.mean(premiums)),
            "mean_access": float(np.mean(accesses)),
            "final_msrp": info["msrp"],
            "final_capacity": info["capacity_multiplier"],
            "final_interval": info["release_interval"],
        })
        if verbose and (ep % 10 == 0 or ep == episodes - 1):
            print(f"[train] ep {ep:3d}  R={total_r:8.1f}  eps={agent.epsilon:.3f}  "
                  f"premium={premiums[-1]:.2f}  access={accesses[-1]:.2f}")

    train_time = (time.time() - t0) / 60
    agent.save(os.path.join(out_dir, "ddqn_super_agent.pt"))

    # --------------------------- TEST --------------------------------- #
    test_rows, action_counts = [], np.zeros(N_ACTIONS, dtype=int)
    best_df = None
    t0 = time.time()

    for ep in range(test_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done, total_r = False, 0.0
        premiums, accesses = [], []
        while not done:
            a = agent.act(obs, greedy=True)
            action_counts[a] += 1
            obs, r, done, _, info = env.step(a)
            total_r += r
            premiums.append(info["premium"])
            accesses.append(info["access"])

        test_rows.append({
            "phase": "test", "episode": ep, "reward": total_r,
            "final_premium": premiums[-1], "mean_premium": float(np.mean(premiums)),
            "peak_premium": float(np.max(premiums)),
            "mean_access": float(np.mean(accesses)),
            "final_msrp": info["msrp"],
            "final_capacity": info["capacity_multiplier"],
            "final_interval": info["release_interval"],
        })
        if ep == 0:
            best_df = env.model.to_dataframe()

    test_time = (time.time() - t0) / 60

    # ----------------------- BASELINE --------------------------------- #
    baseline_rows = []
    for name, fn in BASELINES.items():
        for ep in range(test_episodes):
            df, total_r, acts = env.rollout(fn, seed=10_000 + ep)
            baseline_rows.append({
                "policy": name, "episode": ep, "reward": total_r,
                "final_premium": df["premium_index"].iloc[-1],
                "mean_premium": df["premium_index"].mean(),
                "peak_premium": df["premium_index"].max(),
                "mean_access": df["access_latest"].mean(),
                "final_msrp": df["msrp"].iloc[-1],
                "gini": df["gini_holdings"].iloc[-1],
            })
            if name == "no_op" and ep == 0:
                df.to_csv(os.path.join(out_dir, "trace_no_superagent.csv"), index=False)

    # ------------------------ SALVATAGGIO ----------------------------- #
    pd.DataFrame(train_rows).to_csv(os.path.join(out_dir, "train_log.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(out_dir, "test_log.csv"), index=False)
    pd.DataFrame(baseline_rows).to_csv(os.path.join(out_dir, "baseline_log.csv"), index=False)
    if best_df is not None:
        best_df.to_csv(os.path.join(out_dir, "trace_with_superagent.csv"), index=False)

    pd.DataFrame({"action": [ACTIONS[i] for i in range(N_ACTIONS)],
                  "count": action_counts,
                  "share": action_counts / max(1, action_counts.sum())}
                 ).to_csv(os.path.join(out_dir, "action_distribution.csv"), index=False)

    cfg.to_json(os.path.join(out_dir, "sim_config.json"))
    with open(os.path.join(out_dir, "dqn_config.json"), "w") as fh:
        json.dump(dqn_cfg.__dict__, fh, indent=2)

    rl_mean = np.mean([r["mean_premium"] for r in test_rows])
    print("\n" + "=" * 62)
    print(f"Training: {train_time:.1f} min | Test: {test_time:.1f} min")
    print(f"Premium medio - DDQN: {rl_mean:.3f}")
    bdf = pd.DataFrame(baseline_rows)
    for name in BASELINES:
        m = bdf.loc[bdf.policy == name, "mean_premium"].mean()
        print(f"Premium medio - {name:>9}: {m:.3f}")
    print(f"Risultati in: {out_dir}")
    print("=" * 62)

    return agent, out_dir


# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Training del super-agente DDQN")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--test-episodes", type=int, default=30)
    ap.add_argument("--out", type=str, default="results/rl")
    ap.add_argument("--agents", type=int, default=250)
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--sa-delay", type=int, default=2)
    ap.add_argument("--investor-share", type=float, default=0.25)
    ap.add_argument("--fan-flipper-share", type=float, default=0.25)
    ap.add_argument("--supply-ratio", type=float, default=0.50)
    ap.add_argument("--network", type=str, default="erdos_renyi")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-double", action="store_true",
                    help="usa il target DQN del paper originale")
    args = ap.parse_args()

    cfg = SimConfig(n_agents=args.agents, years=args.years, sa_delay=args.sa_delay,
                    investor_share=args.investor_share,
                    fan_flipper_share=args.fan_flipper_share,
                    supply_ratio=args.supply_ratio,
                    network_type=args.network, use_super_agent=True, seed=args.seed)
    dqn_cfg = DQNConfig(double=not args.no_double)

    train(cfg, dqn_cfg, args.episodes, args.test_episodes, args.out, seed=args.seed)


if __name__ == "__main__":
    main()

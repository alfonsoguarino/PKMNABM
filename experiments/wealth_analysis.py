"""
wealth_analysis.py — Who gets richer? Per-agent wealth with vs. without the
super-agent.

Runs the market twice under the same seeds:
    * "no-SA"   : policy_noop (no intervention at all);
    * "with-SA" : the trained DDQN super-agent (greedy), if a checkpoint is
                  given via --sa-checkpoint; otherwise the hand `heuristic`
                  baseline is used as an active-manager stand-in (so the script
                  also runs without torch).

For every agent it records, at the end of the run:
    budget, inventory marked-to-market, total wealth = budget + inventory,
    net secondary P&L = secondary revenue - secondary spend,
    units held / sold, agent type, echo-chamber membership, network degree.

Outputs (in --out):
    agents_wealth.csv
    wealth_by_type.pdf        box of final wealth by type, no-SA vs with-SA
    profit_by_type.pdf        box of net secondary P&L by type
    lorenz_wealth.pdf         Lorenz curve + Gini, no-SA vs with-SA
    degree_vs_wealth.pdf      scatter degree vs wealth by type (two panels)
    network_wealth.pdf        the graph, nodes coloured by wealth (two panels)

USAGE (from the project root):
    python -m experiments.wealth_analysis \
        --seeds 42 43 44 \
        --sa-checkpoint results/gamma_opt/gamma_0.618_seed42/ddqn_super_agent.pt \
        --out results/wealth
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
from rl.env import TPCSandboxEnv
from rl.train import policy_noop, policy_heuristic

TYPE_ORDER = ["collector", "fan_flipper", "investor"]
TYPE_COLOR = {"collector": "#4c78a8", "fan_flipper": "#f58518", "investor": "#54a24b"}


# --------------------------------------------------------------------------- #
def agent_type(a) -> str:
    if not a.is_fan:
        return "investor"
    return "collector" if a.resell_propensity == 0.0 else "fan_flipper"


def extract_agents(model, scenario: str, seed: int) -> list[dict]:
    pid2prod = {p.pid: p for p in model.products}
    rows = []
    for a in model.agents:
        inv_value = 0.0
        for pid, lots in a.inventory.items():
            prod = pid2prod.get(pid)
            if prod is not None and lots:
                inv_value += len(lots) * model.market_price_of(prod)
        wealth = a.budget + inv_value
        rows.append(dict(
            scenario=scenario, seed=seed, node=a.node_id,
            type=agent_type(a),
            in_echo=bool(a.in_echo_chamber),
            degree=model.graph.degree(a.node_id),
            income=a.income,
            budget=a.budget,
            units_held=a.n_units,
            inv_value=inv_value,
            wealth=wealth,
            net_secondary=a.revenue_secondary - a.spent_secondary,
            units_sold=a.units_sold,
        ))
    return rows


def sa_policy(checkpoint: str, env: TPCSandboxEnv):
    """Return a greedy policy_fn from a saved DDQN checkpoint, or None."""
    if not checkpoint:
        return None
    from rl.ddqn import DDQNAgent, DQNConfig
    state_dim = env.observation_space.shape[0]
    agent = DDQNAgent(state_dim=state_dim, n_actions=env.action_space.n,
                      cfg=DQNConfig())
    agent.load(checkpoint)
    return lambda obs: agent.act(obs, greedy=True)


# --------------------------------------------------------------------------- #
def run(args) -> pd.DataFrame:
    cfg_kwargs = dict(n_agents=args.agents, years=args.years, sa_delay=args.sa_delay,
                      investor_share=args.investor_share,
                      fan_flipper_share=args.fan_flipper_share,
                      supply_ratio=args.supply_ratio, network_type=args.network)

    records = []
    for seed in args.seeds:
        # ---- with-SA (or heuristic proxy) --------------------------------- #
        env = TPCSandboxEnv(SimConfig(seed=seed, **cfg_kwargs))
        pol = sa_policy(args.sa_checkpoint, env)
        if pol is None:
            print("[!] no --sa-checkpoint: using `heuristic` as active-manager proxy")
            pol = policy_heuristic
        env.rollout(pol, seed=seed)
        records += extract_agents(env.model, "with-SA", seed)

        # ---- no-SA -------------------------------------------------------- #
        env = TPCSandboxEnv(SimConfig(seed=seed, **cfg_kwargs))
        env.rollout(policy_noop, seed=seed)
        records += extract_agents(env.model, "no-SA", seed)

    df = pd.DataFrame(records)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "agents_wealth.csv"), index=False)
    return df


# --------------------------------------------------------------------------- #
def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float) - min(0.0, np.min(x)))  # shift if negatives
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return (2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n


def _box_by_type(df, value, ylabel, path):
    scen = ["no-SA", "with-SA"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.36
    for i, sc in enumerate(scen):
        for j, t in enumerate(TYPE_ORDER):
            vals = df[(df.scenario == sc) & (df.type == t)][value].values
            pos = j + (i - 0.5) * width
            bp = ax.boxplot(vals, positions=[pos], widths=width * 0.9,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black"))
            face = TYPE_COLOR[t]
            bp["boxes"][0].set_facecolor(face)
            bp["boxes"][0].set_alpha(0.55 if sc == "no-SA" else 1.0)
            bp["boxes"][0].set_hatch("" if sc == "with-SA" else "///")
    ax.set_xticks(range(len(TYPE_ORDER)), TYPE_ORDER)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", lw=0.6)
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="grey", hatch="///", alpha=0.55, label="no-SA"),
                       Patch(facecolor="grey", label="with-SA")],
              frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _lorenz(df, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for sc, ls in [("no-SA", "--"), ("with-SA", "-")]:
        w = np.sort(df[df.scenario == sc]["wealth"].values)
        w = w - min(0.0, w.min())
        cum = np.cumsum(w) / w.sum()
        cum = np.insert(cum, 0, 0)
        p = np.linspace(0, 1, len(cum))
        ax.plot(p, cum, ls, color="black",
                label=f"{sc} (Gini={gini(df[df.scenario==sc]['wealth'].values):.2f})")
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set_xlabel("cumulative share of agents (poorest → richest)")
    ax.set_ylabel("cumulative share of wealth")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _degree_vs_wealth(df, path):
    scen = ["no-SA", "with-SA"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, sc in zip(axes, scen):
        d = df[df.scenario == sc]
        for t in TYPE_ORDER:
            dt = d[d.type == t]
            ax.scatter(dt.degree, dt.wealth, s=14, alpha=0.6,
                       color=TYPE_COLOR[t], label=t, edgecolor="none")
        ax.set_title(sc); ax.set_xlabel("network degree"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("final wealth (€)")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _network_wealth(df, model_by_scen, path):
    try:
        import networkx as nx
    except Exception:
        print("[!] networkx not available: skipping network_wealth.pdf")
        return
    scen = ["no-SA", "with-SA"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    G = model_by_scen["no-SA"].graph
    pos = nx.spring_layout(G, seed=1, k=1.5 / np.sqrt(G.number_of_nodes()))

    allw = df.groupby(["scenario", "node"])["wealth"].mean()
    vmin = float(allw.min())
    vmax = float(np.percentile(allw.values, 97))
    if vmax <= vmin:
        vmax = vmin + 1.0

    sc_nodes = None
    for ax, sc in zip(axes, scen):
        m = model_by_scen[sc]
        wnode = df[df.scenario == sc].groupby("node")["wealth"].mean()   # aggregate over seeds
        colors = np.array([float(wnode.get(n, 0.0)) for n in m.graph.nodes], dtype=float)
        nx.draw_networkx_edges(m.graph, pos, ax=ax, alpha=0.08)
        sc_nodes = nx.draw_networkx_nodes(
            m.graph, pos, ax=ax, node_size=22, node_color=colors,
            cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(sc)
        ax.axis("off")
    fig.colorbar(sc_nodes, ax=axes, shrink=0.7, label="final wealth (€)")
    fig.savefig(path)
    plt.close(fig)


def plots(df, model_by_scen, out):
    _box_by_type(df, "wealth", "final wealth (€)",
                 os.path.join(out, "wealth_by_type.pdf"))
    _box_by_type(df, "net_secondary", "net secondary P&L (€)",
                 os.path.join(out, "profit_by_type.pdf"))
    _lorenz(df, os.path.join(out, "lorenz_wealth.pdf"))
    _degree_vs_wealth(df, os.path.join(out, "degree_vs_wealth.pdf"))
    if model_by_scen is not None:
        _network_wealth(df, model_by_scen, os.path.join(out, "network_wealth.pdf"))

    # console summary
    print("\n=== median wealth / net P&L by type and scenario ===")
    g = df.groupby(["scenario", "type"])[["wealth", "net_secondary"]].median()
    print(g.round(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--sa-checkpoint", type=str, default="")
    ap.add_argument("--out", type=str, default="results/wealth")
    ap.add_argument("--agents", type=int, default=300)
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--sa-delay", type=int, default=2)
    ap.add_argument("--investor-share", type=float, default=0.25)
    ap.add_argument("--fan-flipper-share", type=float, default=0.25)
    ap.add_argument("--supply-ratio", type=float, default=0.50)
    ap.add_argument("--network", type=str, default="erdos_renyi")
    ap.add_argument("--network-plot", action="store_true",
                    help="also draw the node-coloured network (uses last seed)")
    args = ap.parse_args()

    df = run(args)

    model_by_scen = None
    if args.network_plot:
        seed = args.seeds[-1]
        mbs = {}
        env = TPCSandboxEnv(SimConfig(seed=seed, n_agents=args.agents, years=args.years,
                                  sa_delay=args.sa_delay,
                                  investor_share=args.investor_share,
                                  fan_flipper_share=args.fan_flipper_share,
                                  supply_ratio=args.supply_ratio,
                                  network_type=args.network))
        pol = sa_policy(args.sa_checkpoint, env) or policy_heuristic
        env.rollout(pol, seed=seed); mbs["with-SA"] = env.model
        env = TPCSandboxEnv(SimConfig(seed=seed, n_agents=args.agents, years=args.years,
                                  sa_delay=args.sa_delay,
                                  investor_share=args.investor_share,
                                  fan_flipper_share=args.fan_flipper_share,
                                  supply_ratio=args.supply_ratio,
                                  network_type=args.network))
        env.rollout(policy_noop, seed=seed); mbs["no-SA"] = env.model
        model_by_scen = mbs
        # restrict df to that seed for the network picture
        
        plots(df, mbs, args.out)
        return

    plots(df, model_by_scen, args.out)


if __name__ == "__main__":
    main()

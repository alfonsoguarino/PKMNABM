"""
plots.py — Generazione di tutte le figure.

Stile deliberatamente allineato a quello dei due paper di riferimento: linee
con marker distinti, scala di grigi come default (le riviste stampano in
bianco e nero), griglia leggera, legenda fuori dagli assi. Ogni figura viene
salvata sia in PNG a 300 dpi sia in PDF vettoriale.

Uso:
    python -m experiments.plots --results results --out figures
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
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------- #
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (9, 5),
    "axes.grid": True,
    "grid.linewidth": 0.25,
    "grid.alpha": 0.6,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

MARKERS = ["o-", "s--", "D:", "^-", "v--", "P-.", "X--", "*:", "h-"]
GREYS = ["black", "dimgrey", "grey", "darkgrey", "silver",
         "darkslategrey", "lightslategrey", "slategrey", "gainsboro"]


def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {name}.png / .pdf")


def _legend_right(ax, title=None):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.82, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title=title,
              frameon=False)


def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _has_cols(agg, cols, name):
    """
    Verifica che il DataFrame contenga le colonne richieste dalla figura.

    Se ne manca una, salta la figura con un avviso invece di far crashare
    l'intera pipeline: cosi' un solo esperimento non ancora lanciato, o un
    CSV con uno schema diverso, non impedisce di generare tutte le altre
    figure. Restituisce True se si puo' procedere.
    """
    if agg is None:
        return False
    missing = [c for c in cols if c not in agg.columns]
    if missing:
        print(f"  [skip {name}] missing columns in CSV: {', '.join(missing)}")
        print(f"                available: {', '.join(agg.columns)}")
        return False
    return True


# ====================================================================== #
# FIG. 1-2 : Bubble Frequency vs polarizzazione                          #
# ====================================================================== #
def fig_bubble_vs_polarization(agg: pd.DataFrame, out_dir: str,
                               series_col: str, name: str,
                               series_label: str, ylab: str = None):
    ylab = ylab or "B (bubble frequency)"
    if not _has_cols(agg, [series_col, "network_polarization", "bubble_frequency"], name):
        return
    fig, ax = plt.subplots()
    for i, (val, sub) in enumerate(agg.groupby(series_col)):
        sub = sub.sort_values("network_polarization")
        ax.plot(sub["network_polarization"], sub["bubble_frequency"],
                MARKERS[i % len(MARKERS)], color=GREYS[i % len(GREYS)],
                label=f"{val}")
    ax.axhline(0.5, ls=":", lw=0.8, color="black")
    ax.set_xlabel("$P_n$ (network polarization)")
    ax.set_ylabel(ylab)
    ax.set_ylim(-0.02, 1.02)
    _legend_right(ax, series_label)
    _save(fig, out_dir, name)


def fig_premium_vs_polarization(agg: pd.DataFrame, out_dir: str,
                                series_col: str, name: str, series_label: str):
    if not _has_cols(agg, [series_col, "network_polarization", "premium_mean"], name):
        return
    fig, ax = plt.subplots()
    for i, (val, sub) in enumerate(agg.groupby(series_col)):
        sub = sub.sort_values("network_polarization")
        ax.errorbar(sub["network_polarization"], sub["premium_mean"],
                    yerr=sub["premium_mean_std"], fmt=MARKERS[i % len(MARKERS)],
                    color=GREYS[i % len(GREYS)], capsize=2.5, lw=1.2,
                    elinewidth=0.7, label=f"{val}")
    ax.axhline(1.0, ls="-", lw=0.8, color="black", alpha=0.4)
    ax.axhline(2.0, ls=":", lw=0.9, color="black")
    ax.set_xlabel("$P_n$ (network polarization)")
    ax.set_ylabel("Mean resale premium (secondary price / MSRP)")
    _legend_right(ax, series_label)
    _save(fig, out_dir, name)


# ====================================================================== #
# FIG. 3 : Heatmap offerta x quota di reseller                           #
# ====================================================================== #
def fig_heatmap(agg: pd.DataFrame, out_dir: str, x: str, y: str, z: str,
                name: str, xlabel: str, ylabel: str, zlabel: str,
                fmt: str = "{:.2f}"):
    if not _has_cols(agg, [x, y, z], name):
        return
    piv = agg.pivot_table(index=y, columns=x, values=z)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(piv.values, cmap="Greys", aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(r) for r in piv.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                lo, hi = np.nanmin(piv.values), np.nanmax(piv.values)
                rel = (v - lo) / (hi - lo + 1e-12)
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=9, color="white" if rel > 0.55 else "black")
    fig.colorbar(im, ax=ax, label=zlabel, shrink=0.85)
    _save(fig, out_dir, name)


# ====================================================================== #
# FIG. 4 : Curve a barre per leve discrete                               #
# ====================================================================== #
def fig_bars(agg: pd.DataFrame, out_dir: str, x: str, series: str, z: str,
             name: str, xlabel: str, ylabel: str, series_label: str):
    if not _has_cols(agg, [x, series, z], name):
        return
    piv = agg.pivot_table(index=x, columns=series, values=z)
    fig, ax = plt.subplots()
    n = len(piv.columns)
    width = 0.8 / n
    idx = np.arange(len(piv.index))
    hatches = ["", "///", "...", "xxx", "\\\\\\", "+++"]
    for i, col in enumerate(piv.columns):
        ax.bar(idx + i * width - 0.4 + width / 2, piv[col].values, width,
               label=str(col), color=GREYS[i % len(GREYS)],
               edgecolor="black", linewidth=0.5, hatch=hatches[i % len(hatches)])
    ax.set_xticks(idx, [str(v) for v in piv.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _legend_right(ax, series_label)
    _save(fig, out_dir, name)


# ====================================================================== #
# FIG. 5 : Traiettorie temporali con e senza super-agente                #
# ====================================================================== #
def fig_trajectories(with_sa: pd.DataFrame | None, without_sa: pd.DataFrame | None,
                     out_dir: str, name: str = "fig_trajectory_premium"):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    panels = [("premium_index", "Resale premium (price / MSRP)", 2.0),
              ("access_recent", "Accessibility (collectors served)", None),
              ("margin_rate", "Industrial margin (normalized)", None),
              ("gini_holdings", "Gini of holdings", None)]

    for ax, (col, ylab, hline) in zip(axes, panels):
        if without_sa is not None and col in without_sa:
            ax.plot(without_sa["tick"], without_sa[col], "-", color="black",
                    lw=1.4, label="without super-agent")
        if with_sa is not None and col in with_sa:
            ax.plot(with_sa["tick"], with_sa[col], "--", color="dimgrey",
                    lw=1.4, label="with super-agent (DDQN)")
        if hline:
            ax.axhline(hline, ls=":", lw=0.9, color="black")
            ax.text(1, hline * 1.02, r"$\theta_B$", fontsize=10)
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=10)

    axes[-1].set_xlabel("tick (weeks)")
    fig.suptitle("Market dynamics over time", y=0.995)
    _save(fig, out_dir, name)


# ====================================================================== #
# FIG. 6 : Curve di apprendimento                                        #
# ====================================================================== #
def fig_learning(train_log: pd.DataFrame, test_log: pd.DataFrame | None,
                 out_dir: str, name: str = "fig_learning_curve"):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=False)

    r = train_log["reward"].to_numpy()
    w = max(1, len(r) // 15)
    smooth = pd.Series(r).rolling(w, min_periods=1).mean()
    axes[0].plot(train_log["episode"], r, color="silver", lw=0.9,
                 label="reward per episode")
    axes[0].plot(train_log["episode"], smooth, color="black", lw=1.8,
                 label=f"moving average ({w} ep.)")
    axes[0].set_ylabel("cumulative reward")
    axes[0].set_xlabel("training episode")
    axes[0].legend(frameon=False, fontsize=10)

    ax2 = axes[0].twinx()
    ax2.plot(train_log["episode"], train_log["epsilon"], ls=":", color="dimgrey")
    ax2.set_ylabel(r"$\varepsilon$", color="dimgrey")
    ax2.grid(False)

    axes[1].plot(train_log["episode"], train_log["mean_premium"], "-",
                 color="black", lw=1.3, label="mean premium (training)")
    if test_log is not None:
        axes[1].axhline(test_log["mean_premium"].mean(), ls="--",
                        color="dimgrey", label="mean premium (test, greedy)")
    axes[1].axhline(2.0, ls=":", color="black", lw=0.9)
    axes[1].set_ylabel("mean premium")
    axes[1].set_xlabel("episode")
    axes[1].legend(frameon=False, fontsize=10)

    _save(fig, out_dir, name)


# ====================================================================== #
# FIG. 7 : Distribuzione delle azioni apprese                            #
# ====================================================================== #
def fig_action_distribution(dist: pd.DataFrame, out_dir: str,
                            name: str = "fig_action_distribution"):
    dist = dist.sort_values("share", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(dist["action"], dist["share"], color="dimgrey",
            edgecolor="black", linewidth=0.6)
    ax.set_xlabel("relative frequency in greedy policy")
    ax.set_ylabel("")
    for y, v in zip(range(len(dist)), dist["share"]):
        ax.text(v + 0.005, y, f"{v:.1%}", va="center", fontsize=10)
    ax.set_xlim(0, max(0.05, dist["share"].max() * 1.25))
    _save(fig, out_dir, name)


# ====================================================================== #
# FIG. 8 : Confronto policy                                              #
# ====================================================================== #
def fig_policy_comparison(test_log: pd.DataFrame, baseline_log: pd.DataFrame,
                          out_dir: str, name: str = "fig_policy_comparison"):
    rows = [{"policy": "DDQN",
             "premium": test_log["mean_premium"].mean(),
             "premium_std": test_log["mean_premium"].std(),
             "access": test_log["mean_access"].mean(),
             "reward": test_log["reward"].mean()}]
    for p, sub in baseline_log.groupby("policy"):
        rows.append({"policy": p,
                     "premium": sub["mean_premium"].mean(),
                     "premium_std": sub["mean_premium"].std(),
                     "access": sub["mean_access"].mean(),
                     "reward": sub["reward"].mean()})
    df = pd.DataFrame(rows).sort_values("premium")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(df["policy"], df["premium"], yerr=df["premium_std"],
                color="dimgrey", edgecolor="black", capsize=4, linewidth=0.6)
    axes[0].axhline(2.0, ls=":", color="black", lw=0.9)
    axes[0].set_ylabel("mean resale premium (price / MSRP)")
    axes[0].set_title("Price containment")

    axes[1].bar(df["policy"], df["access"], color="darkgrey",
                edgecolor="black", linewidth=0.6)
    axes[1].set_ylabel("mean accessibility")
    axes[1].set_title("Collectors served")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    _save(fig, out_dir, name)
    df.to_csv(os.path.join(out_dir, "policy_comparison.csv"), index=False)


# ====================================================================== #
# FIG. 9 : Frontiera prezzo / accessibilita'                             #
# ====================================================================== #
def fig_pareto(agg: pd.DataFrame, out_dir: str, color_col: str,
               name: str = "fig_pareto_price_access"):
    if not _has_cols(agg, [color_col, "premium_mean", "access"], name):
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    vals = sorted(agg[color_col].unique())
    for i, v in enumerate(vals):
        sub = agg[agg[color_col] == v].sort_values("premium_mean")
        ax.plot(sub["premium_mean"], sub["access"], MARKERS[i % len(MARKERS)],
                color=GREYS[i % len(GREYS)], label=f"{v}", lw=1.2)
    ax.axvline(2.0, ls=":", color="black", lw=0.9)
    ax.set_xlabel("mean resale premium (price / MSRP)  -  lower is better")
    ax.set_ylabel("accessibility  -  higher is better")
    _legend_right(ax, color_col)
    _save(fig, out_dir, name)


# ====================================================================== #
# FIG. reward-study : sensitivity to reward weights and to gamma          #
# ====================================================================== #
def fig_reward_scenarios(agg: pd.DataFrame, out_dir: str,
                         name: str = "fig24_reward_scenarios"):
    """Premium and accessibility achieved under each regulatory personality."""
    if not _has_cols(agg, ["scenario", "premium", "access"], name):
        return
    order = ["profit_seeker", "baseline", "balanced_soft",
             "access_first", "anti_bubble", "price_hawk"]
    agg = agg.set_index("scenario").reindex(
        [o for o in order if o in agg["scenario"].values]
        if "scenario" in agg.columns else agg.index)
    agg = agg.reset_index()
    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.38
    ax.bar(x - w/2, agg["premium"], w, label="resale premium",
           color="dimgrey", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, agg["access"], w, label="accessibility",
           color="silver", edgecolor="black", linewidth=0.5, hatch="///")
    ax.axhline(1.0, ls=":", lw=0.9, color="black")
    ax.set_xticks(x, agg["scenario"], rotation=20, ha="right")
    ax.set_ylabel("value")
    ax.set_xlabel("reward scenario (regulatory personality)")
    ax.legend(frameon=False)
    _save(fig, out_dir, name)


def fig_gamma_sweep(agg: pd.DataFrame, out_dir: str,
                    name: str = "fig25_gamma_sweep"):
    """How the learned policy shifts as the decision-maker's horizon lengthens."""
    if not _has_cols(agg, ["gamma", "premium", "access"], name):
        return
    agg = agg.sort_values("gamma")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(agg["gamma"], agg["premium"], "o-", color="black",
            label="resale premium")
    ax.plot(agg["gamma"], agg["access"], "s--", color="dimgrey",
            label="accessibility")
    ax.axhline(1.0, ls=":", lw=0.9, color="black")
    ax.set_xlabel(r"discount factor $\gamma$ (myopic $\rightarrow$ patient)")
    ax.set_ylabel("value")
    ax.legend(frameon=False)
    _save(fig, out_dir, name)


# ====================================================================== #
# Orchestratore                                                          #
# ====================================================================== #
def build_all(results_dir: str, out_dir: str) -> None:
    sw = os.path.join(results_dir, "sweeps")
    rl = os.path.join(results_dir, "rl")
    os.makedirs(out_dir, exist_ok=True)
    print("Generating figures...")

    # --- sweep sulla polarizzazione ---
    a = _read(os.path.join(sw, "polarization_agg.csv"))
    if a is not None:
        fig_bubble_vs_polarization(a, out_dir, "activation_threshold",
                                   "fig01_bubble_vs_Pn_by_theta", r"$\theta$")
        fig_premium_vs_polarization(a, out_dir, "activation_threshold",
                                    "fig02_premium_vs_Pn_by_theta", r"$\theta$")

    a = _read(os.path.join(sw, "opinion_agg.csv"))
    if a is not None:
        fig_bubble_vs_polarization(a, out_dir, "opinion_polarization",
                                   "fig03_bubble_vs_Pn_by_Po", r"$P_o$")

    # --- offerta / reseller ---
    a = _read(os.path.join(sw, "supply_agg.csv"))
    if a is not None:
        fig_heatmap(a, out_dir, "supply_ratio", "investor_share",
                    "premium_mean", "fig04_heatmap_supply_investor_premium",
                    "supply / population", "investor share", "mean premium")
        fig_heatmap(a, out_dir, "supply_ratio", "investor_share",
                    "access", "fig05_heatmap_supply_investor_access",
                    "supply / population", "investor share", "accessibility")
        fig_pareto(a, out_dir, "investor_share", "fig06_pareto_price_access")

    # --- MSRP ---
    a = _read(os.path.join(sw, "msrp_agg.csv"))
    if a is not None:
        fig_bars(a, out_dir, "msrp", "investor_share", "premium_mean",
                 "fig07_msrp_premium", "MSRP (EUR)", "mean premium",
                 "investor share")
        fig_bars(a, out_dir, "msrp", "investor_share", "access",
                 "fig08_msrp_access", "MSRP (EUR)", "accessibility",
                 "investor share")

    # --- cadenza ---
    a = _read(os.path.join(sw, "cadence_agg.csv"))
    if a is not None:
        fig_bars(a, out_dir, "releases_per_year", "supply_ratio", "premium_mean",
                 "fig09_cadence_premium", "releases per year", "mean premium",
                 "supply")
        fig_bars(a, out_dir, "releases_per_year", "supply_ratio", "access",
                 "fig10_cadence_access", "releases per year", "accessibility",
                 "supply")

    # --- topologia ---
    a = _read(os.path.join(sw, "topology_agg.csv"))
    if a is not None:
        fig_bubble_vs_polarization(a, out_dir, "network_type",
                                   "fig11_bubble_by_topology", "topology")

    # --- dimensione della rete ---
    a = _read(os.path.join(sw, "size_agg.csv"))
    if a is not None:
        fig_bars(a, out_dir, "n_agents", "network_polarization", "premium_mean",
                 "fig12_size_premium", "number of agents", "mean premium", "$P_n$")

    # --- appetibilita': la leva di raffreddamento ---
    a = _read(os.path.join(sw, "appeal_agg.csv"))
    if a is not None:
        fig_bars(a, out_dir, "appeal_target", "investor_share", "premium_mean",
                 "fig14_appeal_premium", "set appeal", "mean premium",
                 "investor share")
        fig_bars(a, out_dir, "appeal_target", "investor_share", "margin",
                 "fig15_appeal_margin", "set appeal", "margin",
                 "investor share")

    # --- apertura delle box ---
    a = _read(os.path.join(sw, "opening_agg.csv"))
    if a is not None:
        fig_bars(a, out_dir, "open_rate_bounds", "investor_share", "premium_mean",
                 "fig16_opening_premium", "share of product opened",
                 "mean premium", "investor share")

    # --- composizione della popolazione ---
    a = _read(os.path.join(sw, "population_agg.csv"))
    if a is not None:
        fig_heatmap(a, out_dir, "investor_share", "fan_flipper_share",
                    "premium_mean", "fig17_population_premium",
                    "pure investors", "fans who resell", "mean premium")

    # --- finestra di stampa ---
    a = _read(os.path.join(sw, "print_window_agg.csv"))
    if a is not None:
        fig_bars(a, out_dir, "print_window_ticks", "restock_share", "premium_mean",
                 "fig18_print_window", "print window (ticks)",
                 "mean premium", "restock share")

    # --- influencer ---
    a = _read(os.path.join(sw, "influencer_agg.csv"))
    if a is not None:
        fig_bars(a, out_dir, "influencer_power", "investor_share", "premium_mean",
                 "fig19_influencer_premium", "influencer power",
                 "mean premium", "investor share")

    # --- super-agente ---
    tr = _read(os.path.join(rl, "train_log.csv"))
    te = _read(os.path.join(rl, "test_log.csv"))
    bl = _read(os.path.join(rl, "baseline_log.csv"))
    if tr is not None:
        fig_learning(tr, te, out_dir, "fig20_learning_curve")
    if te is not None and bl is not None:
        fig_policy_comparison(te, bl, out_dir, "fig21_policy_comparison")

    d = _read(os.path.join(rl, "action_distribution.csv"))
    if d is not None:
        fig_action_distribution(d, out_dir, "fig22_action_distribution")

    w = _read(os.path.join(rl, "trace_with_superagent.csv"))
    wo = _read(os.path.join(rl, "trace_no_superagent.csv"))
    if w is not None or wo is not None:
        fig_trajectories(w, wo, out_dir, "fig23_trajectories")

    # --- reward / gamma sensitivity study ---
    rs = _read(os.path.join(results_dir, "reward_study", "reward_scenarios_agg.csv"))
    if rs is not None:
        fig_reward_scenarios(rs, out_dir, "fig24_reward_scenarios")
    gs = _read(os.path.join(results_dir, "reward_study", "gamma_sweep_agg.csv"))
    if gs is not None:
        fig_gamma_sweep(gs, out_dir, "fig25_gamma_sweep")

    print(f"Figures saved to: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Generate all figures from sweep and RL result CSVs")
    ap.add_argument("--results", type=str, default="results")
    ap.add_argument("--out", type=str, default="figures")
    args = ap.parse_args()
    build_all(args.results, args.out)


if __name__ == "__main__":
    main()

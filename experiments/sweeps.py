"""
sweeps.py — Esplorazione sistematica dello spazio dei parametri (senza super-agente).

E' il corrispettivo della Sez. 5.1-5.2 di Zaccagnino et al.: prima si
caratterizza il comportamento del mercato "libero", poi si misura quanto la
policy appresa lo migliora.

Metrica principale
------------------
**Bubble Frequency B**: frazione di run in cui il premium index supera
theta_B (default 2.0). E' costruita esattamente come la *virality* V del
modello fake news:

    V = (1/T) * sum_i [ global_cascade_i > theta_V ]
    B = (1/T) * sum_i [ premium_i        > theta_B ]

Ogni combinazione di parametri viene replicata `n_runs` volte con reti
rigenerate da zero, per assorbire l'eterogeneita' strutturale (stessa logica
delle 1.000 iterazioni per punto in Tornberg 2018).

Uso:
    python -m experiments.sweeps --all --runs 30 --out results/sweeps
    python -m experiments.sweeps --sweep polarization --runs 50
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
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from config import SimConfig
from pokesim.model import PokemonMarketModel


# ---------------------------------------------------------------------- #
# Esecuzione di una singola run                                          #
# ---------------------------------------------------------------------- #
def run_once(cfg_kwargs: dict, seed: int) -> dict:
    """Esegue una simulazione e ne restituisce le metriche riassuntive."""
    cfg = SimConfig(**cfg_kwargs).with_(seed=seed)
    model = PokemonMarketModel(cfg)
    df = model.run()

    # Si escludono i primi 26 tick (mezzo anno) come burn-in: il mercato parte
    # sempre vuoto e i primi prezzi non sono informativi.
    burn = min(26, len(df) // 4)
    tail = df.iloc[burn:]

    premium_final = float(df["premium_index"].iloc[-1])
    premium_peak = float(tail["premium_index"].max())
    premium_mean = float(tail["premium_index"].mean())

    return {
        **{k: v for k, v in cfg_kwargs.items()},
        "seed": seed,
        "premium_final": premium_final,
        "premium_peak": premium_peak,
        "premium_mean": premium_mean,
        "bubble": int(premium_peak > cfg.bubble_threshold),
        "access_recent": float(tail["access_recent"].mean()),
        "access_latest": float(tail["access_latest"].mean()),
        "access_any": float(df["access_any"].iloc[-1]),
        "gini": float(df["gini_holdings"].iloc[-1]),
        "reseller_capture": float(df["reseller_capture"].iloc[-1]),
        "unsold_ratio": float(df["unsold_ratio"].iloc[-1]),
        "mean_hype": float(tail["mean_hype"].mean()),
        "brand_equity": float(df["brand_equity"].iloc[-1]),
        "margin_rate": float(tail["margin_rate"].mean()),
        "opened_ratio": float(tail["opened_ratio"].mean()),
        "mean_discount": float(tail["mean_discount"].mean()),
        "investor_capture": float(df["investor_capture"].iloc[-1]),
        "collapse": int(premium_mean < cfg.collapse_threshold),
        "volume_total": int(df["volume"].sum()),
        "realized_polarization": model.net_summary["realized_polarization"],
    }


def _worker(args):
    cfg_kwargs, seed = args
    return run_once(cfg_kwargs, seed)


def run_grid(base: dict, grid: dict, n_runs: int, workers: int | None = None,
             label: str = "") -> pd.DataFrame:
    """Prodotto cartesiano di `grid`, `n_runs` repliche per combinazione."""
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    jobs = []
    for combo in combos:
        kw = dict(base)
        kw.update(dict(zip(keys, combo)))
        for r in range(n_runs):
            jobs.append((kw, 1000 + r))

    print(f"[{label}] {len(combos)} combinazioni x {n_runs} run = {len(jobs)} simulazioni")
    t0 = time.time()

    if workers == 1:
        rows = [_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_worker, jobs, chunksize=1))

    print(f"[{label}] completato in {(time.time()-t0)/60:.1f} min")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Media, deviazione standard e Bubble Frequency per gruppo."""
    g = df.groupby(by, as_index=False).agg(
        bubble_frequency=("bubble", "mean"),
        premium_mean=("premium_mean", "mean"),
        premium_mean_std=("premium_mean", "std"),
        premium_peak=("premium_peak", "mean"),
        access=("access_recent", "mean"),
        access_std=("access_recent", "std"),
        margin=("margin_rate", "mean"),
        brand=("brand_equity", "mean"),
        collapse_frequency=("collapse", "mean"),
        opened=("opened_ratio", "mean"),
        gini=("gini", "mean"),
        reseller_capture=("reseller_capture", "mean"),
        unsold=("unsold_ratio", "mean"),
        n_runs=("bubble", "size"),
    )
    return g


# ---------------------------------------------------------------------- #
# Definizione degli sweep                                                #
# ---------------------------------------------------------------------- #
BASE = dict(n_agents=250, years=8.0, use_super_agent=False)

# soglie di attivazione: gli stessi tre valori dei paper di riferimento
THETAS = [0.270, 0.342, 0.414]
PN_GRID = [round(x, 3) for x in np.linspace(0.0, 1.0, 11)]

SWEEPS = {
    # S1 - analogo diretto della Fig. 9 di Zaccagnino et al.:
    #      effetto echo chamber sulla frequenza delle bolle
    "polarization": dict(
        grid={"network_polarization": PN_GRID,
              "activation_threshold": THETAS},
        by=["network_polarization", "activation_threshold"],
    ),
    # S2 - polarizzazione di opinione (sinergia con quella di rete)
    "opinion": dict(
        grid={"network_polarization": PN_GRID,
              "opinion_polarization": [0.0, 0.10, 0.20]},
        by=["network_polarization", "opinion_polarization"],
    ),
    # S3 - capacita' industriale x composizione della popolazione
    "supply": dict(
        grid={"supply_ratio": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
              "investor_share": [0.05, 0.25, 0.45, 0.65]},
        by=["supply_ratio", "investor_share"],
    ),
    # S4 - prezzo di listino
    "msrp": dict(
        grid={"msrp": [108.0, 162.0, 216.0, 270.0, 324.0, 432.0],
              "investor_share": [0.10, 0.25, 0.45]},
        by=["msrp", "investor_share"],
    ),
    # S5 - cadenza: a capacita' FISSA, piu' release = set piu' piccoli.
    #      E' la formulazione corretta della domanda "piu' o meno uscite?"
    "cadence": dict(
        grid={"releases_per_year": [2, 3, 4, 6, 8, 12],
              "supply_ratio": [0.25, 0.5, 1.0]},
        by=["releases_per_year", "supply_ratio"],
    ),
    # S6 - topologia della rete
    "topology": dict(
        grid={"network_type": ["erdos_renyi", "barabasi_albert", "watts_strogatz"],
              "network_polarization": [0.0, 0.2, 0.4, 0.6, 0.8]},
        by=["network_type", "network_polarization"],
    ),
    # S7 - dimensione della popolazione (robustezza)
    "size": dict(
        grid={"n_agents": [100, 200, 300, 500, 800],
              "network_polarization": [0.2, 0.4, 0.6]},
        by=["n_agents", "network_polarization"],
    ),
    # S8 - potere dell'influencer: la leva culturale
    "influencer": dict(
        grid={"influencer_power": [0.0, 0.10, 0.25, 0.50, 0.75],
              "investor_share": [0.10, 0.25, 0.45]},
        by=["influencer_power", "investor_share"],
    ),
    # S9 - appetibilita' del set: la leva di raffreddamento
    "appeal": dict(
        grid={"appeal_target": [0.10, 0.25, 0.40, 0.55, 0.70, 0.85],
              "investor_share": [0.10, 0.25, 0.45]},
        by=["appeal_target", "investor_share"],
    ),
    # S10 - quota di prodotto che viene aperta: il pozzo di distruzione
    "opening": dict(
        grid={"open_rate_bounds": [(0.0, 0.0), (0.1, 0.3), (0.2, 0.5),
                                   (0.2, 0.8), (0.5, 0.9)],
              "investor_share": [0.10, 0.25, 0.45]},
        by=["open_rate_bounds", "investor_share"],
    ),
    # S11 - composizione: investitori puri vs flipper vs collezionisti
    "population": dict(
        grid={"investor_share": [0.0, 0.15, 0.30, 0.45, 0.60],
              "fan_flipper_share": [0.0, 0.20, 0.40, 0.60]},
        by=["investor_share", "fan_flipper_share"],
    ),
    # S12 - finestra di stampa: quanto a lungo un set resta producibile
    "print_window": dict(
        grid={"print_window_ticks": [26, 52, 104, 156, 260],
              "restock_share": [0.0, 0.20, 0.40]},
        by=["print_window_ticks", "restock_share"],
    ),
}


# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Sweep parametrici del baseline")
    ap.add_argument("--sweep", type=str, default=None, choices=list(SWEEPS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--agents", type=int, default=300)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=str, default="results/sweeps")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = dict(BASE, years=args.years, n_agents=args.agents)

    names = list(SWEEPS) if args.all else [args.sweep]
    if names == [None]:
        ap.error("specificare --sweep NOME oppure --all")

    for name in names:
        spec = SWEEPS[name]
        b = dict(base)
        if name == "size":
            b.pop("n_agents", None)          # n_agents e' nella griglia
        raw = run_grid(b, spec["grid"], args.runs, args.workers, label=name)
        raw.to_csv(os.path.join(args.out, f"{name}_raw.csv"), index=False)
        agg = aggregate(raw, spec["by"])
        agg.to_csv(os.path.join(args.out, f"{name}_agg.csv"), index=False)
        print(agg.head(12).to_string(index=False), "\n")


if __name__ == "__main__":
    main()

# A Regulatory Sandbox for Scarcity Markets

Agent-based simulation and deep reinforcement learning applied to Pokémon TCG collecting.

This project ports the two-tier sandbox architecture of Zaccagnino et al. (2025),
*Turning AI into a regulatory sandbox*, from information disorder to markets built on
artificial scarcity. The working hypothesis is that a speculative bubble on a collectible
and a misinformation cascade are the same formal object: a complex contagion over a
polarized social network, with a higher-level entity trying to intervene.

The agent-based tier is rewritten from NetLogo to **Mesa 3**; the super-agent from Keras
to **PyTorch** (Double DQN).

> 🇮🇹 Full model documentation, mechanism-by-mechanism, is in Italian: [README.it.md](README.it.md)

## Install

Python 3.12 or 3.13 required (Mesa 3.5 declares `requires-python >= 3.12`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Quick check:

```bash
python -c "from config import SimConfig; from pokesim.model import PokemonMarketModel; print(PokemonMarketModel(SimConfig(n_agents=100, years=1.0, seed=0)).run().tail(3))"
```

## Run

Always launch from the project root (the folder containing `config.py`).

```bash
python -m experiments.sweeps --all --runs 30 --workers 8 --out results/sweeps
python -m rl.train --episodes 150 --test-episodes 40 --out results/rl
python -m experiments.plots --results results --out figures
```

Indicative timings (8 cores, N=250, 8 years): one run ≈ 2–4 s, a full sweep at 30
replicates ≈ 40–90 min, training ≈ 30–60 min on CPU.

## Layout

```
config.py        all parameters, in one frozen serializable dataclass
pokesim/         model-driven tier: networks, agents, market, Mesa model, metrics
rl/              data-driven tier: Gymnasium env, Double DQN, training
experiments/     parameter sweeps and figure generation
```

## Model in one paragraph

One tick is one week. Market participants hold a continuous *hype* in [0,1] that spreads
by complex contagion over a polarized network, with unconditional attention decay on top —
that decay term is what produces cycles instead of saturation. Three archetypes (fans,
fan-flippers, pure investors) buy on a first-come-first-served primary market and trade on
a secondary one. Opened boxes leave circulation permanently, which is the only mechanism
that shrinks supply. Industrial capacity is fixed and takes ~1.5 years to expand, so
"print more" carries a real opportunity cost. The super-agent (The Pokémon Company)
observes 8 normalized signals and picks among 11 actions on capacity, MSRP, release
cadence, restocks and set appeal. Its reward weights are the explicit normative stance:
changing them changes the optimal policy, which is precisely what a regulatory sandbox
should let you explore.

## Verified results

- **The echo chamber effect transfers.** Bubble frequency rises with network polarization
  up to 0.4–0.6 and collapses beyond 0.7 — the same inverted U as Törnberg (2018).
- **Accessibility has an interior maximum.** It peaks at ~1.5 units per agent (0.72) and
  *falls* beyond it (0.55 at 4×). Overprinting hurts collectors too.
- **Opening boxes raises prices.** Premium 2.65 → 3.00 as `open_rate` goes 0 → 0.6–0.9.
- **Pure investors are the main driver.** Premium 2.37 without them, 4.48 at 60%.
- **Set appeal is a cooling lever with a cost.** Premium 3.70 → 1.06, but margin drops
  from 3.79 to 0.87.
- **No single lever dominates.** The three best constant actions are within 10% of each
  other and a hand-written mixed policy beats them all.

## Known limitations

Distributors are an aggregate price function, not explicit agents. Only one echo chamber,
as in the reference model. No calibration on real price data. The PSA graded-singles market
is excluded by design.

## References

1. Zaccagnino R., Lettieri N., Malandrino D., Lomasto L., Camoia A., Guarino A. (2025).
   *Turning AI into a regulatory sandbox*. Neural Computing and Applications 37, 18679–18720.
2. Törnberg P. (2018). *Echo chambers and viral misinformation: Modeling fake news as
   complex contagion*. PLoS ONE 13(9): e0203958.
3. ter Hoeven E. et al. (2025). *Mesa 3: Agent-based modeling with Python in 2025*. JOSS 10(107), 7668.
4. van Hasselt H., Guez A., Silver D. (2015). *Deep reinforcement learning with double Q-learning*.

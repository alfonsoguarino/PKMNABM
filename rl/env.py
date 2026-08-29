"""
env.py — Ambiente Gymnasium: il ponte fra i due tier.

E' l'equivalente diretto di `environment/fake_news_diffusion_env.py` del
progetto FakeNewsDetection, con NetLogo sostituito da Mesa (quindi in-process,
senza JVM ne' pyNetLogo: e' il motivo principale per cui questa versione gira
ordini di grandezza piu' velocemente).

Corrispondenze
--------------
    FakeNewsSimulation                  TPCSandboxEnv
    ------------------------------      -----------------------------------
    observation_space = Box(3,)         Box(8,)
    action_space = Discrete(4)          Discrete(9)
    global_cascade                      premium_index
    CalculateReward1                    _reward()
    netlogo.choose_action + go          model.apply_policy_action + step

Nota sul reward
---------------
La componente principale replica la forma del reward originale (segno della
variazione della metrica-obiettivo, pesato da `action_weight`), ma da sola
sarebbe degenere: il super-agente imparerebbe subito a stampare all'infinito.
Sono quindi aggiunti tre termini di costo lato produttore che rendono il
problema un vero trade-off:
  * invenduto      -> stampare troppo distrugge margine e valore del marchio;
  * accessibilita' -> alzare l'MSRP abbassa il premium ma esclude i piu' poveri;
  * affordability  -> penalita' esplicita sulla crescita dell'MSRP reale.
E' questo trade-off, e non il singolo obiettivo, a rendere la policy appresa
scientificamente interessante.
"""

from __future__ import annotations

import math

import numpy as np
from gymnasium import Env, spaces

from config import SimConfig
from pokesim.model import N_ACTIONS, PokemonMarketModel


class TPCSandboxEnv(Env):
    """Sandbox regolatorio: la Pokemon Company come agente RL."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: SimConfig):
        super().__init__()
        self.cfg = cfg

        # La dimensione dell'osservazione viene DEDOTTA dal modello, non
        # scritta a mano: si costruisce un modello effimero e si misura la
        # lunghezza del vettore che produce. Cosi' se un giorno si aggiunge o
        # toglie una componente allo stato, lo spazio di osservazione e la rete
        # del DDQN restano automaticamente coerenti. Fissare il numero a mano
        # e' esattamente cio' che aveva prodotto il disallineamento 6-vs-8.
        obs_dim = PokemonMarketModel(cfg.with_(seed=0)).observation().shape[0]
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.model: PokemonMarketModel | None = None
        self.prev_premium = 1.0
        self._episode_seed = cfg.seed

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.cfg if seed is None else self.cfg.with_(seed=seed)
        self.model = PokemonMarketModel(cfg)
        self.prev_premium = self.model.premium()
        return self.model.observation(), {}

    # ------------------------------------------------------------------ #
    def step(self, action):
        assert self.action_space.contains(int(action)), f"azione non valida: {action}"
        m = self.model

        m.apply_policy_action(int(action))

        # Il super-agente decide 1 volta ogni `sa_delay` tick; nel frattempo
        # il mercato evolve da solo. Identico al parametro `sa-delay` del
        # framework originale.
        for _ in range(max(1, self.cfg.sa_delay)):
            m.step()

        obs = m.observation()
        reward = self._reward(int(action))
        terminated = m.steps >= self.cfg.total_ticks
        info = {
            "premium": m.premium(),
            "access": m.history[-1]["access_recent"],
            "unsold": m.history[-1]["unsold_ratio"],
            "msrp": m.current_msrp,
            "capacity_multiplier": m.capacity_multiplier,
            "restock_share": m.restock_share,
            "release_interval": m.release_interval,
            "tick": m.steps,
        }
        self.prev_premium = m.premium()
        return obs, float(reward), terminated, False, info

    # ------------------------------------------------------------------ #
    def _reward(self, action: int) -> float:
        """
        r = termine-prezzo (forma del paper) + accesso - invenduto
            - crescita MSRP - penalita' bolla
        """
        cfg = self.cfg
        m = self.model
        snap = m.history[-1]

        premium = snap["premium_index"]
        d_premium = premium - self.prev_premium

        # action_weight: stessa struttura di Zaccagnino et al. (Eq. reward).
        # Un mercato con influencer poco caldo e hype medio basso rende ogni
        # intervento piu' "economico", quindi piu' premiante.
        action_weight = (1.0 - snap["influencer_hype"]) + (1.0 - snap["mean_hype"])

        if action == 0:                              # no_op
            base = 1.0 if d_premium <= 0 else 0.0
        else:
            if d_premium <= 0:
                base = (1.0 + action_weight) * 0.5 - d_premium
            else:
                base = (0.0 + action_weight) * 0.5

        access = snap["access_recent"]
        unsold = snap["unsold_ratio"]
        msrp_growth = max(0.0, m.current_msrp / cfg.msrp - 1.0)
        bubble = 1.0 if premium > cfg.bubble_threshold else 0.0
        collapse = max(0.0, cfg.collapse_threshold - premium)

        m = snap["margin_rate"]
        if m >= 0.0:
            margin = math.log1p(m) / math.log1p(cfg.margin_reference)
        else:
            margin = max(-2.0, m)          # le perdite pesano per intero

        return (base
                + cfg.w_access * access
                + cfg.w_margin * margin
                - cfg.w_unsold * unsold
                - cfg.w_afford * msrp_growth
                - cfg.w_bubble * bubble
                - cfg.w_collapse * collapse)

    # ------------------------------------------------------------------ #
    def close(self):
        self.model = None

    # -- utilita' per gli esperimenti ---------------------------------- #
    def rollout(self, policy_fn, seed: int | None = None):
        """
        Esegue un episodio completo con una policy arbitraria e restituisce
        (dataframe della simulazione, reward totale, lista delle azioni).
        `policy_fn(obs) -> int`.
        """
        obs, _ = self.reset(seed=seed)
        total, actions, done = 0.0, [], False
        while not done:
            a = int(policy_fn(obs))
            obs, r, done, _, _ = self.step(a)
            total += r
            actions.append(a)
        return self.model.to_dataframe(), total, actions

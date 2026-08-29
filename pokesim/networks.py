"""
networks.py — Generazione della rete sociale e costruzione dell'echo chamber.

La procedura di costruzione del cluster e' volutamente identica a quella di
Tornberg (2018, PLoS ONE) ripresa da Zaccagnino et al. (2025):

  (i)   si sceglie una frazione c degli N nodi come echo chamber;
  (ii)  si selezionano P_n * |E_cross| archi che collegano il cluster all'esterno;
  (iii) si rimuovono e si sostituiscono con archi interni al cluster;
  (iv)  i nodi del cluster ricevono una soglia ridotta theta - P_o.

La novita' rispetto al modello base e' che qui la rete non porta un'opinione
binaria (fake/true) ma un livello continuo di *hype* su ciascun prodotto
aggregato, e che il nodo a centralita' massima viene promosso a `influencer`.
"""

from __future__ import annotations

import random
from typing import Sequence

import networkx as nx


# ---------------------------------------------------------------------- #
# 1. Generatori di topologia                                             #
# ---------------------------------------------------------------------- #
def build_base_graph(n: int, network_type: str, avg_degree: int,
                     ws_rewire_p: float, rng: random.Random) -> nx.Graph:
    """Crea il grafo di partenza con grado medio ~ avg_degree."""
    seed = rng.randint(0, 2**31 - 1)

    if network_type == "erdos_renyi":
        p = min(1.0, avg_degree / max(1, n - 1))
        g = nx.erdos_renyi_graph(n, p, seed=seed)

    elif network_type == "barabasi_albert":
        m = max(1, avg_degree // 2)          # grado medio ~ 2m
        g = nx.barabasi_albert_graph(n, m, seed=seed)

    elif network_type == "watts_strogatz":
        k = max(2, avg_degree if avg_degree % 2 == 0 else avg_degree + 1)
        g = nx.watts_strogatz_graph(n, k, ws_rewire_p, seed=seed)

    else:
        raise ValueError(f"network_type sconosciuto: {network_type!r}")

    # Garantisce che ogni nodo abbia almeno un vicino: un nodo isolato non puo'
    # partecipare al contagio complesso e falserebbe le medie.
    isolated = [v for v in g.nodes if g.degree(v) == 0]
    for v in isolated:
        u = rng.choice([w for w in g.nodes if w != v])
        g.add_edge(v, u)

    return g


# ---------------------------------------------------------------------- #
# 2. Echo chamber                                                        #
# ---------------------------------------------------------------------- #
def carve_echo_chamber(g: nx.Graph, fraction: float, polarization: float,
                       rng: random.Random) -> set[int]:
    """
    Seleziona il cluster e ne aumenta la coesione riallocando archi.

    Ritorna l'insieme dei nodi appartenenti all'echo chamber.
    Il grafo `g` viene modificato in place.
    """
    nodes = list(g.nodes)
    n_cluster = max(2, int(round(fraction * len(nodes))))
    cluster = set(rng.sample(nodes, n_cluster))

    if polarization <= 0:
        return cluster

    cross_edges = [(u, v) for u, v in g.edges
                   if (u in cluster) != (v in cluster)]
    rng.shuffle(cross_edges)
    n_to_move = int(round(polarization * len(cross_edges)))

    cluster_list = sorted(cluster)
    moved = 0
    for u, v in cross_edges[:n_to_move]:
        inside = u if u in cluster else v
        # cerca un partner interno non ancora collegato
        candidates = [w for w in cluster_list
                      if w != inside and not g.has_edge(inside, w)]
        if not candidates:
            continue                      # cluster gia' saturo (clique)
        g.remove_edge(u, v)
        g.add_edge(inside, rng.choice(candidates))
        moved += 1

    # Evita nodi isolati creati dalla rimozione
    for v in [w for w in g.nodes if g.degree(w) == 0]:
        u = rng.choice([w for w in g.nodes if w != v])
        g.add_edge(v, u)

    return cluster


# ---------------------------------------------------------------------- #
# 3. Individuazione degli influencer                                     #
# ---------------------------------------------------------------------- #
def pick_influencers(g: nx.Graph, n_influencers: int,
                     criterion: str = "betweenness") -> list[int]:
    """
    Ritorna gli `n_influencers` nodi a centralita' massima.

    `betweenness` e' la scelta richiesta dal disegno sperimentale (il nodo
    che fa da ponte fra comunita' e' quello che monetizza l'attenzione);
    `pagerank` e `degree` sono forniti per l'analisi di sensitivita', in
    parallelo alle tre criteria del framework originale.
    """
    if criterion == "betweenness":
        # k-sampling: su reti > 500 nodi il calcolo esatto e' O(nm) e domina
        # il tempo di setup senza cambiare il ranking dei top-k.
        k = None if g.number_of_nodes() <= 500 else 200
        scores = nx.betweenness_centrality(g, k=k, seed=42)
    elif criterion == "pagerank":
        scores = nx.pagerank(g)
    elif criterion == "degree":
        scores = dict(g.degree())
    else:
        raise ValueError(f"criterio sconosciuto: {criterion!r}")

    ranked = sorted(scores, key=lambda v: scores[v], reverse=True)
    return ranked[:n_influencers]


# ---------------------------------------------------------------------- #
# 4. Descrittori strutturali (per le tabelle dell'articolo)              #
# ---------------------------------------------------------------------- #
def network_summary(g: nx.Graph, cluster: Sequence[int]) -> dict:
    cluster = set(cluster)
    internal = sum(1 for u, v in g.edges if u in cluster and v in cluster)
    cross = sum(1 for u, v in g.edges if (u in cluster) != (v in cluster))
    degrees = [d for _, d in g.degree()]
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "avg_degree": sum(degrees) / len(degrees),
        "clustering": nx.average_clustering(g),
        "cluster_size": len(cluster),
        "cluster_internal_edges": internal,
        "cluster_cross_edges": cross,
        "realized_polarization": internal / max(1, internal + cross),
    }

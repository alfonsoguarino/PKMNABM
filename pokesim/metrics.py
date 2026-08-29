"""
metrics.py — Indicatori di stato del mercato.

L'indicatore centrale e' il **premium index**: e' l'analogo strutturale del
`global_cascade` di Zaccagnino et al. (2025). Come li' il global cascade era la
frazione di nodi contagiati e la *virality* V era la frazione di run in cui il
cascade superava theta_V = 0.5, qui il premium index e' il rapporto medio
prezzo secondario / MSRP e la **Bubble Frequency** B e' la frazione di run in
cui il premium supera theta_B (default 2.0, cioe' il doppio del prezzo di
listino).

    Tornberg / Zaccagnino          Questo modello
    ---------------------------    -------------------------------------
    global_cascade (frazione)      premium_index (rapporto prezzo/MSRP)
    theta_V = 0.5                  theta_B = 2.0
    Virality V                     Bubble Frequency B
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------- #
# Disuguaglianza                                                         #
# ---------------------------------------------------------------------- #
def gini(values) -> float:
    """Coefficiente di Gini. 0 = perfetta uguaglianza, 1 = massima concentrazione."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(np.clip(x, 0, None))
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


# ---------------------------------------------------------------------- #
# Prezzo                                                                 #
# ---------------------------------------------------------------------- #
def premium_index(products, tick: int, window: int = 8,
                  order_book=None) -> float:
    """
    Premium medio del mercato, pesato per volume recente.

    Priorita' delle fonti di prezzo, in ordine:
      1. prezzi effettivamente scambiati nella finestra recente;
      2. se un prodotto non ha scambi recenti ma ha uno storico, l'ultimo prezzo;
      3. se non ha mai scambiato ma ha ask a book, il miglior ask (prezzo
         richiesto: e' informativo perche' segnala l'ancoraggio dei venditori);
      4. altrimenti il prodotto e' escluso.

    Ritorna 1.0 se nessun prodotto ha informazione di prezzo (mercato al MSRP).
    """
    ratios, weights = [], []
    for p in products:
        vol = p.recent_volume(tick, window)
        candidates = []
        if p.last_traded is not None:
            candidates.append((p.last_traded, max(1.0, float(vol))))
        if order_book is not None:
            ask = order_book.best_ask(p.pid)
            if ask is not None:
                candidates.append((ask.price, 0.5))
        # Il prodotto ancora a magazzino compete con il secondario: nessuno
        # paga 2x sul mercato dell'usato cio' che il negozio sta svendendo.
        # Ignorare questa fonte sovrastimerebbe il premium proprio nei casi di
        # sovraccarico produttivo, cioe' dove serve misurarlo bene.
        if p.is_available():
            candidates.append((p.price, 1.0))
        if not candidates:
            continue
        price, w = min(candidates, key=lambda c: c[0])
        ratios.append(price / p.msrp_at_release)
        weights.append(w)
    if not ratios:
        return 1.0
    return float(np.average(ratios, weights=weights))


def median_secondary_price(products) -> float | None:
    prices = [p.last_traded for p in products if p.last_traded is not None]
    return float(np.median(prices)) if prices else None


# ---------------------------------------------------------------------- #
# Accessibilita'                                                         #
# ---------------------------------------------------------------------- #
def access_rate(agents, product_id: int | None) -> float:
    """
    Frazione di collezionisti puri che hanno OTTENUTO il set indicato.

    Si usa `was_served` e non `owns`: un fan che ha comprato la box e l'ha
    aperta e' stato servito a tutti gli effetti, anche se non ha piu' nulla
    di sigillato in inventario. Misurare il possesso residuo confonderebbe
    l'accessibilita' con la tesaurizzazione.
    """
    collectors = [a for a in agents if a.is_pure_collector]
    if not collectors or product_id is None:
        return 0.0
    return sum(1 for a in collectors if a.was_served(product_id)) / len(collectors)


def access_recent(agents, products, tick: int, window: int = 52) -> float:
    """
    Frazione di collezionisti puri serviti da ALMENO UN set uscito nell'ultimo
    anno. E' la misura di accessibilita' corretta per il confronto fra cadenze
    diverse.

    Misurare invece la copertura del solo set piu' recente introduce un
    artefatto grave: con una cadenza di 26 settimane quel set resta "il piu'
    recente" per il triplo del tempo rispetto a una cadenza di 9, e accumula
    acquirenti solo per questo. Rallentare le uscite risulterebbe cosi'
    artificialmente premiante, che e' esattamente la conclusione sbagliata da
    far emergere da un sandbox regolatorio.
    """
    collectors = [a for a in agents if a.is_pure_collector]
    if not collectors:
        return 0.0
    recent = [p.pid for p in products if p.release_tick >= tick - window]
    if not recent:
        return 0.0
    return sum(1 for a in collectors
               if any(a.was_served(pid) for pid in recent)) / len(collectors)


def access_rate_any(agents) -> float:
    collectors = [a for a in agents if a.is_pure_collector]
    if not collectors:
        return 0.0
    return sum(1 for a in collectors
               if a.n_units > 0 or a.opened_pids) / len(collectors)


def opened_ratio(products) -> float:
    """Quota del venduto che e' stata aperta e quindi distrutta."""
    sold = sum(p.units_sold_primary for p in products)
    if sold == 0:
        return 0.0
    return sum(p.units_opened for p in products) / sold


def investor_capture(products) -> float:
    """Quota della tiratura primaria finita a investitori puri."""
    sold = sum(p.units_sold_primary for p in products)
    if sold == 0:
        return 0.0
    return sum(p.units_to_investors for p in products) / sold


def sealed_per_capita(products, n_agents: int) -> float:
    """Sigillato ancora in circolo per agente: la vera offerta disponibile."""
    if n_agents == 0:
        return 0.0
    return sum(p.sealed_supply for p in products) / n_agents


def affordability(agents, price: float) -> float:
    """Frazione di agenti il cui reddito mensile copre almeno una copia."""
    agents = list(agents)
    if not agents:
        return 0.0
    return sum(1 for a in agents if a.income >= price) / len(agents)


# ---------------------------------------------------------------------- #
# Offerta                                                                #
# ---------------------------------------------------------------------- #
def unsold_ratio(products, tick: int | None = None,
                 window: int | None = None) -> float:
    """
    Quota di tiratura rimasta invenduta sul primario.

    Va misurata su una finestra RECENTE: includendo tutti i set mai usciti, i
    vecchi prodotti esauriti dominano il denominatore e l'indicatore resta
    incollato a zero, diventando cieco proprio al sovraccarico produttivo che
    dovrebbe segnalare.
    """
    if tick is not None and window is not None:
        products = [p for p in products if p.release_tick >= tick - window]
    released = sum(p.initial_stock for p in products)
    if released == 0:
        return 0.0
    return sum(p.stock for p in products) / released


def reseller_capture(products) -> float:
    """Quota della tiratura primaria finita in mano a chi rivende."""
    sold = sum(p.units_sold_primary for p in products)
    if sold == 0:
        return 0.0
    return sum(p.units_to_resellers for p in products) / sold


def sellout_ticks(product, tick: int) -> int | None:
    """Tick impiegati a esaurire lo stock (None se ancora disponibile)."""
    return None if product.stock > 0 else tick - product.release_tick


# ---------------------------------------------------------------------- #
# Aggregatore per il DataCollector                                       #
# ---------------------------------------------------------------------- #
def snapshot(model) -> dict:
    """Fotografia completa dello stato del mercato in un tick."""
    agents = list(model.agents)
    products = model.active_products()
    latest = model.latest_product

    return {
        "tick": model.steps,
        "premium_index": premium_index(products, model.steps,
                                       order_book=model.book),
        "median_price": median_secondary_price(products),
        "mean_hype": float(np.mean([a.hype for a in agents])) if agents else 0.0,
        "influencer_hype": float(np.mean([a.hype for a in model.influencers]))
                           if model.influencers else 0.0,
        "access_latest": access_rate(agents, latest.pid if latest else None),
        "access_recent": access_recent(agents, products, model.steps),
        "access_any": access_rate_any(agents),
        "gini_holdings": gini([a.n_units for a in agents]),
        "unsold_ratio": unsold_ratio(products, model.steps, model.cfg.unsold_window),
        "scarcity_signal": model.scarcity_signal,
        "brand_equity": model.brand_equity,
        "margin_rate": model.margin_rate(),
        "mean_discount": float(np.mean([p.discount for p in products])) if products else 0.0,
        "reseller_capture": reseller_capture(products),
        "investor_capture": investor_capture(products),
        "opened_ratio": opened_ratio(products),
        "sealed_per_capita": sealed_per_capita(products, len(agents)),
        "units_opened_tick": model.tick_opened,
        "n_listings": model.book.n_listings(),
        "volume": model.tick_volume,
        "msrp": model.current_msrp,
        "capacity_multiplier": model.capacity_multiplier,
        "target_capacity": model.target_capacity,
        "restock_share": model.restock_share,
        "mean_appeal": float(np.mean([p.appeal for p in products])) if products else 0.5,
        "latest_appeal": latest.appeal if latest else 0.5,
        "n_liquidity_shocks": sum(1 for a in agents if a.in_liquidity_shock),
        "release_interval": model.release_interval,
        "mean_budget": float(np.mean([a.budget for a in agents])) if agents else 0.0,
        "n_products": len(products),
    }

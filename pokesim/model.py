"""
model.py — Il modello ad agenti (tier *model-driven* del framework).

Corrisponde al file `.nlogo` del progetto FakeNewsDetection, riscritto in
Mesa 3.x. Espone la stessa interfaccia concettuale che li' era offerta da
`NetlogoCommands`: `setup()` (qui il costruttore), `step()` (= `go`), le
azioni del super-agente e i reporter di stato.

Struttura di un tick (1 tick = 1 settimana)
-------------------------------------------
 1. `_monthly_income()`      ogni 4 tick ricarica i budget
 2. `_maybe_release()`       rilascia un nuovo set e inietta hype nel cluster
 3. `_primary_market()`      allocazione FCFS a MSRP
 4. `_diffuse_hype()`        contagio complesso + segnale dell'influencer
 5. `_post_listings()`       i reseller quotano sul secondario
 6. `_secondary_trading()`   i compratori colpiscono il miglior ask
 7. `_collect()`             metriche

Il super-agente NON e' un `mesa.Agent`: e' esterno al modello e agisce tramite
`apply_policy_action()`, esattamente come nel framework originale dove il
super-agente vive nel tier data-driven e comanda la simulazione dall'esterno.
"""

from __future__ import annotations

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from config import SimConfig, TICKS_PER_MONTH, TICKS_PER_YEAR
from pokesim import metrics
from pokesim.agents import CollectorAgent
from pokesim.market import OrderBook, Product
from pokesim.networks import (build_base_graph, carve_echo_chamber,
                              network_summary, pick_influencers)


# ---------------------------------------------------------------------- #
# Azioni del super-agente (Pokemon Company)                              #
# ---------------------------------------------------------------------- #
# Due assenze sono deliberate.
#
# Il tetto di copie per persona NON compare fra le azioni: la Pokemon Company
# vende a distributori intermedi, non ai consumatori finali, e non ha alcun
# modo di far rispettare un limite individuale lungo tutta la filiera (per
# tacere del fatto che basterebbe ordinare una copia da ogni rivenditore).
# Modellarlo come leva disponibile darebbe al super-agente uno strumento che
# nella realta' non possiede.
#
# Non esiste nemmeno un'azione "stampa di piu' subito": la capacita' annua e'
# fissa e sempre saturata, quindi l'unica leva immediata e' la RIPARTIZIONE
# fra prodotto nuovo e ristampe (`restock_more` / `restock_less`), mentre
# l'espansione della capacita' (`capacity_up`) si materializza solo dopo
# `capacity_lag_ticks`. Chi decide oggi vede l'effetto fra un anno e mezzo:
# e' questa distanza fra decisione ed esito a rendere il problema di controllo
# interessante invece che una massimizzazione miope.
ACTIONS = {
    0: "no_op",
    1: "capacity_up",
    2: "capacity_down",
    3: "msrp_up",
    4: "msrp_down",
    5: "cadence_faster",
    6: "cadence_slower",
    7: "restock_more",
    8: "restock_less",
    9: "appeal_up",
    10: "appeal_down",
}
N_ACTIONS = len(ACTIONS)


class PokemonMarketModel(Model):
    """Mercato del collezionismo Pokemon TCG con echo chamber e influencer."""

    def __init__(self, cfg: SimConfig):
        super().__init__(seed=cfg.seed)
        self.cfg = cfg

        # ---------------- rete ---------------- #
        self.graph = build_base_graph(cfg.n_agents, cfg.network_type,
                                      cfg.avg_degree, cfg.ws_rewire_p,
                                      self.random)
        self.cluster = carve_echo_chamber(self.graph,
                                          cfg.echo_chamber_fraction,
                                          cfg.network_polarization,
                                          self.random)
        influencer_nodes = set(pick_influencers(self.graph, cfg.n_influencers,
                                                cfg.influencer_centrality))
        self.net_summary = network_summary(self.graph, self.cluster)

        # ---------------- agenti ---------------- #
        self._by_node: dict[int, CollectorAgent] = {}

        # Composizione della popolazione: investitori puri, fan-flipper, fan.
        nodes_shuffled = sorted(self.graph.nodes)
        self.random.shuffle(nodes_shuffled)
        n_inv = int(round(cfg.investor_share * cfg.n_agents))
        investor_nodes = set(nodes_shuffled[:n_inv])
        fan_nodes = nodes_shuffled[n_inv:]
        n_flip = int(round(cfg.fan_flipper_share * len(fan_nodes)))
        flipper_nodes = set(fan_nodes[:n_flip])

        for node in sorted(self.graph.nodes):
            in_cluster = node in self.cluster

            income = float(np.exp(self.random.gauss(
                np.log(cfg.income_median), cfg.income_sigma)))
            income = max(30.0, income)

            thr = cfg.activation_threshold + self.random.uniform(
                -cfg.threshold_jitter, cfg.threshold_jitter)
            if in_cluster:
                thr -= cfg.opinion_polarization       # theta - P_o
            thr = min(0.99, max(0.01, thr))

            is_fan = node not in investor_nodes
            if not is_fan:
                lo, hi = cfg.investor_propensity_bounds
                prop = self.random.uniform(lo, hi)
                open_rate = 0.0                  # l'investitore non apre nulla
            elif node in flipper_nodes:
                m = cfg.reseller_propensity_mean
                prop = float(np.random.default_rng(
                    self.random.randint(0, 2**31 - 1)).beta(4.0 * m, 4.0 * (1 - m)))
                prop = min(1.0, max(0.05, prop))
                open_rate = self.random.uniform(*cfg.open_rate_bounds) * 0.5
            else:
                prop = 0.0
                open_rate = self.random.uniform(*cfg.open_rate_bounds)

            agent = CollectorAgent(self, node, income, thr, prop, in_cluster,
                                   is_fan=is_fan, open_rate=open_rate,
                                   liquidity_prob=0.0)
            agent.is_influencer = node in influencer_nodes
            self._by_node[node] = agent

        # Probabilita' di shock di liquidita', decrescente col reddito.
        # Chi ha grandi disponibilita' liquida di rado (un evento ogni ~12
        # anni), chi ha budget limitati assai piu' spesso (fino a ogni ~3).
        # Si usa il reddito e non la cassa del momento perche' cio' che conta
        # e' la fragilita' strutturale, non un saldo temporaneo.
        incomes = sorted(a.income for a in self.agents)
        n_inc = len(incomes)
        p_rich = 1.0 / (cfg.liquidity_years_rich * TICKS_PER_YEAR)
        p_poor = 1.0 / (cfg.liquidity_years_poor * TICKS_PER_YEAR)
        for agent in self.agents:
            pct = incomes.index(agent.income) / max(1, n_inc - 1)
            agent.liquidity_prob = p_poor + (p_rich - p_poor) * pct

        self.influencers = [self._by_node[n] for n in influencer_nodes]

        # Audience: vicini di grafo + campione casuale della popolazione.
        # E' la formalizzazione del fatto che il pubblico di un creator non
        # coincide con la sua cerchia sociale.
        all_nodes = sorted(self.graph.nodes)
        n_reach = int(round(cfg.influencer_reach * cfg.n_agents))
        self.influencer_audience: dict[int, list[CollectorAgent]] = {}
        for node in influencer_nodes:
            followers = set(self.graph.neighbors(node))
            pool = [v for v in all_nodes if v != node and v not in followers]
            extra = self.random.sample(pool, min(len(pool), max(0, n_reach - len(followers))))
            followers.update(extra)
            self.influencer_audience[node] = [self._by_node[v] for v in followers]

        # ---------------- mercato ---------------- #
        self.book = OrderBook()
        self.products: list[Product] = []
        self._product_index: dict[int, Product] = {}
        self.latest_product: Product | None = None
        self._next_pid = 0
        self.tick_volume = 0
        self.tick_opened = 0
        self.scarcity_signal = 1.0        # 1 = tutto esaurito, 0 = tutto invenduto
        self.brand_equity = 1.0           # capitale di marca, erosione di lungo periodo
        self._primary_sales_log: list[tuple[int, int, float]] = []  # (tick, unita', msrp)

        # ---------------- leve di policy ---------------- #
        self.current_msrp = cfg.msrp
        self.capacity_multiplier = 1.0        # capacita' industriale effettiva
        self.target_capacity = 1.0            # obiettivo deliberato
        self._pending_capacity: list[tuple[int, float]] = []
        self.restock_share = cfg.restock_share
        self.appeal_target = cfg.appeal_target
        self._pending_appeal: list[tuple[int, float]] = []
        self.effective_appeal = cfg.appeal_target
        self.msrp_multiplier = 1.0
        self.release_interval = cfg.release_interval
        self._next_release_tick = 0
        self.action_log: list[tuple[int, int]] = []

        # ---------------- raccolta dati ---------------- #
        self.datacollector = DataCollector(
            model_reporters={k: (lambda m, key=k: metrics.snapshot(m)[key])
                             for k in ()}   # placeholder, vedi _collect()
        )
        self.history: list[dict] = []

        self._maybe_release()              # prima release al tick 0
        self._collect()

    # ================================================================== #
    # Accessori                                                          #
    # ================================================================== #
    def neighbors_of(self, node_id: int) -> list[CollectorAgent]:
        return [self._by_node[n] for n in self.graph.neighbors(node_id)]

    def active_products(self) -> list[Product]:
        cut = self.steps - self.cfg.product_lifetime
        return [p for p in self.products if p.release_tick >= cut]

    def product_by_id(self, pid: int) -> Product | None:
        return self._product_index.get(pid)

    @property
    def novelty(self) -> float:
        """
        Freschezza del set piu' recente, in [0,1]: decadimento esponenziale
        con emivita `attention_half_life`. Modula il guadagno di hype: e' cio'
        che impedisce al contagio complesso di saturare a 1 e non tornare piu'
        indietro, e che genera i cicli di attenzione fra una release e l'altra.
        """
        if self.latest_product is None:
            return 0.0
        age = self.steps - self.latest_product.release_tick
        return float(np.exp(-age / max(1e-6, self.cfg.attention_half_life)))

    @property
    def scarcity_appeal(self) -> float:
        """
        Moltiplicatore del guadagno di hype dovuto alla scarsita' percepita.
        Vale 1 quando il set precedente e' andato esaurito, `scarcity_floor`
        quando e' rimasto interamente sugli scaffali.
        """
        f = self.cfg.scarcity_floor
        # La saturazione agisce su due orizzonti: il segnale di breve
        # (`scarcity_signal`, l'ultimo set) e quello di lungo (`brand_equity`,
        # la reputazione accumulata). Il prodotto dei due e' cio' che il
        # sovraccarico produttivo distrugge davvero.
        return f + (1.0 - f) * self.scarcity_signal * self.brand_equity

    def margin_rate(self) -> float:
        """
        Margine industriale dell'ultimo anno, normalizzato sul margine atteso
        a parametri di default (= 1.0 al baseline).

            margine = (unita' VENDUTE x prezzo) - (unita' STAMPATE x costo)

        L'asimmetria fra stampato e venduto e' il punto: chi sovraccarica la
        produzione paga il costo di tutte le copie ma incassa solo su quelle
        che escono dal magazzino. E' l'obiettivo commerciale che il
        super-agente non puo' ignorare, e cio' che impedisce alla policy
        "stampa sempre di piu'" di essere una soluzione.
        """
        cfg = self.cfg
        cut = self.steps - TICKS_PER_YEAR
        unit_cost = cfg.production_cost_ratio * cfg.msrp

        # Il ricavo si realizza al prezzo a cui il prodotto si e' effettivamente
        # mosso: le unita' collocate in svendita fruttano una frazione del
        # listino, e sotto una certa soglia non coprono nemmeno il costo
        # industriale. E' il meccanismo che rende il sovraccarico distruttivo.
        revenue = sum(u * take for t, u, take in self._primary_sales_log if t > cut)

        # Il costo si sostiene su TUTTO lo stampato, non sul venduto: la
        # capacita' e' sempre saturata, quindi e' sempre pagata per intero.
        printed = cfg.capacity_per_year * self.capacity_multiplier

        margin = revenue - printed * unit_cost
        expected = (cfg.capacity_per_year * cfg.msrp
                    * (cfg.channel_wholesale_ratio - cfg.production_cost_ratio))
        return margin / max(1e-9, expected)

    def price_excess_over(self, agent) -> float:
        """
        Quanto il prezzo piu' basso reperibile eccede la WTP dell'agente,
        in unita' di WTP. 0 se il prodotto e' accessibile.
        """
        prod = self.latest_product
        if prod is None:
            return 0.0
        if prod.stock > 0:
            price = prod.price                     # disponibile a listino o a sconto
        else:
            ask = self.book.best_ask(prod.pid)
            price = ask.price if ask else (prod.last_traded or prod.price)
        wtp = agent.willingness_to_pay(prod)
        if wtp <= 0:
            return 0.0
        return max(0.0, (price - wtp) / wtp)

    # ================================================================== #
    # Ciclo di simulazione                                               #
    # ================================================================== #
    def step(self) -> None:
        super().step()                     # incrementa self.steps
        self.tick_volume = 0
        self.tick_opened = 0

        self.book.expire(self.steps, self.cfg.listing_ttl)
        self._apply_pending_capacity()
        self._apply_pending_appeal()
        self._monthly_income()
        self._maybe_release()
        self._do_restocks()
        self._update_clearance()
        self._primary_market()
        self._diffuse_hype()
        self._liquidity_shocks()
        self._open_boxes()
        self._post_listings()
        self._secondary_trading()
        self._collect()

    # ------------------------------------------------------------------ #
    def _apply_pending_appeal(self) -> None:
        """
        Rende effettive le decisioni sull'appetibilita'. Un set si progetta con
        circa un anno di anticipo: la scelta di quali carte chase inserire non
        puo' rispondere al mercato di questa settimana.
        """
        if not self._pending_appeal:
            return
        due = [(t, v) for t, v in self._pending_appeal if t <= self.steps]
        if due:
            self.effective_appeal = due[-1][1]
            self._pending_appeal = [(t, v) for t, v in self._pending_appeal
                                    if t > self.steps]

    def _draw_appeal(self) -> float:
        """Estrae l'appetibilita' del prossimo set attorno all'obiettivo."""
        cfg = self.cfg
        m = min(0.98, max(0.02, self.effective_appeal))
        k = cfg.appeal_concentration
        val = float(np.random.default_rng(
            self.random.randint(0, 2**31 - 1)).beta(m * k, (1 - m) * k))
        lo, hi = cfg.appeal_bounds
        return min(hi, max(lo, val))

    def _apply_pending_capacity(self) -> None:
        """
        Rende effettive le espansioni o riduzioni di capacita' deliberate in
        passato. Il ritardo non e' un dettaglio implementativo: e' il tempo
        industriale reale fra la decisione e la carta stampata, ed e' cio' che
        obbliga a decidere sulla base di aspettative invece che sullo stato
        corrente del mercato.
        """
        if not self._pending_capacity:
            return
        due = [(t, v) for t, v in self._pending_capacity if t <= self.steps]
        if due:
            self.capacity_multiplier = due[-1][1]
            self._pending_capacity = [(t, v) for t, v in self._pending_capacity
                                      if t > self.steps]

    def _liquidity_shocks(self) -> None:
        """Estrae gli agenti che entrano in stato di bisogno di contante."""
        for agent in self.agents:
            agent.maybe_liquidity_shock()

    def per_release_capacity(self) -> int:
        """Unita' stampabili per release. L'azienda satura sempre la capacita'."""
        annual = self.cfg.capacity_per_year * self.capacity_multiplier
        per_release = annual * self.release_interval / TICKS_PER_YEAR
        return max(1, int(round(per_release)))

    def _monthly_income(self) -> None:
        if self.steps % TICKS_PER_MONTH == 0:
            self.agents.do("receive_income")

    # ------------------------------------------------------------------ #
    def _maybe_release(self) -> None:
        if self.steps < self._next_release_tick:
            return

        # Prima di stampare il nuovo set si valuta com'e' andato il precedente:
        # e' il feedback che lega sovraccarico produttivo e desiderabilita'.
        if self.latest_product is not None:
            cfg = self.cfg
            prev = self.latest_product
            leftover = prev.stock / max(1, prev.initial_stock)
            sold_through = 1.0 - leftover
            a = cfg.scarcity_memory
            self.scarcity_signal = a * self.scarcity_signal + (1 - a) * sold_through

            # Capitale di marca: si erode con l'invenduto oltre la soglia
            # fisiologica e si ricostruisce lentamente a mercato sano.
            # L'asimmetria fra i due tassi e' voluta: la reputazione di
            # scarsita' si perde in una stagione e si riguadagna in anni.
            excess = max(0.0, leftover - cfg.brand_unsold_tolerance)
            self.brand_equity -= cfg.brand_damage * excess
            if excess == 0.0:
                self.brand_equity += cfg.brand_recovery * (1.0 - self.brand_equity)
            self.brand_equity = min(1.0, max(cfg.brand_floor, self.brand_equity))

        # La capacita' della finestra si ripartisce fra prodotto nuovo e
        # ristampe: cio' che va all'uno viene tolto all'altro.
        total = self.per_release_capacity()
        self._restock_budget = int(round(total * self.restock_share))
        units = max(1, total - self._restock_budget)
        self.current_msrp = self.cfg.msrp * self.msrp_multiplier

        appeal = self._draw_appeal()
        product = Product(pid=self._next_pid,
                          release_tick=self.steps,
                          msrp_at_release=self.current_msrp,
                          initial_stock=units,
                          appeal=appeal)
        product.company_take = self.cfg.channel_wholesale_ratio * self.current_msrp
        self._next_pid += 1
        self.products.append(product)
        self._product_index[product.pid] = product
        self.latest_product = product
        self._next_release_tick = self.steps + self.release_interval

        # Seeding del cascade, identico nella logica a Tornberg (2018):
        # si attiva UN nodo seme dentro l'echo chamber e i suoi vicini, non
        # l'intero cluster. E' questa localizzazione che rende la coesione
        # interna del cluster una variabile causale: se il seme dovesse gia'
        # attivare tutto il cluster, P_n non avrebbe piu' nulla da spiegare.
        # A questo si somma uno shock globale debole (la campagna marketing,
        # che raggiunge tutti ma non convince nessuno da sola).
        # Lo shock di lancio e' proporzionale all'appetibilita': un set
        # dimenticabile non accende nessuna cascata, per quanto sia nuovo.
        w = self.cfg.appeal_hype_weight
        shock = self.cfg.release_hype_shock * ((1.0 - w * 0.5) + w * appeal)
        shock = max(0.0, shock)
        seed_node = self.random.choice(sorted(self.cluster))
        seeded = {seed_node, *self.graph.neighbors(seed_node)}
        for inf in self.influencers:
            seeded.add(inf.node_id)          # gli influencer sono sempre informati

        for agent in self.agents:
            if agent.node_id in seeded:
                agent.hype = min(1.0, agent.hype + shock)
            else:
                agent.hype = min(1.0, agent.hype + self.cfg.outside_shock_ratio * shock)
            agent.bought_this_release.pop(product.pid, None)

    # ------------------------------------------------------------------ #
    def market_price_of(self, product) -> float:
        """Prezzo di mercato corrente: lo scambio piu' recente, o il listino."""
        if product.last_traded is not None:
            return product.last_traded
        ask = self.book.best_ask(product.pid)
        return ask.price if ask else product.price

    def _do_restocks(self) -> None:
        """
        Alloca la quota di capacita' destinata alle ristampe.

        Si ristampa solo cio' che e' ANCORA IN STAMPA (entro i due anni) e che
        tira davvero (premium sopra soglia), dando priorita' al set piu' caldo.
        Le unita' entrano al prezzo di mercato corrente: la ristampa non
        riporta il set a listino, altrimenti ne distruggerebbe l'appeal.
        """
        budget = getattr(self, "_restock_budget", 0)
        if budget <= 0:
            return

        cfg = self.cfg
        candidates = [p for p in self.products
                      if p.in_print(self.steps, cfg.print_window_ticks)
                      and p.pid != (self.latest_product.pid if self.latest_product else -1)
                      and p.premium >= cfg.restock_min_premium]
        if not candidates:
            return

        candidates.sort(key=lambda p: p.premium, reverse=True)
        share = max(1, budget // len(candidates[:3]))
        for product in candidates[:3]:
            if budget <= 0:
                break
            units = min(share, budget)
            product.restock(units, self.market_price_of(product),
                            cfg.restock_price_ratio)
            budget -= units
        self._restock_budget = budget

    def _update_clearance(self) -> None:
        """Ribassa il prezzo dell'eccedenza rimasta a magazzino."""
        for p in self.active_products():
            p.update_clearance(self.steps, self.cfg.primary_window, self.cfg)

    def _primary_market(self) -> None:
        """
        Allocazione FCFS.

        La coda e' ordinata per una `priorita' di arrivo` = combinazione di
        hype, intento speculativo e rumore. Non e' un ordinamento arbitrario:
        rappresenta il fatto che chi monitora i drop (bot, alert Discord,
        preordini) arriva sistematicamente prima del collezionista occasionale.
        Il rumore garantisce che la coda non sia deterministica.
        """
        on_shelf = [p for p in self.active_products() if p.is_available()]
        if not on_shelf:
            return

        cfg, agents = self.cfg, list(self.agents)
        priority = {}
        for a in agents:
            priority[a.unique_id] = (cfg.queue_w_hype * a.hype
                                     + cfg.queue_w_resell * a.resell_propensity
                                     + cfg.queue_w_noise * self.random.random())
        queue = sorted(agents, key=lambda a: priority[a.unique_id], reverse=True)

        on_shelf.sort(key=lambda p: p.price)
        for product in on_shelf:
            if product.stock <= 0:
                continue
            sold = 0
            for agent in queue:
                if product.stock <= 0:
                    break
                sold += agent.attempt_primary_purchase(product)
            if sold:
                # Si registra l'incasso EFFETTIVO dell'azienda per unita': varia
                # fra prodotto nuovo (prezzo all'ingrosso ordinario), ristampa
                # (80% del mercato) e svendita (frazione del listino).
                take = product.company_take
                if product.price < product.msrp_at_release:
                    take = self.cfg.channel_wholesale_ratio * product.price
                self._primary_sales_log.append((self.steps, sold, take))

    # ------------------------------------------------------------------ #
    def _diffuse_hype(self) -> None:
        self.agents.shuffle_do("diffuse_hype")

        signal = self._influencer_signal()
        if signal == 0.0:
            return
        power = self.cfg.influencer_power
        for inf in self.influencers:
            inf.hype = min(1.0, max(0.0, inf.hype + 0.5 * signal * inf.hype_step))
            for follower in self.influencer_audience[inf.node_id]:
                follower.receive_influencer_signal(signal, power)

    def _influencer_signal(self) -> float:
        """
        Regola dell'influencer.

        pump  (+1): nella finestra attorno alla release, quando conviene
                    generare traffico sul contenuto;
        dump  (-1): quando il premium e' cosi' alto che conviene monetizzare
                    l'inventario;
        0        : altrimenti.
        """
        if self.latest_product is None:
            return 0.0

        lo, hi = self.cfg.influencer_pump_window
        delta = self.steps - self.latest_product.release_tick
        if lo <= delta <= hi:
            # Un influencer non promuove un set che nessuno vuole vedere: la
            # copertura segue l'appetibilita'. E' il canale attraverso cui una
            # release deliberatamente debole non riceve amplificazione.
            if self.random.random() < self.latest_product.appeal:
                return 1.0
            return 0.0

        prem = metrics.premium_index(self.active_products(), self.steps,
                                     order_book=self.book)
        holds = any(inf.n_units > 0 for inf in self.influencers)
        if prem > self.cfg.influencer_dump_premium and holds:
            return -1.0
        return 0.0

    # ------------------------------------------------------------------ #
    def _post_listings(self) -> None:
        """I reseller decidono se e a quanto quotare."""
        products = {p.pid: p for p in self.active_products()}
        for agent in self.agents:
            if not agent.is_reseller or not agent.inventory:
                continue
            p_list = agent.listing_probability()
            if p_list <= 0 or self.random.random() > p_list:
                continue

            owned = [pid for pid in agent.inventory if pid in products]
            if not owned:
                continue
            pid = self.random.choice(owned)
            product = products[pid]

            # Non quota piu' copie di quante ne possieda gia' a book.
            listed = sum(1 for a in self.book._asks.get(pid, ())
                         if a.seller_id == agent.unique_id)
            if listed >= len(agent.inventory[pid]):
                continue

            best = self.book.best_ask(pid)
            price = agent.ask_price(product,
                                    best.price if best else None,
                                    product.last_traded)
            self.book.add_ask(pid, agent.unique_id, price, self.steps)

    # ------------------------------------------------------------------ #
    def _secondary_trading(self) -> None:
        """
        Ogni agente prova un acquisto per tick sul secondario.

        Regola: guarda l'offerta piu' economica su tutto il mercato e la
        colpisce se soddisfa TRE vincoli simultanei
          (a) prezzo <= budget disponibile
          (b) prezzo <= willingness-to-pay = MSRP * (1 + elasticita' * hype)
          (c) prezzo <= ultimo scambiato * max_over_last_traded
        Il vincolo (c) e' l'ancoraggio: nessuno accetta di pagare molto piu'
        dell'ultimo prezzo osservato, ed e' cio' che impedisce ai prezzi di
        esplodere istantaneamente e li costringe a salire per gradini.
        """
        products = {p.pid: p for p in self.active_products()}
        by_uid = {a.unique_id: a for a in self.agents}
        cfg = self.cfg

        buyers = list(self.agents)
        self.random.shuffle(buyers)

        for buyer in buyers:
            ask = self.book.cheapest_overall(set(products))
            if ask is None:
                break
            if ask.seller_id == buyer.unique_id:
                continue

            product = products[ask.product_id]

            # I collezionisti puri non ricomprano cio' che hanno gia'.
            if buyer.is_pure_collector and buyer.was_served(product.pid):
                continue
            if buyer.hype < 0.10:
                continue

            wtp = buyer.willingness_to_pay(product)
            anchor = (product.last_traded * buyer.price_anchor_tolerance()
                      if product.last_traded is not None else float("inf"))

            if ask.price > min(wtp, anchor, buyer.budget):
                continue

            # --- esecuzione ---
            self.book.pop_best(product.pid)
            seller = by_uid.get(ask.seller_id)
            if seller is None or not seller.owns(product.pid):
                continue
            seller.register_sale(product.pid, ask.price, cfg.marketplace_fee)
            buyer.register_purchase(product.pid, ask.price)
            product.record_trade(ask.price, self.steps)
            self.tick_volume += 1

    # ------------------------------------------------------------------ #
    def _open_boxes(self) -> None:
        """
        Fase di distruzione dell'offerta.

        Le box aperte spariscono: non tornano sul secondario, non possono
        essere quotate, non contano piu' nel circolante. Chi le ha aperte
        resta comunque "servito" ai fini dell'accessibilita'. Poiche' un
        annuncio potrebbe riferirsi a una copia appena aperta, gli ordini
        orfani vengono ripuliti subito dopo.
        """
        for agent in self.agents:
            n = agent.open_boxes()
            if n:
                self.tick_opened += n
                self.book.prune_orphans(agent.unique_id, agent.inventory)

    def _collect(self) -> None:
        self.history.append(metrics.snapshot(self))

    # ================================================================== #
    # Interfaccia del super-agente (tier data-driven)                    #
    # ================================================================== #
    def apply_policy_action(self, action: int) -> None:
        """
        Applica un'azione della Pokemon Company.

        Le azioni su offerta, prezzo e cadenza agiscono sulle release FUTURE:
        una tiratura gia' stampata non si puo' cambiare. L'unica eccezione e'
        `reprint`, che aggiunge stock a un prodotto gia' uscito ed e' quindi
        l'unico strumento davvero reattivo (e ha un cooldown).
        """
        cfg = self.cfg
        name = ACTIONS[int(action)]
        self.action_log.append((self.steps, int(action)))

        if name == "capacity_up":
            lo, hi = cfg.capacity_bounds
            self.target_capacity = min(hi, self.target_capacity * (1 + cfg.capacity_step))
            self._pending_capacity.append(
                (self.steps + cfg.capacity_lag_ticks, self.target_capacity))

        elif name == "capacity_down":
            lo, hi = cfg.capacity_bounds
            self.target_capacity = max(lo, self.target_capacity * (1 - cfg.capacity_step))
            self._pending_capacity.append(
                (self.steps + cfg.capacity_lag_ticks, self.target_capacity))

        elif name == "msrp_up":
            lo, hi = cfg.msrp_multiplier_bounds
            self.msrp_multiplier = min(hi, self.msrp_multiplier * (1 + cfg.msrp_step))

        elif name == "msrp_down":
            lo, hi = cfg.msrp_multiplier_bounds
            self.msrp_multiplier = max(lo, self.msrp_multiplier * (1 - cfg.msrp_step))

        elif name == "cadence_faster":
            lo, hi = cfg.cadence_bounds
            self.release_interval = max(lo, self.release_interval - cfg.cadence_step)
            self._next_release_tick = min(self._next_release_tick,
                                          self.steps + self.release_interval)

        elif name == "cadence_slower":
            lo, hi = cfg.cadence_bounds
            self.release_interval = min(hi, self.release_interval + cfg.cadence_step)

        elif name == "restock_more":
            lo, hi = cfg.restock_share_bounds
            self.restock_share = min(hi, self.restock_share + cfg.restock_share_step)

        elif name == "restock_less":
            lo, hi = cfg.restock_share_bounds
            self.restock_share = max(lo, self.restock_share - cfg.restock_share_step)

        elif name == "appeal_up":
            lo, hi = cfg.appeal_bounds
            self.appeal_target = min(hi, self.appeal_target + cfg.appeal_step)
            self._pending_appeal.append(
                (self.steps + cfg.appeal_lag_ticks, self.appeal_target))

        elif name == "appeal_down":
            lo, hi = cfg.appeal_bounds
            self.appeal_target = max(lo, self.appeal_target - cfg.appeal_step)
            self._pending_appeal.append(
                (self.steps + cfg.appeal_lag_ticks, self.appeal_target))

        # "no_op": osserva senza intervenire. E' un'azione a pieno titolo:
        # con una capacita' fissa e un ritardo di un anno e mezzo, l'astensione
        # e' spesso la scelta corretta.

    # ================================================================== #
    # Reporter di stato (equivalenti a NetlogoCommands.get_*)            #
    # ================================================================== #
    def observation(self) -> np.ndarray:
        """
        Stato osservato dal super-agente. 6 componenti, tutte in [0,1].

        s1  premium normalizzato   <- analogo di `global cascade`
        s2  hype medio             <- analogo di `mean opinion`
        s3  invenduto              (costo per l'azienda)
        s4  accessibilita'         (obiettivo dichiarato)
        s5  cattura dei reseller   (chi si e' preso la tiratura)
        s6  hype dell'influencer   <- analogo di `max page-rank fra i contagiati`
        s7  capitale di marca      (erosione di lungo periodo del sovraccarico)
        s8  appetibilita' media    (quanto tirano i set attualmente in circolo)

        La settima componente e' indispensabile: senza un segnale esplicito di
        saturazione il super-agente non puo' distinguere "prezzi bassi perche'
        il mercato e' sano" da "prezzi bassi perche' l'ho inondato di prodotto".
        """
        cfg = self.cfg
        snap = self.history[-1]
        prem = snap["premium_index"]
        prem_norm = np.clip((prem - 1.0) / max(1e-9, cfg.premium_cap_for_norm - 1.0),
                            0.0, 1.0)
        return np.array([
            prem_norm,
            snap["mean_hype"],
            snap["unsold_ratio"],
            snap["access_latest"],
            snap["reseller_capture"],
            snap["influencer_hype"],
            snap["brand_equity"],
            snap["mean_appeal"],
        ], dtype=np.float32)

    def premium(self) -> float:
        return self.history[-1]["premium_index"]

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.history)

    def run(self, ticks: int | None = None):
        """Esegue la simulazione senza super-agente (baseline)."""
        ticks = ticks or self.cfg.total_ticks
        for _ in range(ticks):
            self.step()
        return self.to_dataframe()

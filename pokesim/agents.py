"""
agents.py — I "basic agent" del modello: i partecipanti al mercato.

Mappatura rispetto a Zaccagnino et al. (2025):

    nodo con activation threshold  ->  partecipante con soglia di contagio
    opinione (fake/neutro/true)    ->  hype in [0,1]
    op_step                        ->  hype_step (rigidita' della convinzione)
    is-fake-active                 ->  hype >= hype_active_threshold
    super-agente che persuade      ->  Pokemon Company che modula offerta e prezzo

Tre archetipi
-------------
La popolazione non e' omogenea, e la distinzione conta perche' i tre tipi
reagiscono a segnali diversi e lasciano tracce diverse sull'offerta:

* **fan** - compra per possedere. Deriva utilita' dal prodotto in se', quindi
  la sua willingness-to-pay e' guidata dall'hype (valore di consumo). Ne apre
  una parte: quelle box escono per sempre dal circolante.
* **fan-flipper** - fan che rivende una quota di cio' che prende. Apre meno,
  perche' il sigillato e' cio' che ha valore di scambio.
* **investitore puro** - non apre nulla, non e' un appassionato. La sua WTP e'
  finanziaria: prezzo atteso futuro scontato di un margine richiesto. Segue il
  momentum, non il fandom.

Il pozzo di distruzione (`open_rate`) e' strutturalmente importante: e' l'unico
meccanismo che *riduce* l'offerta nel tempo, e crea una retroazione negativa
sui prezzi (piu' il sigillato vale, meno se ne apre, piu' sigillato resta in
circolo, meno i prezzi salgono).
"""

from __future__ import annotations

from mesa import Agent


class CollectorAgent(Agent):
    """Un partecipante al mercato del collezionismo."""

    def __init__(self, model, node_id: int, income: float, threshold: float,
                 resell_propensity: float, in_echo_chamber: bool,
                 is_fan: bool, open_rate: float, liquidity_prob: float):
        super().__init__(model)
        cfg = model.cfg

        self.node_id = node_id
        self.in_echo_chamber = in_echo_chamber
        self.is_influencer = False

        # --- tipo ---
        self.is_fan = is_fan
        self.open_rate = open_rate          # 0 per gli investitori puri

        # --- economia ---
        self.income = income
        self.budget = income * cfg.initial_budget_months
        self.spent_primary = 0.0
        self.spent_secondary = 0.0
        self.revenue_secondary = 0.0

        # --- comportamento ---
        self.threshold = threshold
        self.hype = cfg.hype_initial
        self.hype_step = cfg.hype_step
        self.resell_propensity = resell_propensity

        # Boost temporaneo indotto dall'influencer sulla tolleranza al prezzo.
        # Decade da solo: e' l'effetto di un video, non un tratto stabile.
        self.influence_boost = 0.0

        # Shock di liquidita': quando scatta, l'agente liquida tutto a sconto
        # per fare cassa in fretta, indipendentemente da prezzo di carico e
        # aspettative. E' offerta scorrelata dal ciclo dell'hype.
        self.liquidity_shock_until = -1
        # Probabilita' individuale di aver bisogno di contante: dipende dalla
        # solidita' finanziaria, non e' uniforme sulla popolazione.
        self.liquidity_prob = liquidity_prob

        # --- inventario ---
        # `inventory`   : sigillato, in circolo, potenzialmente vendibile
        # `pending_open`: copie destinate all'apertura, gia' fuori dal mercato
        # La destinazione e' decisa all'acquisto con probabilita' `open_rate`:
        # e' cosi' che il parametro assume il significato dichiarato, cioe'
        # la QUOTA di prodotto che verra' aperta e distrutta. Il tasso
        # `open_hazard` governa soltanto *quando*, non *quanto*.
        self.inventory: dict[int, list[float]] = {}
        self.pending_open: dict[int, list[float]] = {}
        self.bought_this_release: dict[int, int] = {}
        self.opened_pids: set[int] = set()

        # --- statistiche individuali ---
        self.units_bought_primary = 0
        self.units_bought_secondary = 0
        self.units_sold = 0
        self.units_opened = 0

    # ------------------------------------------------------------------ #
    # Proprieta'                                                         #
    # ------------------------------------------------------------------ #
    @property
    def is_investor(self) -> bool:
        return not self.is_fan

    @property
    def is_reseller(self) -> bool:
        return self.resell_propensity > 0.0

    @property
    def is_pure_collector(self) -> bool:
        """Fan che non rivende mai: e' il soggetto che la policy vuole servire."""
        return self.is_fan and self.resell_propensity == 0.0

    @property
    def in_liquidity_shock(self) -> bool:
        return self.model.steps <= self.liquidity_shock_until

    def avg_cost(self, product_id: int) -> float | None:
        """Prezzo medio di carico sulle copie possedute di quel set."""
        lots = list(self.inventory.get(product_id, ())) + \
               list(self.pending_open.get(product_id, ()))
        return sum(lots) / len(lots) if lots else None

    @property
    def is_hype_active(self) -> bool:
        return self.hype >= self.model.cfg.hype_active_threshold

    @property
    def n_units(self) -> int:
        """Solo il sigillato in circolo: cio' che puo' ancora essere venduto."""
        return sum(len(v) for v in self.inventory.values())

    @property
    def n_held(self) -> int:
        """Tutto cio' che detiene, comprese le copie destinate all'apertura."""
        return self.n_units + sum(len(v) for v in self.pending_open.values())

    def owns(self, product_id: int) -> bool:
        return len(self.inventory.get(product_id, ())) > 0

    def was_served(self, product_id: int) -> bool:
        """
        Ha ottenuto il set, che lo possieda ancora sigillato o lo abbia aperto.
        L'accessibilita' va misurata cosi': un fan che ha comprato la box e
        l'ha aperta E' stato servito, anche se non ha piu' nulla in inventario.
        """
        return (self.owns(product_id)
                or product_id in self.opened_pids
                or bool(self.pending_open.get(product_id)))

    # ------------------------------------------------------------------ #
    # Ciclo economico                                                    #
    # ------------------------------------------------------------------ #
    def receive_income(self) -> None:
        self.budget += self.income

    def maybe_liquidity_shock(self) -> bool:
        """Estrae l'evento 'ho bisogno di contante adesso'."""
        cfg = self.model.cfg
        if self.in_liquidity_shock or self.n_units == 0:
            return False
        if self.model.random.random() < self.liquidity_prob:
            self.liquidity_shock_until = self.model.steps + cfg.liquidity_shock_duration
            return True
        return False

    # ------------------------------------------------------------------ #
    # Diffusione dell'hype: contagio complesso                           #
    # ------------------------------------------------------------------ #
    def diffuse_hype(self) -> None:
        """
        Regola di attivazione, nella forma dei due paper di riferimento:

            |vicini attivi| / |vicini| > theta_a  =>  hype += hype_step
            altrimenti                            =>  hype -= hype_decay

        con tre termini aggiuntivi che un mercato richiede e una cascata
        una tantum no:

        1. **novita'**: il guadagno e' pesato dalla freschezza del set;
        2. **saturazione**: e' pesato anche dall'appeal da scarsita'. Un
           mercato inondato di prodotto genera meno entusiasmo, ma il termine
           ha un pavimento (`scarcity_floor`): la domanda si comprime, non
           sparisce. Qualcuno compra Pokemon in ogni condizione di mercato;
        3. **frustrazione**: se il prezzo eccede la propria WTP l'agente si
           disaffeziona. E' il freno endogeno alla bolla.
        """
        cfg = self.model.cfg
        base = cfg.hype_baseline
        novelty = self.model.novelty
        appeal = self.model.scarcity_appeal

        decay = cfg.hype_decay * (1.0 + cfg.attention_decay_strength * (1.0 - novelty))

        # Il decadimento e' INCONDIZIONATO e il contagio si somma sopra.
        # Applicarlo solo nel ramo "else" rende quel ramo irraggiungibile
        # appena tutti i vicini sono attivi: l'hype salirebbe a 1 senza mai
        # tornare indietro. Qui vince il termine piu' forte e il segno del
        # saldo si inverte da solo quando la novita' si esaurisce.
        h = self.hype - decay
        neighbors = self.model.neighbors_of(self.node_id)
        if neighbors:
            frac_active = sum(1 for a in neighbors if a.is_hype_active) / len(neighbors)
            if frac_active > self.threshold:
                h += self.hype_step * novelty * appeal
        self.hype = min(1.0, max(base, h))

        excess = self.model.price_excess_over(self)
        if excess > 0:
            self.hype = max(base, self.hype - cfg.frustration_strength * min(2.0, excess))

        # L'effetto di un contenuto virale si esaurisce da solo
        self.influence_boost *= cfg.influencer_boost_decay

    def receive_influencer_signal(self, signal: float, power: float) -> None:
        """
        Spinta dell'influencer sui propri follower.

        Agisce su DUE canali, e il secondo e' quello che conta davvero:

        1. sull'hype (desiderabilita' percepita);
        2. sulla **tolleranza al prezzo** (`influence_boost`), cioe' sul
           giudizio "a questa cifra non lo compro". E' il meccanismo con cui
           un promotore sposta l'asticella di chi riteneva il mercato troppo
           caro: non convince che il prodotto sia piu' bello, convince che il
           prezzo corrente sia giustificato. Senza questo canale l'influencer
           resta ininfluente proprio dove il suo effetto e' piu' visibile.
        """
        if signal == 0.0:
            return
        if self.model.random.random() >= power:
            return
        self.hype = min(1.0, max(0.0, self.hype + signal * self.hype_step))
        self.influence_boost = min(1.0, max(0.0, self.influence_boost + signal * 0.5))

    # ------------------------------------------------------------------ #
    # Apertura delle box: il pozzo di distruzione dell'offerta           #
    # ------------------------------------------------------------------ #
    def _route_units(self, product_id: int, prices: list[float]) -> None:
        """
        Smista le copie appena acquistate fra sigillato e da-aprire, con
        probabilita' `open_rate`. Un investitore puro ha open_rate=0 e manda
        tutto al sigillato.
        """
        rnd = self.model.random.random
        for pr in prices:
            if self.open_rate > 0 and rnd() < self.open_rate:
                self.pending_open.setdefault(product_id, []).append(pr)
            else:
                self.inventory.setdefault(product_id, []).append(pr)

    def open_boxes(self) -> int:
        """
        Apre le copie destinate all'apertura. Sono DISTRUTTE: non tornano sul
        secondario, non possono essere quotate, escono dal circolante.

        Un ripensamento e' possibile: se il sigillato ha preso troppo valore,
        una copia destinata all'apertura puo' essere dirottata sul mercato
        ("e' un peccato aprirla"). E' la retroazione negativa che lega prezzi
        e distruzione dell'offerta: prezzi alti -> si apre meno -> resta piu'
        sigillato in circolo -> i prezzi si calmierano.
        """
        if not self.pending_open:
            return 0

        cfg = self.model.cfg
        rnd = self.model.random.random
        opened = 0

        for pid in list(self.pending_open):
            product = self.model.product_by_id(pid)
            if product is None:
                continue
            greed = min(cfg.max_reconsider_prob,
                        cfg.open_price_sensitivity * max(0.0, product.premium - 1.0))

            keep = []
            for price in self.pending_open[pid]:
                if rnd() < greed:                      # ripensamento: la rivende
                    self.inventory.setdefault(pid, []).append(price)
                elif rnd() < cfg.open_hazard:          # la apre: distrutta
                    opened += 1
                    self.units_opened += 1
                    self.opened_pids.add(pid)
                    product.units_opened += 1
                else:
                    keep.append(price)
            if keep:
                self.pending_open[pid] = keep
            else:
                del self.pending_open[pid]
        return opened

    # ------------------------------------------------------------------ #
    # Valutazione: fan e investitori usano metriche diverse              #
    # ------------------------------------------------------------------ #
    def perceived_desirability(self, product) -> float:
        """
        Desiderabilita' del singolo set per questo agente: combina l'hype
        personale (quanto e' caldo il momento) con l'appetibilita' intrinseca
        del prodotto (quanto quel set specifico vale la pena). Un set poco
        appetibile non si vende nemmeno in piena euforia, e uno molto
        appetibile trova compratori anche a mercato freddo.
        """
        w = self.model.cfg.appeal_wtp_weight
        return self.hype * ((1.0 - w) + w * 2.0 * product.appeal)

    def willingness_to_pay(self, product) -> float:
        """
        Prezzo massimo che l'agente e' disposto a pagare.

        * **fan**: valore di consumo. WTP = MSRP * (1 + elasticita' * hype),
          amplificata dal boost dell'influencer.
        * **investitore**: valore finanziario. WTP = prezzo atteso futuro
          scontato del margine richiesto. Il prezzo atteso e' estrapolato dal
          momentum recente: se il mercato sale e' disposto a pagare di piu'
          oggi. E' cio' che rende gli investitori pro-ciclici.
        """
        cfg = self.model.cfg
        boost = 1.0 + cfg.influencer_wtp_boost * self.influence_boost

        if self.is_fan:
            return (product.msrp_at_release
                    * (1.0 + cfg.hype_price_elasticity
                       * self.perceived_desirability(product)) * boost)

        # L'investitore ancora la propria valutazione al prezzo di carico, non
        # al trend: compra sotto media, non sopra. Il tetto resta legato al
        # prezzo di mercato per non pagare piu' del dovuto quando il book e'
        # sottile.
        ref = self.avg_cost(product.pid) or product.msrp_at_release
        market = product.last_traded or product.price
        return max(ref, market) * boost / max(1e-9, cfg.investor_required_margin) ** 0

    def wants_to_buy_at(self, product, price: float) -> bool:
        """
        Filtro di mediazione del prezzo di carico, applicato a chi ha un
        movente speculativo (investitori e flipper).

        La regola non e' l'inseguimento del momentum ma il suo contrario: si
        compra quando il prezzo scende SOTTO il proprio carico medio, per
        abbassarlo; si e' riluttanti quando il prezzo e' salito sopra. Chi non
        ha ancora posizione usa l'MSRP come riferimento. L'effetto aggregato
        e' duplice: freno alle bolle (nessuno insegue i rialzi) e sostegno
        nelle discese (si accumula sui ribassi).
        """
        if self.resell_propensity <= 0:
            return True                     # il collezionista puro non specula

        cfg = self.model.cfg
        rnd = self.model.random.random
        ref = self.avg_cost(product.pid) or product.msrp_at_release

        if price <= ref * (1.0 - cfg.cost_averaging_gap):
            return True                     # media al ribasso: e' l'occasione
        if price <= ref:
            return rnd() < cfg.investor_flat_buy_prob
        return rnd() < cfg.investor_chase_prob

    def price_anchor_tolerance(self) -> float:
        """
        Quanto sopra l'ultimo prezzo scambiato l'agente accetta di spingersi.
        L'influencer alza proprio questa soglia: e' l'ancoraggio, non il gusto,
        cio' che blocca gli acquisti quando i prezzi corrono.
        """
        cfg = self.model.cfg
        return cfg.max_over_last_traded + cfg.influencer_anchor_boost * self.influence_boost

    # ------------------------------------------------------------------ #
    # Strategia di acquisto sul primario                                 #
    # ------------------------------------------------------------------ #
    def desired_primary_qty(self, product) -> int:
        """Quante copie vorrebbe al lancio, a MSRP."""
        cfg = self.model.cfg
        rnd = self.model.random.random

        if self.is_fan:
            base = 0
            # Domanda di fondo: una quota di appassionati compra il set nuovo
            # a prezzo di listino a prescindere dall'hype del momento. E' il
            # pavimento della domanda: non va mai a zero, nemmeno a mercato
            # completamente saturo.
            bargain = product.price < cfg.bargain_threshold * product.msrp_at_release
            gate = cfg.core_demand + (cfg.bargain_demand_boost if bargain else 0.0)
            if self.perceived_desirability(product) > cfg.fan_hype_gate or rnd() < gate:
                base = cfg.collector_desired_qty
            if self.resell_propensity > 0 and self.wants_to_buy_at(product, product.price):
                base += int(round(cfg.fan_flipper_max_qty
                                  * self.resell_propensity
                                  * (0.3 + 0.7 * self.perceived_desirability(product))))
            elif self.was_served(product.pid):
                return 0                      # collezionista puro: gli basta una
            return base

        # Investitore: entra solo se supera il filtro di mediazione del carico
        # e se il prezzo resta sotto la propria valutazione.
        if not self.wants_to_buy_at(product, product.price):
            return 0
        if self.willingness_to_pay(product) <= product.price:
            return 0
        scale = self.resell_propensity * (0.3 + 0.7 * self.perceived_desirability(product))
        return max(1, int(round(cfg.investor_max_qty * scale)))

    def attempt_primary_purchase(self, product, cap: int | None = None) -> int:
        """Acquisto sul mercato primario in regime FCFS. Ritorna le unita' prese."""
        if product.stock <= 0:
            return 0
        desired = self.desired_primary_qty(product)
        if desired <= 0:
            return 0
        if cap is not None:
            already = self.bought_this_release.get(product.pid, 0)
            desired = min(desired, max(0, cap - already))

        qty = min(desired, int(self.budget // product.price), product.stock)
        if qty <= 0:
            return 0

        cost = qty * product.price
        self.budget -= cost
        self.spent_primary += cost
        self._route_units(product.pid, [product.price] * qty)
        self.bought_this_release[product.pid] = \
            self.bought_this_release.get(product.pid, 0) + qty
        self.units_bought_primary += qty

        product.stock -= qty
        product.units_sold_primary += qty
        if self.is_investor:
            product.units_to_investors += qty
        if self.is_reseller:
            product.units_to_resellers += qty
        return qty

    # ------------------------------------------------------------------ #
    # Strategia di rivendita                                             #
    # ------------------------------------------------------------------ #
    def listing_probability(self) -> float:
        """
        Probabilita' di quotare in questo tick.

        * fan-flipper: l'hype alto NON incentiva a vendere, incentiva a tenere
          (hoarding). E' il meccanismo che restringe l'offerta secondaria
          proprio durante le bolle;
        * investitore: monetizza quando il momentum gira. Entra in salita ed
          esce sul cambio di trend, il che amplifica entrambe le fasi.
        """
        cfg = self.model.cfg
        if self.in_liquidity_shock:
            return 1.0            # deve fare cassa: quota tutto, subito
        if self.resell_propensity <= 0:
            return 0.0

        if self.is_fan:
            return self.resell_propensity * (1.0 - cfg.hoarding_strength * self.hype)

        prod = self.model.latest_product
        momentum = (prod.momentum(self.model.steps, cfg.investor_momentum_window)
                    if prod else 0.0)
        if momentum < 0:
            return self.resell_propensity
        return self.resell_propensity * cfg.investor_hold_factor

    def ask_price(self, product, best_ask: float | None,
                  last_traded: float | None) -> float:
        """
        Per vendere in fretta si taglia del 5% il miglior ask presente; se il
        book e' vuoto ci si ancora all'ultimo scambiato o, senza storico, si
        parte da un prezzo alto. Non si vende mai sotto MSRP*(1+min_margin).
        """
        cfg = self.model.cfg

        # Chi ha bisogno urgente di contante svende: taglia sul prezzo di
        # mercato per liquidare in fretta e ignora sia il prezzo di carico sia
        # il pavimento sul margine.
        if self.in_liquidity_shock:
            lo, hi = cfg.liquidity_sale_ratio
            ref = best_ask or last_traded or (product.msrp_at_release * cfg.no_market_markup)
            return ref * self.model.random.uniform(lo, hi)

        floor = product.msrp_at_release * (1.0 + cfg.min_margin)
        # Se il negozio sta ancora svendendo, il rivenditore non puo' quotare
        # sopra quel prezzo: nessuno comprerebbe. Il pavimento sul margine
        # salta, e chi ha comprato a listino incassa la perdita. E' la
        # capitolazione che si osserva dopo ogni sovraccarico produttivo.
        if product.is_available():
            floor = min(floor, product.price * (1.0 - cfg.undercut))
        if best_ask is not None:
            price = best_ask * (1.0 - cfg.undercut)
        elif last_traded is not None:
            price = last_traded * (1.0 + cfg.undercut)
        else:
            price = product.msrp_at_release * cfg.no_market_markup
        return max(floor, price)

    # ------------------------------------------------------------------ #
    # Transazioni sul secondario                                         #
    # ------------------------------------------------------------------ #
    def register_sale(self, product_id: int, price: float, fee: float) -> None:
        lots = self.inventory.get(product_id)
        if lots:
            lots.pop()
            if not lots:
                del self.inventory[product_id]
        net = price * (1.0 - fee)
        self.budget += net
        self.revenue_secondary += net
        self.units_sold += 1

    def register_purchase(self, product_id: int, price: float) -> None:
        self.budget -= price
        self.spent_secondary += price
        self._route_units(product_id, [price])
        self.units_bought_secondary += 1

"""
config.py — Parametri centrali del sandbox regolatorio "Pokemon TCG scarcity market".

Tutti i parametri della simulazione vivono qui, in un'unica dataclass immutabile
(`SimConfig`). Questo replica il ruolo che in FakeNewsDetection avevano
`netlogo/simulation_parameters.py` + gli slider del file .nlogo, ma in forma
serializzabile: ogni esperimento salva la propria config in JSON accanto ai
risultati, cosi' le figure sono sempre riproducibili.

Convenzione temporale
---------------------
1 tick = 1 settimana.
TICKS_PER_MONTH = 4  -> il budget si ricarica ogni 4 tick (stipendio).
52 tick = 1 anno     -> `releases_per_year=6` => una release ogni ~9 tick.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, replace
from typing import Literal

TICKS_PER_MONTH = 4
TICKS_PER_YEAR = 52

NetworkType = Literal["erdos_renyi", "barabasi_albert", "watts_strogatz"]


@dataclass(frozen=True)
class SimConfig:
    # ------------------------------------------------------------------ #
    # 1. Rete sociale                                                    #
    # ------------------------------------------------------------------ #
    n_agents: int = 300
    network_type: NetworkType = "erdos_renyi"
    avg_degree: int = 8               # k  (come nei due paper: k = 8)
    ws_rewire_p: float = 0.1          # solo per watts_strogatz

    # Echo chamber (costruita esattamente come in Tornberg 2018 / Zaccagnino 2025)
    echo_chamber_fraction: float = 0.20   # c : quota di nodi nel cluster
    network_polarization: float = 0.40    # P_n : quota di archi esterni riallocati dentro
    opinion_polarization: float = 0.10    # P_o : riduzione della soglia dentro il cluster

    # ------------------------------------------------------------------ #
    # 2. Agenti (collezionisti / investitori)                            #
    # ------------------------------------------------------------------ #
    # Reddito mensile lognormale -> ricchi e poveri nella stessa rete.
    income_median: float = 320.0      # EUR/mese destinabili all'hobby (mediana)
    income_sigma: float = 0.75        # sigma della lognormale (disuguaglianza)
    initial_budget_months: float = 2.0  # cassa iniziale = N mesi di reddito

    # Soglia di attivazione del contagio complesso (theta).
    # E' l'analogo diretto della "credulita'" dei due paper.
    activation_threshold: float = 0.27
    threshold_jitter: float = 0.05    # eterogeneita' individuale (uniforme +/-)

    # Dinamica dell'hype
    hype_step: float = 0.10           # analogo di `opinion-metric-step`
    hype_decay: float = 0.04          # decadimento passivo per tick
    hype_active_threshold: float = 0.5  # sopra questa soglia il nodo e' "contagioso"
    hype_initial: float = 0.20
    hype_baseline: float = 0.05       # interesse residuo di fondo
    release_hype_shock: float = 0.55  # shock di hype iniettato nella echo chamber
    outside_shock_ratio: float = 0.10  # quota dello shock che arriva fuori dal cluster

    # Decadimento dell'attenzione (novelty).
    # Senza questo termine il contagio complesso satura a hype=1 e non torna
    # mai indietro: il mercato collassa su un plateau statico. La novita' di un
    # set si esaurisce, ed e' questo che genera i cicli boom-bust.
    attention_half_life: float = 6.0     # tick perche' la novita' si dimezzi
    attention_decay_strength: float = 1.2  # quanto accelera il decadimento

    # Frustrazione da esclusione: se il prezzo di mercato supera la propria
    # willingness-to-pay, l'agente si disinteressa. E' il freno endogeno alla
    # bolla, e agisce prima di qualunque intervento regolatorio.
    frustration_strength: float = 0.05

    # Erosione dell'appeal da scarsita' (overprinting).
    # Se il set precedente e' rimasto sugli scaffali, il successivo nasce
    # gia' meno desiderabile: la scarsita' percepita e' parte del valore.
    # E' il meccanismo che punisce il sovraccarico produttivo, ed e' cio' che
    # impedisce alla policy "stampa sempre di piu'" di essere banalmente
    # ottimale. Corrisponde a un fenomeno documentato nel collezionismo.
    scarcity_floor: float = 0.30      # guadagno di hype residuo a mercato saturo
    scarcity_memory: float = 0.5      # smoothing esponenziale del segnale
    unsold_window: int = 52           # finestra su cui si misura l'invenduto

    # --- Capitale di marca (danno di lungo periodo del sovraccarico) ---
    # Con margini lordi del 65%, sovrastampare resta profittevole nel breve
    # anche vendendone meta': e' precisamente il calcolo che ha portato al
    # sovraccarico reale del 2021-2023. Il costo arriva dopo, come erosione
    # della proposta di valore da collezione, e va su un orizzonte pluriennale.
    # `brand_equity` e' lenta a scendere e ancora piu' lenta a risalire: e'
    # questo, e non il conto economico del trimestre, a rendere non banale il
    # problema di controllo.
    brand_damage: float = 0.35        # erosione per unita' di invenduto oltre soglia
    brand_recovery: float = 0.04      # recupero per release a mercato sano
    brand_unsold_tolerance: float = 0.10  # invenduto fisiologico, non penalizzato
    brand_floor: float = 0.35         # pavimento: la domanda si comprime, non sparisce

    # Influencer
    n_influencers: int = 1
    influencer_centrality: Literal["betweenness", "pagerank", "degree"] = "betweenness"
    influencer_power: float = 0.25    # spinta diretta sui follower per tick
    # Un influencer NON raggiunge solo i propri vicini di grafo: ha un
    # pubblico. `influencer_reach` e' la quota della popolazione che lo segue,
    # assegnata al setup e stabile nel tempo (i vicini di grafo sono sempre
    # inclusi). Senza questo parametro il potere dell'influencer e' nullo per
    # costruzione, perche' k=8 vicini su N=300 non muovono nessuna aggregata.
    influencer_reach: float = 0.25
    influencer_pump_window: tuple[int, int] = (-2, 6)  # tick attorno alla release
    influencer_dump_premium: float = 2.5  # sopra questo premium l'influencer scarica
    # L'influencer non agisce solo sul desiderio ma sul GIUDIZIO DI PREZZO:
    # sposta l'asticella di chi riteneva il mercato troppo caro. Questi due
    # parametri sono il canale attraverso cui lo fa.
    influencer_wtp_boost: float = 0.45     # dilatazione della willingness-to-pay
    influencer_anchor_boost: float = 0.25  # dilatazione della tolleranza all'ancora
    influencer_boost_decay: float = 0.85   # decadimento per tick dell'effetto

    # ------------------------------------------------------------------ #
    # 2b. Composizione della popolazione                                 #
    # ------------------------------------------------------------------ #
    # Non tutti sono appassionati. Gli investitori puri non aprono mai nulla
    # e valutano il prodotto solo finanziariamente: sono un tipo di agente
    # distinto, non un fan piu' avido.
    investor_share: float = 0.25          # quota di investitori puri
    fan_flipper_share: float = 0.25       # quota DEI FAN che rivende anche

    reseller_propensity_mean: float = 0.6  # media della Beta sui fan-flipper
    investor_propensity_bounds: tuple[float, float] = (0.6, 1.0)
    hoarding_strength: float = 0.8    # quanto l'hype alto blocca la vendita (0..1)

    # --- Apertura delle box (pozzo di distruzione dell'offerta) ---
    # Quota del posseduto che un appassionato finisce per aprire. Le box
    # aperte sono distrutte: escono dal circolante e non tornano mai sul
    # secondario. E' l'unico meccanismo che riduce l'offerta nel tempo.
    open_rate_bounds: tuple[float, float] = (0.20, 0.80)
    open_hazard: float = 0.20         # velocita' di apertura (non la quota)
    open_price_sensitivity: float = 0.10  # quanto il sovrapprezzo fa ripensare
    max_reconsider_prob: float = 0.25  # tetto alla probabilita di ripensamento

    # --- Shock di liquidita' ---
    # Un agente puo' trovarsi ad avere bisogno urgente di contante: allora
    # liquida tutto a sconto sul mercato per vendere in fretta, ignorando
    # prezzo di carico e logiche di accumulo. E' una fonte di offerta
    # scorrelata dal ciclo dell'hype, e quindi un innesco possibile per le
    # inversioni.
    # La fragilita' finanziaria non e' uniforme: chi ha grandi disponibilita'
    # ha bisogno di liquidare molto di rado, chi ha budget limitati assai piu'
    # spesso. I due estremi sono espressi come intervallo medio fra due eventi.
    liquidity_years_rich: float = 12.0   # un evento ogni ~12 anni
    liquidity_years_poor: float = 3.0    # un evento ogni ~3 anni
    liquidity_shock_duration: int = 8
    liquidity_sale_ratio: tuple[float, float] = (0.85, 0.90)  # quota del prezzo di mercato

    # --- Mediazione del prezzo di carico (cost averaging) ---
    # Investitori e flipper non inseguono i rialzi: comprano quando il prezzo
    # scende SOTTO il proprio prezzo medio di carico, per abbassarlo. E'
    # l'opposto dell'inseguimento del momentum, e agisce da freno endogeno
    # alle bolle e da sostegno nelle discese.
    cost_averaging_gap: float = 0.10   # sconto richiesto sul carico per mediare
    investor_flat_buy_prob: float = 0.25   # acquisto in area di pareggio
    investor_chase_prob: float = 0.05      # quanto raramente insegue un rialzo

    # --- Comportamento degli investitori ---
    investor_momentum_window: int = 8     # tick su cui misurano il trend
    investor_momentum_weight: float = 1.0 # quanto estrapolano il trend
    investor_required_margin: float = 0.85  # compra sotto l'85% del prezzo atteso
    investor_hold_factor: float = 0.15    # quanto poco vende finche' il trend sale
    investor_max_qty: int = 8

    # --- Domanda di fondo dei fan ---
    fan_hype_gate: float = 0.15       # sopra questo hype il fan vuole il set
    core_demand: float = 0.12         # ...e sotto, compra comunque con questa prob.
    fan_flipper_max_qty: int = 4

    # Strategia d'acquisto
    hype_price_elasticity: float = 2.0  # WTP = MSRP * (1 + elasticity * hype)
    max_over_last_traded: float = 1.15  # non paga piu' del 15% sopra l'ultimo scambiato
    collector_desired_qty: int = 1
    reseller_max_qty: int = 6           # quante copie prova a prendere un flipper

    # Strategia di rivendita
    undercut: float = 0.05            # taglia il 5% sul miglior ask (vendita rapida)
    no_market_markup: float = 1.60    # se il book e' vuoto: prezzo "abbastanza alto"
    min_margin: float = 0.05          # non vende sotto MSRP*(1+min_margin)
    marketplace_fee: float = 0.10     # commissione piattaforma sul venduto
    listing_ttl: int = 8              # tick dopo cui un annuncio invenduto scade

    # Coda FCFS: pesi della priorita' di arrivo.
    # Non e' un ordinamento arbitrario ma la formalizzazione del vantaggio
    # informativo di chi monitora i drop (bot, alert, preordini).
    queue_w_hype: float = 0.35
    queue_w_resell: float = 0.30
    queue_w_noise: float = 0.35

    # ------------------------------------------------------------------ #
    # 3. Prodotto / Pokemon Company                                      #
    # ------------------------------------------------------------------ #
    msrp: float = 216.0               # EUR, booster box mercato europeo
    releases_per_year: int = 6
    supply_ratio: float = 0.50        # tiratura per release = supply_ratio * n_agents
    primary_window: int = 6           # tick di permanenza a listino pieno
    production_cost_ratio: float = 0.35  # costo industriale per unita', in quota di MSRP

    # --- Canale distributivo e svendita dell'eccedenza ---
    # I distributori non rifiutano il prodotto in eccesso: lo assorbono, ma
    # solo a prezzi di realizzo che affondano il margine. Il riferimento
    # empirico e' l'episodio delle Elite Trainer Box di Lost Origin (era
    # Sword & Shield), collocate attorno a 25-28$ contro un MSRP di 54$:
    # circa il 46% del listino, ben sotto il normale prezzo all'ingrosso.
    # Da qui i valori di `clearance_floor` e `channel_wholesale_ratio`.
    channel_wholesale_ratio: float = 0.55  # quota del prezzo che l'azienda realizza
    clearance_floor: float = 0.46     # prezzo minimo di svendita, in quota di MSRP
    clearance_decay: float = 0.10     # velocita' di discesa del prezzo per tick
    clearance_sensitivity: float = 1.5  # quanto l'eccedenza accelera lo sconto
    bargain_threshold: float = 0.85   # sotto questa quota di MSRP scatta l'occasione
    bargain_demand_boost: float = 0.45  # domanda aggiuntiva attivata dallo sconto

    # --- Ciclo di vita del prodotto e capacita' industriale ---
    # Un set resta IN STAMPA per due anni dalla release: entro quella finestra
    # puo' essere ristampato, dopodiche' non e' piu' producibile e resta solo
    # cio' che e' gia' in mano alle persone. E' il vincolo che trasforma la
    # scarsita' da parametro esogeno a proprieta' temporale del prodotto.
    print_window_ticks: int = 104     # 2 anni di finestra di stampa

    # La capacita' produttiva e' FISSA e sempre saturata: l'azienda stampa
    # sempre al massimo. Aumentare la tiratura di un set significa toglierla a
    # un altro, e ampliare la capacita' complessiva richiede anni. E' questo,
    # piu' di qualunque penalita' nel reward, a rendere "stampare di piu'" una
    # decisione con un costo opportunita' reale invece che un pasto gratis.
    capacity_lag_ticks: int = 78      # ~1.5 anni perche' l'espansione si realizzi
    capacity_step: float = 0.20       # variazione richiesta per azione
    capacity_bounds: tuple[float, float] = (0.4, 2.5)

    # Quota della capacita' dedicata alle ristampe invece che al prodotto nuovo.
    restock_share: float = 0.20
    restock_share_step: float = 0.10
    restock_share_bounds: tuple[float, float] = (0.0, 0.60)
    restock_min_premium: float = 1.20  # si ristampa solo cio' che tira davvero
    # La ristampa non torna al prezzo originale: viene ceduta al distributore
    # all'80% del prezzo di mercato corrente, cosi' il set non perde appeal.
    # L'azienda cattura buona parte del sovrapprezzo; il distributore lavora
    # su un margine piu' sottile del solito.
    restock_price_ratio: float = 0.80

    # --- Appetibilita' del set ---
    # Ogni set ha una desiderabilita' intrinseca (carte chase, Pokemon iconici,
    # alternate art): non e' esogena, la Pokemon Company la decide in fase di
    # progettazione. Alza lo shock di hype al lancio, la willingness-to-pay e
    # l'interesse degli influencer; ad appetibilita' molto alta gli agenti sono
    # disposti a pagare sopra MSRP gia' al giorno del rilascio.
    # E' anche una LEVA DI RAFFREDDAMENTO: una sequenza di set poco appetibili
    # ristagna sugli scaffali, finisce in svendita e trascina al ribasso il
    # sentiment dell'intero mercato, non solo il proprio prezzo.
    appeal_mean: float = 0.55         # media della Beta da cui si estrae
    appeal_concentration: float = 8.0  # concentrazione (piu' alta = meno varianza)
    appeal_target: float = 0.55       # obiettivo deliberato dall'azienda
    appeal_step: float = 0.10
    appeal_bounds: tuple[float, float] = (0.05, 0.95)
    # Un set si progetta con circa un anno di anticipo: la decisione
    # sull'appetibilita' non ha effetto immediato.
    appeal_lag_ticks: int = 52
    appeal_hype_weight: float = 1.2   # quanto l'appetibilita' amplifica lo shock
    appeal_wtp_weight: float = 0.7    # quanto pesa nella willingness-to-pay
    # Il costo si sostiene sulle unita' STAMPATE, il ricavo si incassa su
    # quelle VENDUTE. E' questa asimmetria a rendere il sovraccarico produttivo
    # oneroso: senza di essa premiare il fatturato lordo renderebbe "stampa
    # sempre di piu'" banalmente ottimale, che e' esattamente il fallimento
    # che si vuole evitare in un sandbox regolatorio.
    product_lifetime: int = 156       # orizzonte delle metriche "attive" (3 anni)

    # ------------------------------------------------------------------ #
    # 4. Super-agente (Pokemon Company pilotata da DDQN)                 #
    # ------------------------------------------------------------------ #
    use_super_agent: bool = False
    sa_delay: int = 2                 # agisce 1 volta ogni sa_delay tick
    msrp_step: float = 0.10           # +/- 10% di MSRP per azione
    msrp_multiplier_bounds: tuple[float, float] = (0.7, 2.0)
    cadence_step: int = 1             # +/- 1 settimana di intervallo fra release
    cadence_bounds: tuple[int, int] = (4, 26)


    # ------------------------------------------------------------------ #
    # 5. Metriche / soglie                                               #
    # ------------------------------------------------------------------ #
    bubble_threshold: float = 2.0     # theta_B: premium oltre cui c'e' "bolla"
    premium_cap_for_norm: float = 5.0 # normalizzazione dello stato osservato

    # ------------------------------------------------------------------ #
    # 6. Reward del super-agente                                         #
    # ------------------------------------------------------------------ #
    w_access: float = 1.0             # premia i collezionisti serviti
    w_unsold: float = 0.8             # penalizza l'invenduto (costo per l'azienda)
    w_afford: float = 0.8             # penalizza l'MSRP che esclude i poveri
    w_bubble: float = 1.0             # penalizza il superamento di theta_B
    # Il fallimento e' simmetrico. Un mercato in cui il sigillato vale meno del
    # listino non e' "accessibile": e' collassato, e chiunque abbia comprato ha
    # perso denaro. L'obiettivo dichiarato e' un prezzo vicino all'MSRP, non un
    # prezzo qualunque purche' basso. Senza questo termine la policy ottimale
    # sarebbe distruggere il mercato, che e' l'esito peggiore per tutti.
    w_collapse: float = 1.5
    collapse_threshold: float = 0.95  # sotto questo premium il mercato e' rotto
    w_margin: float = 1.0             # premia il MARGINE industriale, non il fatturato
    # Il margine spazia su piu' di un ordine di grandezza (da 0.5 a oltre 7
    # una volta che le ristampe a prezzo di mercato entrano in gioco). Un
    # semplice tetto renderebbe indistinguibili un margine di 2 e uno di 7,
    # cioe' cieco l'agente proprio sulla leva piu' redditizia. Si usa quindi
    # una trasformazione ASIMMETRICA: rendimenti decrescenti sui profitti
    # (logaritmica, normalizzata su `margin_reference`) e penalita' lineare
    # piena sulle perdite, che per un'azienda non sono simmetriche ai guadagni.
    margin_reference: float = 4.0
    # Senza `w_margin` il super-agente puo' azzerare il premium semplicemente
    # smettendo di vendere (rallentando le uscite): il trade-off diventa a tre
    # vie -- prezzo accessibile, collezionisti serviti, ricavi dell'azienda --
    # ed e' questa tensione a rendere la policy appresa non banale.

    # ------------------------------------------------------------------ #
    # 7. Esecuzione                                                      #
    # ------------------------------------------------------------------ #
    years: float = 5.0
    seed: int | None = None

    # ------------------------------------------------------------------ #
    # Proprieta' derivate                                                #
    # ------------------------------------------------------------------ #
    @property
    def total_ticks(self) -> int:
        return int(round(self.years * TICKS_PER_YEAR))

    @property
    def release_interval(self) -> int:
        return max(1, int(round(TICKS_PER_YEAR / self.releases_per_year)))

    @property
    def reseller_share(self) -> float:
        """Quota complessiva della popolazione con intento di rivendita."""
        return self.investor_share + (1 - self.investor_share) * self.fan_flipper_share

    @property
    def base_supply_units(self) -> int:
        """Tiratura di riferimento per singola release."""
        return max(1, int(round(self.supply_ratio * self.n_agents)))

    @property
    def capacity_per_year(self) -> int:
        """
        Capacita' industriale annua, in unita'. E' il vincolo duro: l'azienda
        la satura sempre, ripartendola fra prodotto nuovo e ristampe.
        """
        return self.base_supply_units * self.releases_per_year

    # ------------------------------------------------------------------ #
    # Utility                                                            #
    # ------------------------------------------------------------------ #
    def with_(self, **kwargs) -> "SimConfig":
        """Ritorna una copia modificata (la dataclass e' frozen)."""
        return replace(self, **kwargs)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, default=str)

    def as_dict(self) -> dict:
        return asdict(self)


# Preset comodi per gli esperimenti -------------------------------------- #

BASELINE = SimConfig()

HIGH_SCARCITY = SimConfig(supply_ratio=0.25, investor_share=0.45,
                          fan_flipper_share=0.35,
                          network_polarization=0.6, activation_threshold=0.27)

LOW_SCARCITY = SimConfig(supply_ratio=1.5, investor_share=0.05,
                         fan_flipper_share=0.10)

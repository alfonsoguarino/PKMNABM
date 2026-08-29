"""
market.py — Prodotti e microstruttura del mercato.

Due mercati distinti, come nella realta' del collezionismo:

* **Primario** (Pokemon Company -> pubblico): prezzo fisso all'MSRP, stock
  finito, allocazione **FCFS**. La coda non e' casuale: chi e' piu' in hype e
  chi ha intento speculativo arriva prima (bot, alert, preordini). Questo e'
  il canale attraverso cui la rete sociale si traduce in allocazione fisica.

* **Secondario** (peer-to-peer): order book di soli **ask** (chi ha la merce
  quota, chi compra colpisce il miglior ask). Non serve un book a due lati:
  nel collezionismo la domanda e' quasi sempre "market taker".

Il prezzo emergente e' `last_traded_price`: e' la variabile che il super-agente
deve controllare, l'analogo esatto del `global_cascade` nel modello fake news.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------- #
# Prodotto                                                               #
# ---------------------------------------------------------------------- #
@dataclass
class Product:
    """Un set / booster box rilasciato dalla Pokemon Company."""
    pid: int
    release_tick: int
    msrp_at_release: float
    initial_stock: int
    appeal: float = 0.55               # desiderabilita' intrinseca del set, in [0,1]

    stock: int = 0
    price: float = 0.0                 # prezzo al pubblico corrente (= MSRP)
    units_sold_primary: int = 0
    units_to_resellers: int = 0
    units_to_investors: int = 0
    units_opened: int = 0              # box aperte: distrutte, fuori dal circolante
    reprints: int = 0
    units_restocked: int = 0
    # Quanto l'azienda incassa per unita' venduta. Cambia con le ristampe:
    # il prodotto nuovo viaggia al prezzo all'ingrosso ordinario, la ristampa
    # e' ceduta all'80% del prezzo di mercato corrente.
    company_take: float = 0.0

    # storia del mercato secondario
    last_traded: float | None = None
    trade_prices: list[float] = field(default_factory=list)
    trade_ticks: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.stock = self.initial_stock
        self.price = self.msrp_at_release
        self.printed = self.initial_stock

    # -- stato --------------------------------------------------------- #
    def is_available(self) -> bool:
        """
        Finche' resta stock il prodotto e' acquistabile. Cambia il PREZZO, non
        la disponibilita': dopo la finestra di lancio l'eccedenza non svanisce,
        viene collocata a sconto crescente lungo il canale.
        """
        return self.stock > 0

    def in_launch_window(self, tick: int, window: int) -> bool:
        return tick <= self.release_tick + window

    def in_print(self, tick: int, print_window: int) -> bool:
        """
        Il set e' ancora producibile. Oltre la finestra di stampa non esiste
        piu' alcuna leva sull'offerta: resta soltanto cio' che e' gia' nelle
        mani delle persone, e da quel momento l'unico modo in cui l'offerta
        puo' cambiare e' diminuire (le box che vengono aperte).
        """
        return tick <= self.release_tick + print_window

    @property
    def discount(self) -> float:
        """Sconto corrente rispetto all'MSRP, in [0,1]."""
        if self.msrp_at_release <= 0:
            return 0.0
        return max(0.0, 1.0 - self.price / self.msrp_at_release)

    @property
    def overhang(self) -> float:
        """Quota di tiratura ancora invenduta: la misura dell'eccedenza."""
        return self.stock / max(1, self.initial_stock)

    def update_clearance(self, tick: int, window: int, cfg) -> None:
        """
        Aggiorna il prezzo al pubblico dopo la finestra di lancio.

        Il prezzo scende verso un obiettivo tanto piu' basso quanto piu' grande
        e' l'eccedenza, con pavimento a `clearance_floor * MSRP`. La discesa e'
        graduale: il canale svuota il magazzino per gradi, non con un unico
        taglio. Un prodotto andato esaurito non scende mai, perche' non c'e'
        nulla da collocare.
        """
        if self.stock <= 0 or self.in_launch_window(tick, window):
            return
        floor = cfg.clearance_floor
        pressure = min(1.0, self.overhang * cfg.clearance_sensitivity)
        target = self.msrp_at_release * (floor + (1.0 - floor) * (1.0 - pressure))
        step = self.msrp_at_release * cfg.clearance_decay
        self.price = max(target, self.price - step)

    def sold_out(self) -> bool:
        return self.stock <= 0

    @property
    def premium(self) -> float:
        """Rapporto ultimo prezzo scambiato / MSRP. 1.0 se mai scambiato."""
        if self.last_traded is None:
            return 1.0
        return self.last_traded / self.msrp_at_release

    @property
    def sealed_supply(self) -> int:
        """Unita' ancora sigillate e quindi potenzialmente scambiabili."""
        return max(0, self.units_sold_primary - self.units_opened)

    @property
    def opened_ratio(self) -> float:
        if self.units_sold_primary == 0:
            return 0.0
        return self.units_opened / self.units_sold_primary

    @property
    def investor_capture(self) -> float:
        if self.units_sold_primary == 0:
            return 0.0
        return self.units_to_investors / self.units_sold_primary

    def momentum(self, tick: int, window: int = 8) -> float:
        """
        Variazione relativa del prezzo fra la finestra corrente e la precedente,
        misurata in TICK e non in numero di scambi.

        La distinzione e' sostanziale. Con un momentum contato sugli ultimi N
        scambi, un mercato che si e' bloccato continua a segnalare l'ultimo
        trend rialzista per sempre, e gli investitori non escono mai: la bolla
        non ha modo di sgonfiarsi. Misurando sul tempo, l'assenza di scambi
        produce un momentum negativo -- ed e' esattamente cio' che accade nella
        realta', dove un mercato illiquido e' il primo segnale di inversione.
        """
        if not self.trade_ticks:
            return 0.0

        cur = [pr for pr, t in zip(self.trade_prices, self.trade_ticks)
               if t > tick - window]
        prev = [pr for pr, t in zip(self.trade_prices, self.trade_ticks)
                if tick - 2 * window < t <= tick - window]

        if not cur:
            # nessuno scambia piu': il segnale e' di raffreddamento
            return -0.5 if prev else 0.0
        if not prev:
            return 0.0

        a, b = sum(prev) / len(prev), sum(cur) / len(cur)
        if a <= 0:
            return 0.0
        return float(max(-1.0, min(1.0, (b - a) / a)))

    @property
    def reseller_capture(self) -> float:
        if self.units_sold_primary == 0:
            return 0.0
        return self.units_to_resellers / self.units_sold_primary

    def recent_volume(self, tick: int, window: int = 8) -> int:
        return sum(1 for t in self.trade_ticks if t > tick - window)

    def record_trade(self, price: float, tick: int) -> None:
        self.last_traded = price
        self.trade_prices.append(price)
        self.trade_ticks.append(tick)

    def restock(self, units: int, market_price: float, price_ratio: float) -> None:
        """
        Ristampa: le nuove unita' NON tornano al prezzo originale ma entrano
        al prezzo di mercato corrente, cosi' il set non perde appeal. Il
        distributore le acquista a `price_ratio` del mercato e lavora quindi
        su un margine piu' sottile del solito.
        """
        self.stock += units
        self.printed += units
        self.units_restocked += units
        self.reprints += 1
        self.price = market_price
        self.company_take = price_ratio * market_price


# ---------------------------------------------------------------------- #
# Mercato secondario                                                     #
# ---------------------------------------------------------------------- #
@dataclass(order=True)
class Ask:
    price: float
    seq: int                            # tie-break FIFO deterministico
    seller_id: int = field(compare=False)
    product_id: int = field(compare=False)
    posted_tick: int = field(compare=False, default=0)


class OrderBook:
    """Book di soli ask, uno per prodotto. Ordinato per prezzo crescente."""

    def __init__(self):
        self._asks: dict[int, list[Ask]] = {}
        self._seq = 0

    def add_ask(self, product_id: int, seller_id: int, price: float,
                tick: int = 0) -> None:
        self._seq += 1
        book = self._asks.setdefault(product_id, [])
        book.append(Ask(price=price, seq=self._seq, seller_id=seller_id,
                        product_id=product_id, posted_tick=tick))
        book.sort()

    def expire(self, tick: int, ttl: int) -> int:
        """
        Rimuove gli annunci invenduti da piu' di `ttl` tick.

        Senza scadenza il book accumula offerte fuori mercato che nessuno
        colpira' mai, e il `best_ask` resta ancorato a prezzi stantii: il
        venditore, nella realta', ritira e riquota piu' in basso.
        """
        removed = 0
        for pid in list(self._asks):
            keep = [a for a in self._asks[pid] if tick - a.posted_tick <= ttl]
            removed += len(self._asks[pid]) - len(keep)
            if keep:
                self._asks[pid] = keep
            else:
                del self._asks[pid]
        return removed

    def best_ask(self, product_id: int) -> Ask | None:
        book = self._asks.get(product_id)
        return book[0] if book else None

    def pop_best(self, product_id: int) -> Ask | None:
        book = self._asks.get(product_id)
        if not book:
            return None
        ask = book.pop(0)
        if not book:
            del self._asks[product_id]
        return ask

    def cheapest_overall(self, product_ids=None) -> Ask | None:
        """
        Miglior offerta su TUTTO il mercato.
        Implementa la regola "compra il prodotto piu' economico fra quelli
        disponibili": il compratore non e' fedele al set, e' fedele al prezzo.
        """
        best = None
        for pid, book in self._asks.items():
            if product_ids is not None and pid not in product_ids:
                continue
            if book and (best is None or book[0].price < best.price):
                best = book[0]
        return best

    def n_listings(self, product_id: int | None = None) -> int:
        if product_id is None:
            return sum(len(b) for b in self._asks.values())
        return len(self._asks.get(product_id, ()))

    def prune_orphans(self, seller_id: int, inventory: dict) -> None:
        """
        Rimuove gli annunci di un venditore che non corrispondono piu' a copie
        realmente possedute. Serve dopo l'apertura delle box: una copia aperta
        e' distrutta, ma il suo annuncio resterebbe a book a inquinare il
        `best_ask` con un'offerta ineseguibile.
        """
        for pid in list(self._asks):
            owned = len(inventory.get(pid, ()))
            mine = [a for a in self._asks[pid] if a.seller_id == seller_id]
            if len(mine) <= owned:
                continue
            drop = set(id(a) for a in mine[owned:])
            self._asks[pid] = [a for a in self._asks[pid] if id(a) not in drop]
            if not self._asks[pid]:
                del self._asks[pid]

    def remove_seller_listings(self, seller_id: int) -> None:
        for pid in list(self._asks):
            self._asks[pid] = [a for a in self._asks[pid]
                               if a.seller_id != seller_id]
            if not self._asks[pid]:
                del self._asks[pid]

    def all_prices(self) -> list[float]:
        return [a.price for book in self._asks.values() for a in book]

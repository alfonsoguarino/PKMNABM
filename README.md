# Un sandbox regolatorio per i mercati della scarsità
### Simulazione ad agenti e deep reinforcement learning applicati al collezionismo Pokémon TCG

Questo progetto trasferisce l'architettura a due tier di Zaccagnino et al. (2025),
*Turning AI into a regulatory sandbox*, dal dominio dell'information disorder a
quello dei mercati basati sulla scarsità artificiale. Il modello ad agenti è
riscritto da NetLogo a **Mesa 3**, il super-agente da Keras a **PyTorch**.

---

## 1. La mappatura concettuale

L'ipotesi di lavoro è che una bolla speculativa su un bene collezionabile e una
cascata di disinformazione siano lo stesso oggetto formale: un contagio complesso
su una rete sociale polarizzata, in cui un'entità superiore cerca di intervenire.

| Zaccagnino et al. / Törnberg | Questo modello |
|---|---|
| nodo con soglia di attivazione | partecipante al mercato con soglia di contagio |
| opinione fake / neutro / true | **hype** continuo in [0,1] |
| `op_step` (rigidità dell'opinione) | idem: gradualità del cambio di convinzione |
| echo chamber ($P_n$, $P_o$) | nicchia di collezionisti iper-connessi |
| `global_cascade` | **premium index** = prezzo di mercato / MSRP |
| *Virality* $V$ = frazione di run con cascade > 0.5 | **Bubble Frequency** $B$ = frazione di run con premium > $\theta_B$ = 2.0 |
| super-agente = autorità pubblica | super-agente = The Pokémon Company |
| warning / reiterate / static_true / no_op | 11 azioni su capacità, prezzo, cadenza, ristampe, appetibilità |

**Validazione strutturale.** Il modello riproduce l'*effetto echo chamber* di
Törnberg: la frequenza delle bolle cresce con la polarizzazione di rete fino a
$P_n \approx 0.4$–$0.6$ e crolla oltre $0.7$ — la stessa U rovesciata della sua
Fig. 1 — e la polarizzazione di opinione ha effetto monotono crescente. Che il
risultato strutturale trasferisca da un dominio all'altro è il principale
argomento a sostegno dell'impianto.

---

## 2. Installazione

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Python 3.12 o 3.13 obbligatorio.** Mesa 3.5 dichiara `requires-python >= 3.12`.
Il vincolo `mesa<4.0` in `requirements.txt` è deliberato: l'API di Mesa 4 (in
alpha) non è compatibile.

Verifica rapida:

```bash
python -c "from config import SimConfig; from pokesim.model import PokemonMarketModel; \
           print(PokemonMarketModel(SimConfig(n_agents=100, years=1.0, seed=0)).run().tail(3))"
```

---

## 3. Struttura

```
config.py               tutti i parametri, in una dataclass congelata e serializzabile
pokesim/                TIER MODEL-DRIVEN
  networks.py             topologie + costruzione echo chamber alla Törnberg
  agents.py               fan, fan-flipper, investitori puri
  market.py               prodotti, order book, svendita del canale
  model.py                il modello Mesa: orchestrazione del tick
  metrics.py              premium index, accessibilità, Gini, Bubble Frequency
rl/                     TIER DATA-DRIVEN
  env.py                  ambiente Gymnasium + funzione di reward
  ddqn.py                 Double DQN in PyTorch
  train.py                training, test e confronto con le baseline
experiments/
  sweeps.py               12 sweep parametrici sistematici
  plots.py                generazione di tutte le figure
```

---

## 4. Come funziona il modello

**Tempo.** 1 tick = 1 settimana; 4 tick = 1 mese (ricarica del budget); 52 = 1 anno.

**Un tick** esegue in ordine: espansioni di capacità e appetibilità in scadenza →
ricarica redditi → eventuale release → ristampe → aggiornamento prezzi di svendita
→ mercato primario FCFS → diffusione dell'hype → shock di liquidità → apertura
delle box → quotazioni → scambi sul secondario → metriche.

### Meccanismi che determinano la dinamica

**Contagio complesso con decadimento dell'attenzione.** L'hype sale se la
frazione di vicini attivi supera la soglia individuale, ma il decadimento è
*incondizionato* e il contagio si somma sopra. Applicarlo solo nel ramo `else`
— come in un modello di cascata singola — rende quel ramo irraggiungibile appena
tutti i vicini sono attivi, e l'hype satura a 1 senza tornare indietro. È il
termine che genera i cicli invece di un plateau.

**Tre archetipi.** Il *fan* valuta per valore di consumo (WTP guidata dall'hype
e dall'appetibilità del set) e apre parte di ciò che compra. Il *fan-flipper*
rivende. L'*investitore puro* non apre nulla e valuta finanziariamente.

**Pozzo di distruzione dell'offerta.** `open_rate` è la *quota* di prodotto che
verrà aperta, decisa all'acquisto; le box aperte escono per sempre dal circolante.
È l'unico meccanismo che riduce l'offerta nel tempo. Se il sigillato prende
troppo valore una copia destinata all'apertura può essere dirottata sul mercato:
prezzi alti → si apre meno → resta più sigillato → i prezzi si calmierano.

**Capacità industriale fissa.** L'azienda stampa sempre al massimo. Aumentare la
tiratura di un set significa toglierla a un altro; ampliare la capacità
complessiva richiede ~1,5 anni. È questo, più di qualsiasi penalità nel reward,
a rendere «stampare di più» una decisione con un costo opportunità reale.

**Finestra di stampa.** Un set è producibile per 2 anni dalla release. Oltre,
resta solo ciò che è già in mano alle persone.

**Ristampe a prezzo di mercato.** Non tornano a listino: entrano all'80% del
prezzo di mercato corrente, così il set non perde appeal. L'azienda cattura il
sovrapprezzo, il distributore lavora su margine sottile.

**Svendita del canale.** L'eccedenza non viene rifiutata: viene collocata a
sconto crescente, con pavimento al 46% dell'MSRP. Il riferimento empirico è
l'episodio delle Elite Trainer Box di *Lost Origin*, collocate attorno a 25–28 $
contro un MSRP di 54 $. Conseguenze a catena: il rivenditore capitola (nessuno
quota sopra il prezzo a cui il negozio sta svendendo), il premium scende sotto 1,
il margine va in negativo.

**Appetibilità del set.** Ogni prodotto ha una desiderabilità intrinseca che
l'azienda decide in progettazione, con circa un anno di anticipo. Alza lo shock
di hype al lancio, la WTP e l'interesse degli influencer. È anche una **leva di
raffreddamento**: set poco appetibili ristagnano, finiscono in svendita e
trascinano al ribasso il sentiment dell'intero mercato.

**Influencer.** Agisce su due canali, e il secondo è quello che conta: non solo
l'hype, ma la *tolleranza al prezzo* — sposta il giudizio «a questa cifra non lo
compro». Il suo pubblico non coincide con la cerchia sociale (`influencer_reach`).

**Mediazione del prezzo di carico.** Chi ha movente speculativo non insegue i
rialzi: compra sotto il proprio carico medio per abbassarlo. Freno alle bolle e
sostegno nelle discese.

**Shock di liquidità.** Probabilità decrescente col reddito: un evento ogni ~3
anni nel quartile povero, ogni ~12 nel quartile ricco. Chi lo subisce liquida
tutto all'85–90% del mercato. È offerta scorrelata dal ciclo dell'hype.

---

## 5. Il super-agente

**Osservazione** (8 componenti in [0,1]): premium normalizzato, hype medio,
invenduto, accessibilità, cattura degli speculatori, hype dell'influencer,
capitale di marca, appetibilità media.

**Azioni** (11): `no_op`, `capacity_up/down` (ritardo ~1,5 anni),
`msrp_up/down`, `cadence_faster/slower`, `restock_more/less` (ripartizione
immediata della capacità), `appeal_up/down` (ritardo ~1 anno).

Due assenze sono deliberate. Non esiste un tetto di copie per persona: la
Pokémon Company vende a distributori intermedi e non può farlo rispettare lungo
la filiera. Non esiste un'azione «stampa di più subito»: la capacità è fissa.

**Reward** — la componente principale replica la forma del paper originale
(segno della variazione della metrica-obiettivo, pesato da `action_weight`), con
in più:

- `+ w_access ·` accessibilità (misurata su finestra annuale, **neutrale rispetto
  alla cadenza**: contare solo il set più recente premierebbe artificialmente il
  rallentamento delle uscite);
- `+ w_margin ·` margine industriale, con trasformazione **asimmetrica** —
  logaritmica sui profitti (rendimenti decrescenti), lineare piena sulle perdite;
- `− w_unsold ·` invenduto, `− w_afford ·` crescita dell'MSRP;
- `− w_bubble` se il premium supera $\theta_B$, `− w_collapse ·` se scende sotto
  0.95. **Il fallimento è simmetrico**: un mercato in cui il sigillato vale meno
  del listino non è accessibile, è collassato.

I pesi del reward sono la **posizione normativa esplicita**. Cambiarli cambia la
policy ottima, ed è precisamente ciò che un sandbox regolatorio deve permettere
di esplorare.

**Rete**: `Dense(8) → 24 (ReLU, He) → 12 (ReLU, He) → 11 (lineare)`, loss di
Huber, Adam(1e-3), replay 50k, target network ogni 100 step, ε-greedy con
decadimento esponenziale. Identica al paper, con una correzione: il codice
originale implementa il target **DQN** (`max_a' Q_target`); qui si usa il
disaccoppiamento DDQN di van Hasselt (`Q_target(argmax_a' Q_main)`).
`--no-double` ripristina il comportamento originale per l'ablation.

---

## 6. Esecuzione

```bash
# 1. Sweep del baseline (senza super-agente). Il più lungo: parallelizza.
python -m experiments.sweeps --all --runs 30 --workers 8 --out results/sweeps

# oppure un singolo sweep
python -m experiments.sweeps --sweep polarization --runs 50

# 2. Training del super-agente
python -m rl.train --episodes 150 --test-episodes 40 --out results/rl

# 3. Tutte le figure
python -m experiments.plots --results results --out figures
```

> **Esecuzione.** Gli script vanno lanciati **dalla cartella radice** del
> progetto (quella che contiene `config.py`, `pokesim/`, `rl/`). La forma
> `python -m experiments.sweeps` è quella consigliata. Gli entry-point
> aggiungono comunque la root a `sys.path` da soli, quindi funzionano anche
> lanciati come file (`python experiments/sweeps.py`) o con un percorso
> assoluto da un'altra directory. Se vedi `ModuleNotFoundError: No module
> named 'pokesim'` (o `'config'`), stai quasi certamente eseguendo da dentro
> `experiments/`: torna alla root.

Tempi indicativi (8 core, N=250, 8 anni): una run ≈ 2–4 s; uno sweep completo a
30 repliche ≈ 40–90 min; il training ≈ 30–60 min su CPU.

---

## 7. Risultati già verificati

Su cui costruire l'articolo. Tutti riprodotti su repliche multiple.

**L'effetto echo chamber trasferisce.** $B$ cresce con $P_n$ fino a 0.4–0.6 e
crolla oltre 0.7; $P_o$ ha effetto monotono crescente.

**L'accessibilità ha un massimo interno.** Cresce con l'offerta fino a ~1,5
unità per agente (0.72) e poi *scende* (0.55 a 4×). Sovrastampare danneggia
anche i collezionisti: satura il mercato e spegne la domanda.

**Biforcazione sul decadimento dell'hype.** Sotto ~0.03 il sistema entra in
regime runaway (hype saturo, accessibilità al 4%); sopra, in regime oscillatorio.

**Aprire le box fa salire i prezzi.** Premium 2.65 → 3.00 passando da `open_rate`
0 a 0.6–0.9: la supply distrutta non torna.

**Gli investitori puri sono il driver principale.** Premium 2.37 senza, 4.48 al
60%.

**L'appetibilità è una leva di raffreddamento che costa.** Premium 3.70 → 1.06
abbassandola, ma il margine scende da 3.79 a 0.87.

**Nessuna leva singola domina.** Le tre migliori azioni costanti stanno entro il
10% e una policy mista scritta a mano le batte tutte: il problema di controllo
è ben posto e non degenere.

---

## 8. Limiti noti

- I distributori non sono agenti espliciti: il canale è modellato come funzione
  di prezzo aggregata. È una scelta deliberata, ma preclude di studiare
  l'allocazione fra rivenditori.
- Un solo echo chamber, come nel modello di riferimento. Le reti reali ne hanno
  molti.
- Nessuna calibrazione su dati reali di prezzo. I valori empirici usati (MSRP
  216 €, pavimento di svendita al 46%) sono ancoraggi puntuali, non una stima.
- Il mercato PSA delle singole carte è escluso per scelta: la Pokémon Company
  non ha leve su di esso.
- La capacità è un unico numero aggregato: nessuna distinzione fra linee
  produttive o formati di prodotto.

---

## 9. Riferimenti

1. Zaccagnino R., Lettieri N., Malandrino D., Lomasto L., Camoia A., Guarino A.
   (2025). *Turning AI into a regulatory sandbox*. Neural Computing and
   Applications 37, 18679–18720.
2. Törnberg P. (2018). *Echo chambers and viral misinformation: Modeling fake
   news as complex contagion*. PLoS ONE 13(9): e0203958.
3. ter Hoeven E. et al. (2025). *Mesa 3: Agent-based modeling with Python in
   2025*. JOSS 10(107), 7668.
4. van Hasselt H., Guez A., Silver D. (2015). *Deep reinforcement learning with
   double Q-learning*.
5. Centola D., Macy M. (2007). *Complex contagions and the weakness of long
   ties*. AJS 113(3).

**Sul contesto empirico** non esiste letteratura ABM sul mercato Pokémon. Gli
agganci più prossimi sono il resale di sneaker in edizione limitata, i mercati di
token speculativi (TokenLab, arXiv:2412.07512) e la linea ICAIF/ABIDES sul deep
RL in mercati simulati. Nessuno di questi colloca però un *regolatore appreso*
sopra un ABM con echo chamber: è lo spazio che questo lavoro occupa.

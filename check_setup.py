"""
check_setup.py — Verifica che il progetto sia strutturato correttamente.

Esegui da riga di comando, dalla cartella radice:

    python check_setup.py

Controlla che ogni file sia nella cartella giusta e che i package siano
importabili. NON richiede che mesa/torch siano installati: verifica solo la
struttura. Se tutto e' verde, gli script gireranno una volta installate le
dipendenze con `pip install -r requirements.txt`.
"""

import os
import sys

EXPECTED = {
    ".": ["config.py", "requirements.txt", "README.md"],
    "pokesim": ["__init__.py", "agents.py", "market.py", "metrics.py",
                "model.py", "networks.py"],
    "rl": ["__init__.py", "ddqn.py", "env.py", "train.py"],
    "experiments": ["__init__.py", "plots.py", "sweeps.py"],
}

root = os.path.dirname(os.path.abspath(__file__))
ok = True

print("Verifica della struttura del progetto\n" + "-" * 40)
for folder, files in EXPECTED.items():
    path = root if folder == "." else os.path.join(root, folder)
    label = "(radice)" if folder == "." else folder + "/"
    if folder != "." and not os.path.isdir(path):
        print(f"  [MANCA]   cartella {label}")
        ok = False
        continue
    for f in files:
        if os.path.isfile(os.path.join(path, f)):
            print(f"  [ok]      {label}{f}")
        else:
            print(f"  [MANCA]   {label}{f}")
            ok = False

print("-" * 40)
if not ok:
    print("STRUTTURA INCOMPLETA. Sistema i file mancanti seguendo l'albero")
    print("mostrato nel README (sezione 3) prima di procedere.")
    sys.exit(1)

print("Struttura corretta. Provo a importare i package...")
try:
    for mod in ("config", "pokesim.model", "rl.env", "experiments.sweeps"):
        __import__(mod)
    print("Import riusciti: il progetto e' pronto.")
    print("\nProssimo passo:  pip install -r requirements.txt")
except ImportError as e:
    msg = str(e)
    if any(dep in msg for dep in ("mesa", "torch", "gymnasium",
                                  "numpy", "pandas", "networkx")):
        print(f"Struttura ok. Manca una dipendenza esterna: {msg}")
        print("Installa tutto con:  pip install -r requirements.txt")
    else:
        print(f"Struttura ok ma un import interno fallisce: {msg}")
        sys.exit(1)

"""Rechnet die Reihe für die Standardeinstellungen von Experiment 3 vor und schreibt `tools/vorgerechnet.json` (siehe ugm_vorgerechnet.py).

Aufruf (im Ordner der Demo): <Test-venv>/Scripts/python.exe -X utf8 tools/vorrechnen.py      (rund 10 Sekunden)
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ugm_evaluation as E  # noqa: E402
from ugm_vorgerechnet import DATEI, schluessel  # noqa: E402


def main():
    daten = {}
    for name, params, rechne in E.vorzurechnen():
        t = time.time()
        daten[schluessel(name, **params)] = E.jsonbar(rechne())
        print(f"{name} {params}: {time.time() - t:.1f} s", flush=True)
    DATEI.write_text(json.dumps(daten, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"geschrieben: {DATEI} ({DATEI.stat().st_size} Byte)")


if __name__ == "__main__":
    main()

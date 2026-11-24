"""Vorgerechnete Reihen: `tools/vorgerechnet.json` enthält das Ergebnis der Ausreißer-Reihe für die Standardeinstellungen der App (erzeugt von `tools/vorrechnen.py` mit dem Code der Demo).

Wozu: Die Reihe über 200 Instanzen mit je 100 Quellen und 100 Zielen braucht auf dem Entwicklungsrechner einige Sekunden (die Normalzahlen kommen aus einem plattformfesten
Generator mit `decimal`), auf der Cloud wahrscheinlich 2- bis 3-mal so lange. Mit den Standardeinstellungen erscheint die Seite sofort; bei jeder anderen Einstellung rechnet die App
live (Spinner, Zwischenspeicher). `tests/test_claims.py` rechnet den Eintrag neu und vergleicht ihn als Band."""

import json
from functools import lru_cache
from pathlib import Path

DATEI = Path(__file__).resolve().parent / "tools" / "vorgerechnet.json"


def schluessel(name: str, **params) -> str:
    """Eindeutiger Schlüssel aus Name und Parametern (Zahlen als int bzw. float, Texte als Texte)."""
    return name + "|" + ",".join(f"{k}={params[k]!r}" for k in sorted(params))


@lru_cache(maxsize=1)
def _lade() -> dict:
    try:
        return json.loads(DATEI.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def hole(name: str, **params):
    """Der vorgerechnete Eintrag oder None."""
    return _lade().get(schluessel(name, **params))

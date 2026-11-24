"""Messreihen der README (alle Zahlen, die die README nennt): rechnet mit dem Code der Demo (Zufalls-Seed 2026 der Reihen, SplitMix64) und schreibt `tools/messreihen.json`.

Aufruf (im Ordner der Demo): <Test-venv>/Scripts/python.exe -X utf8 tools/messreihen.py      (rund 20 Sekunden)
`tests/test_claims.py` bildet die README-Zahlen aus dieser Datei und rechnet die Reihen neu (als Band)."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ugm_constants as C  # noqa: E402
import ugm_evaluation as E  # noqa: E402

GROSSE_LISTEN = ("ersparnis_liste", "rho_voll_liste", "raster", "raster_bewegt", "raster_bedient", "rho_alle_liste")
SEEDS_ANDERE = (1, 2, 3, 4, 5)       # andere Zufalls-Seeds der Reihe bei ρ = 0,1 (Streuung von „nicht den ganzen Bestand“)
RHO_PRESET3 = 6.0                    # ρ des Presets „Ausreißer 5 %“


def kurz(x):
    """Entfernt die langen Listen (Werte je Instanz, Raster der Diagramme), die die README nicht braucht."""
    if isinstance(x, dict):
        return {k: kurz(v) for k, v in x.items() if k not in GROSSE_LISTEN}
    if isinstance(x, list):
        return [kurz(v) for v in x]
    return x


def zeit(name, f):
    t = time.time()
    r = f()
    print(f"{name}: {time.time() - t:.1f} s", flush=True)
    return kurz(E.jsonbar(r))


def ansicht_zahlen():
    an = E.ansicht(E.Settings())
    st = an.strasse
    return dict(seed=E.Settings().seed, ueberhang=st.ueberhang, norm_kosten=an.norm.bewegungskosten, teil_kosten=an.teil.bewegungskosten, ersparnis=an.ersparnis,
                zaehl_norm=list(an.zaehl_norm), zaehl_teil=list(an.zaehl_teil), norm_zielwert=an.norm.zielwert, teil_zielwert=an.teil.zielwert, ungeliefert=E.ungeliefert(st, an.teil),
                rho_voll=an.rho_voll_rel, grenzkosten=an.kurve.grenzkosten_letzte)


def main():
    d0 = C.DEFAULTS
    d = {"meta": {"seed": C.SERIEN_SEED, "anzahl": d0["inst1"], "n_kunden": d0["n_kunden"], "rho1": d0["rho1"], "ueberhang2": 1.3, "anteil3": d0["anteil3"], "abstand3": d0["abstand3"],
                  "anzahl3": d0["inst3"], "anzahl_weich": C.ANZAHL_WEICH}}
    d["ansicht"] = zeit("ansicht", ansicht_zahlen)
    d["tab1"] = zeit("tabelle 1", lambda: E.reihen_ueberhang(d0["rho1"], d0["n_kunden"], d0["inst1"]))
    d["tab1_rho01"] = zeit("reihe rho = 0,1", lambda: E.reihe_normalisieren(1.3, 0.1, d0["n_kunden"], d0["inst1"]))
    d["tab1_n60"] = zeit("reihe 60 Kunden", lambda: E.reihe_normalisieren(1.3, d0["rho1"], 60, d0["inst1"]))
    d["strafe"] = zeit("strafe", lambda: E.reihe_strafe(C.RHO_TABELLE, 1.3, d0["n_kunden"], d0["inst1"]))
    d["tab1_rho01_seeds"] = zeit("reihe rho = 0,1, andere Seeds", lambda: dict(seeds=list(SEEDS_ANDERE), nicht_voll=[E.reihe_normalisieren(1.3, 0.1, d0["n_kunden"], d0["inst1"], seed=s)["nicht_voll"] for s in SEEDS_ANDERE]))
    ra = E.reihe_ausreisser(d0["anteil3"], d0["abstand3"], d0["inst3"])
    d["ausreisser"] = kurz(E.jsonbar(ra))
    d["ausreisser_preset"] = dict(rho=RHO_PRESET3, alle_bedient=E.anzahl_alle_bedient(ra, RHO_PRESET3), anzahl=ra["anzahl"])
    d["weich"] = zeit("weich", E.weich_messung)
    (ROOT / "tools" / "messreihen.json").write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("geschrieben: tools/messreihen.json")


if __name__ == "__main__":
    main()

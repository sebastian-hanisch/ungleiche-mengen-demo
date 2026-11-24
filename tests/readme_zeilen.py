"""Zeilen und Sätze der README, aus `tools/messreihen.json` mit denselben Formatfunktionen und Verdikten wie in der App gebildet (ugm_evaluation).

Die README muss jede dieser Zeilen und jeden dieser Sätze wörtlich enthalten (`tests/test_claims.py`); `python tests/readme_zeilen.py` druckt sie zum Einfügen."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ugm_constants as C  # noqa: E402
import ugm_evaluation as E  # noqa: E402

J = json.loads((ROOT / "tools" / "messreihen.json").read_text(encoding="utf-8"))


def zeile_tab1(z: dict) -> str:
    return (f"| {E.de_g(z['ueberhang'], 3)} | {E.mittel_text(z['ersparnis'])} | {E.prozent(z['leer'][0], 1)} | {z['mit_leer']} von {z['anzahl']} | "
            f"{E.de(z['norm_kosten'][0], 2)} / {E.de(z['teil_kosten'][0], 2)} |")


def zeile_strafe(r: dict, i: int) -> str:
    return f"| {E.de_g(r['rho'][i], 3)} | {E.mittel_text(r['bewegt'][i])} | {E.prozent(r['leer'][i][0], 1)} |"


def zeilen_ausreisser(r: dict) -> list:
    return [f"| erster Ausreißer wird bedient ab ρ_erste | {E.mittel_text(r['rho_erste'], 2, False)} |",
            f"| alle Ausreißer werden bedient ab ρ_alle | {E.mittel_text(r['rho_alle'], 2, False)} |",
            f"| Untergrenze d_min/2 (halber kürzester Weg zu einem Ausreißer) | {E.mittel_text(r['d_min2'], 2, False)} |",
            f"| Untergrenze d_max/2 für ρ_alle (halber größter der kürzesten Wege je Ausreißer) | {E.mittel_text(r['d_max2'], 2, False)} |"]


def zeilen_weich(w: dict) -> list:
    kopf1 = "| τ/ε | " + " | ".join(E.de_g(t, 3) for t in w["tau_eps"]) + " | τ/ε → ∞ |"
    z1 = "| Masse im 2 × 2-Beispiel | " + " | ".join(E.de(x, 3) for x in w["zwei_mal_zwei"]) + f" | {E.de(w['grenzwert'], 4)} |"
    z2 = f"| Masse, {w['n']} × {w['n']}, Mittel [Minimum; Maximum] | " + " | ".join(E.mittel_text(b, 2, False) for b in w["masse"]) + " | – |"
    z3 = "| Matrix-Vektor-Produkte (Median) | " + " | ".join(E.ganz(p) for p in w["produkte_median"]) + " | – |"
    kopf2 = "| ρ/OPT (Modell mit Schlupf) | " + " | ".join(E.de_g(r, 2) for r in w["rho_opt"]) + " |"
    y1 = "| bewegte Masse, Mittel [Minimum; Maximum] | " + " | ".join(E.mittel_text(b, 2, False) for b in w["schlupf_bewegt"]) + " |"
    y2 = "| leere Lager (Mittel, größte Zahl) | " + " | ".join(f"{E.de(le[0], 1)} ({le[1]})" for le in w["schlupf_leer"]) + " |"
    return [kopf1, z1, z2, z3, kopf2, y1, y2]


def saetze() -> dict:
    """Name → Satz; die README enthält jeden Satz wörtlich."""
    a = J["ansicht"]
    tab = {z["ueberhang"]: z for z in J["tab1"]}
    r13 = tab[1.3]
    out = {}
    out["verdikt_1"] = E.verdikt_normalisieren(r13)[1]
    out["verdikt_gleich"] = E.verdikt_normalisieren(tab[1.0])[1]
    out["verdikt_2"] = E.verdikt_strafe(J["strafe"])[1]
    out["verdikt_3"] = E.verdikt_ausreisser(J["ausreisser"])[1]
    out["verdikt_weich"] = E.verdikt_weich(J["weich"])[1]
    out["ansicht"] = (f"In der Standardstraße der App (Zufalls-Seed {a['seed']}, {J['meta']['n_kunden']} Kunden, R = {E.de_g(a['ueberhang'], 3)}, ρ = {E.de_g(J['meta']['rho1'], 3)} Straßenlängen, eine Instanz) "
                      f"kostet das Normalisieren {E.de(a['norm_kosten'], 2)} Bewegungskosten, die Teilbedienung {E.de(a['teil_kosten'], 2)} ({E.prozent(a['ersparnis'], 1)} weniger, Basis Normalisieren); "
                      f"bei der Teilbedienung bekommen {a['zaehl_teil'][0]} Kunden alles, {a['zaehl_teil'][1]} nichts und {a['zaehl_teil'][2]} einen Teil, und beide Pläne lassen {E.de_g(a['ungeliefert'], 4)} Einheiten ungeliefert.")
    out["nie_teurer"] = (f"Teurer als das Normalisieren war die Teilbedienung in {sum(z['teurer'] for z in J['tab1'])} von {sum(z['anzahl'] for z in J['tab1'])} Instanzen "
                         f"(fünf Reihen mit R = 1 bis 2, ρ = {E.de_g(J['meta']['rho1'], 3)} Straßenlängen).")
    k01, k60 = J["tab1_rho01"], J["tab1_n60"]
    sd = J["tab1_rho01_seeds"]
    out["rho_klein"] = (f"Bei ρ = 0,1 Straßenlängen (R = 1,3) bewegt die Teilbedienung in {k01['nicht_voll']} von {k01['anzahl']} Instanzen nicht den ganzen Bestand (in diesen Instanzen im Mittel {E.prozent(k01['bewegt_nicht_voll'][0], 1)}, über alle {k01['anzahl']} Instanzen im Mittel {E.prozent(k01['bewegt'][0], 1)} des Bestands bewegt; "
                        f"bei {len(sd['seeds'])} anderen Zufalls-Seeds der Reihe {min(sd['nicht_voll'])} bis {max(sd['nicht_voll'])} von {k01['anzahl']}); "
                        f"die Ersparnis von im Mittel {E.prozent(k01['ersparnis'][0], 1)} gilt dann für weniger bewegte Einheiten und ist kein Vergleich auf gleicher Grundlage.")
    out["n60"] = (f"Mit 60 statt 30 Kunden (R = 1,3, ρ = {E.de_g(J['meta']['rho1'], 3)}) ist die Ersparnis im Mittel {E.mittel_text(k60['ersparnis'])}, in {k60['mit_leer']} von {k60['anzahl']} Instanzen ist mindestens ein Kunde leer.")
    return out


def tabellenzeilen() -> list:
    out = [zeile_tab1(z) for z in J["tab1"]]
    out += [zeile_strafe(J["strafe"], i) for i in range(len(J["strafe"]["rho"]))]
    out += zeilen_ausreisser(J["ausreisser"])
    out += zeilen_weich(J["weich"])
    return out


if __name__ == "__main__":
    print("## Tabellenzeilen")
    for z in tabellenzeilen():
        print(z)
    print("## Sätze")
    for k, v in saetze().items():
        print(f"[{k}] {v}\n")

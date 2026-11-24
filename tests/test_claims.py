"""Jede Zahl der README (DEMO-PLAYBOOK §4 und §11a).

Die Messreihen liegen in `tools/messreihen.json` (erzeugt von `tools/messreihen.py` mit dem Code der Demo), die Ausreißer-Reihe der Standardeinstellungen der App in
`tools/vorgerechnet.json`. Dieser Test macht drei Dinge:
(1) die README-Zeilen und -Sätze werden aus der Datei mit denselben Formatfunktionen und Verdikten wie in der App gebildet und müssen wörtlich in der README stehen (so kann keine Zahl
    der README von der Messung abweichen, und App und README sagen dasselbe);
(2) die Reihen werden neu gerechnet und als Band mit den Dateien verglichen (Zählungen auf 2 % und 2, Gleitkommazahlen auf 1e-6 relativ), weil Gleitkomma-Summen und die Wahl der
    Ecke auf anderen Plattformen abweichen können;
(3) die Zahlen der Beschreibung der Tests (Zahl der Instanzen der Orakel) und die Testzahl stehen in der README, wie sie sind."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import ugm_constants as C
import ugm_evaluation as E
from orakel_zahlen import ORAKEL
from readme_zeilen import J, saetze, tabellenzeilen, zeilen_weich

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
V = json.loads((ROOT / "tools" / "vorgerechnet.json").read_text(encoding="utf-8"))
ZAEHLFELDER = {"mit_leer", "teurer", "nicht_voll", "anzahl", "anzahl_aus", "erste_ueber_untergrenze", "alle_ueber_untergrenze", "ueber_bestand", "ueber_beide", "schlupf_ganz", "nicht_konvergiert"}
GROSSE = ("ersparnis_liste", "rho_voll_liste", "raster", "raster_bewegt", "raster_bedient", "rho_alle_liste")


def naehe(a, b, schluessel="", rel=1e-6, abs_=1e-12):
    """Band-Vergleich zweier Ergebnisse (verschachtelte Listen und Dicts): Zählfelder auf 2 % und 2, Zählungen aus Mittelwerten (Produkte) auf 5 %, sonst relativ `rel`;
    Texte und Wahrheitswerte exakt."""
    if isinstance(b, dict):
        assert set(a) == set(b), (schluessel, set(a) ^ set(b))
        for k in b:
            naehe(a[k], b[k], k, rel, abs_)
    elif isinstance(b, list):
        assert len(a) == len(b), schluessel
        for x, y in zip(a, b):
            naehe(x, y, schluessel, rel, abs_)
    elif isinstance(b, bool) or isinstance(b, str) or b is None:
        assert a == b, schluessel
    elif schluessel in ZAEHLFELDER:
        assert abs(a - b) <= max(2.0, 0.02 * abs(b)), (schluessel, a, b)
    elif schluessel == "produkte_median":
        assert a == pytest.approx(b, rel=0.05), (schluessel, a, b)
    else:
        assert a == pytest.approx(b, rel=rel, abs=abs_), (schluessel, a, b)


def _kurz(x):
    return E.jsonbar(_ohne_grosse(E.jsonbar(x)))


def _ohne_grosse(x):
    if isinstance(x, dict):
        return {k: _ohne_grosse(v) for k, v in x.items() if k not in GROSSE}
    if isinstance(x, list):
        return [_ohne_grosse(v) for v in x]
    return x


# ----------------------------------------------------------------------------------------------------------------------------------
# Meta, Zeilen, Sätze
# ----------------------------------------------------------------------------------------------------------------------------------

def test_meta_der_messreihen_ist_der_stand_der_demo():
    d = C.DEFAULTS
    assert J["meta"] == {"seed": 2026, "anzahl": 200, "n_kunden": 30, "rho1": 1.0, "ueberhang2": 1.3, "anteil3": 5, "abstand3": 10.0, "anzahl3": 200, "anzahl_weich": 10}
    assert C.SERIEN_SEED == 2026 and d["inst1"] == 200 and d["n_kunden"] == 30 and d["rho1"] == 1.0 and d["anteil3"] == 5 and d["abstand3"] == 10.0 and d["inst3"] == 200 and C.ANZAHL_WEICH == 10
    for satz in ("Zufalls-Seed 2026", "200 Instanzen je Reihe", "5 Ziele bei 10 ± 0,1", "höchstens 1e-7 Einheiten geliefert", "unter 1e-9 (float64)"):
        assert satz in README, satz


def test_tabellenzeilen_der_readme():
    for zeile in tabellenzeilen():
        assert zeile in README, zeile


def test_saetze_der_readme_sind_die_verdikte_der_app():
    for name, satz in saetze().items():
        assert satz in README, name


def test_weich_tabelle_hat_ihre_kopfzeilen_mit_trennzeile():
    z = zeilen_weich(J["weich"])
    assert z[0] + "\n|---|---|---|---|---|---|" in README and z[4] + "\n|---|---|---|---|---|" in README


def test_ansicht_zahlen_stehen_in_der_standardstrasse_der_app():
    a = E.ansicht(E.Settings())
    assert J["ansicht"]["norm_kosten"] == pytest.approx(a.norm.bewegungskosten, rel=1e-6) and J["ansicht"]["teil_kosten"] == pytest.approx(a.teil.bewegungskosten, rel=1e-6)
    assert J["ansicht"]["zaehl_teil"] == list(a.zaehl_teil) and J["ansicht"]["seed"] == C.DEFAULTS["seed_input"]


# ----------------------------------------------------------------------------------------------------------------------------------
# Neu rechnen als Band
# ----------------------------------------------------------------------------------------------------------------------------------

def test_tabelle_1_neu_gerechnet():
    neu = _kurz(E.reihen_ueberhang(C.DEFAULTS["rho1"], C.DEFAULTS["n_kunden"], C.DEFAULTS["inst1"]))
    naehe(neu, J["tab1"])


def test_reihen_bei_kleinem_rho_und_60_kunden_neu_gerechnet():
    naehe(_kurz(E.reihe_normalisieren(1.3, 0.1, 30, 200)), J["tab1_rho01"])
    naehe(_kurz(E.reihe_normalisieren(1.3, 1.0, 60, 200)), J["tab1_n60"])


def test_teilbedienung_nie_teurer_in_den_fuenf_reihen_und_bei_rho_1_immer_der_ganze_bestand():
    assert sum(z["teurer"] for z in J["tab1"]) == 0 and all(z["nicht_voll"] == 0 for z in J["tab1"]) and J["tab1_n60"]["nicht_voll"] == 0
    assert 0 < J["tab1_rho01"]["nicht_voll"] < J["tab1_rho01"]["anzahl"]


def test_streuung_von_nicht_voll_bei_anderen_zufalls_seeds_neu_gerechnet():
    """Die README nennt die Spanne der Zahl „nicht den ganzen Bestand“ bei anderen Zufalls-Seeds der Reihe (ρ = 0,1, R = 1,3); Zählungen als Band (± 2)."""
    sd = J["tab1_rho01_seeds"]
    neu = [E.reihe_normalisieren(1.3, 0.1, C.DEFAULTS["n_kunden"], C.DEFAULTS["inst1"], seed=s)["nicht_voll"] for s in sd["seeds"]]
    naehe(neu, sd["nicht_voll"], "nicht_voll")
    assert min(sd["nicht_voll"]) < J["tab1_rho01"]["nicht_voll"] < max(sd["nicht_voll"]) + 1 and len(sd["seeds"]) == 5
    assert f"bei 5 anderen Zufalls-Seeds der Reihe {min(sd['nicht_voll'])} bis {max(sd['nicht_voll'])} von 200" in README


def test_mittel_ueber_betroffene_und_ueber_alle_instanzen_bei_rho_0_1_sind_verschieden_und_beide_genannt():
    k = J["tab1_rho01"]
    assert k["bewegt_nicht_voll"][0] < k["bewegt"][0] < 1.0 and E.prozent(k["bewegt_nicht_voll"][0], 1) == "92,4 %" and E.prozent(k["bewegt"][0], 1) == "96,1 %"
    assert "in diesen Instanzen im Mittel 92,4 %, über alle 200 Instanzen im Mittel 96,1 % des Bestands bewegt" in README
    assert J["tab1_n60"]["bewegt_nicht_voll"] is None and all(z["bewegt_nicht_voll"] is None for z in J["tab1"])


def test_readme_nennt_96_1_und_96_2_aus_verschiedenen_instanzstroemen():
    assert E.prozent(J["tab1_rho01"]["bewegt"][0], 1) == "96,1 %" and E.prozent(J["strafe"]["bewegt"][J["strafe"]["rho"].index(0.1)][0], 1) == "96,2 %"
    assert C.ZWECK_REIHE1 != C.ZWECK_STRAFE and "beide Reihen benutzen verschiedene Instanzen" in README


def test_reihe_strafe_neu_gerechnet_und_rho_voll_ist_die_halbe_grenzkosten():
    neu = _kurz(E.reihe_strafe(C.RHO_TABELLE, 1.3, 30, 200))
    naehe(neu, J["strafe"])
    assert J["strafe"]["rho_voll"][0] == pytest.approx(J["strafe"]["grenzkosten"][0] / 2, rel=1e-12)


def test_reihe_ausreisser_neu_gerechnet_und_vorrechnung_gleich_messreihe():
    neu = E.jsonbar(E.reihe_ausreisser(C.DEFAULTS["anteil3"], C.DEFAULTS["abstand3"], C.DEFAULTS["inst3"]))
    naehe(_ohne_grosse(neu), J["ausreisser"])
    vor = V["reihe_ausreisser|abstand=10.0,anteil_prozent=5,anzahl=200"]
    naehe(_ohne_grosse(vor), J["ausreisser"])
    naehe(vor["raster_bedient"], neu["raster_bedient"])
    naehe(vor["rho_alle_liste"], neu["rho_alle_liste"])
    pr = J["ausreisser_preset"]                                           # Preset „Ausreißer 5 %“: ρ = 6, alle fünf bedient in n von 200 Instanzen (Band ± 2)
    assert pr["rho"] == C.PRESETS["Ausreißer 5 %"]["rho3"] == 6.0 and pr["anzahl"] == 200
    assert abs(E.anzahl_alle_bedient(neu, pr["rho"]) - pr["alle_bedient"]) <= 2 and abs(E.anzahl_alle_bedient(vor, pr["rho"]) - pr["alle_bedient"]) <= 2
    hilfe = C.PRESET_HELP["Ausreißer 5 %"]
    assert f"in {pr['alle_bedient']} von {pr['anzahl']} Instanzen alle fünf bedient (Mittel von ρ_alle: {E.de(J['ausreisser']['rho_alle'][0], 2)})" in hilfe
    assert "reicht im Mittel" not in hilfe


def test_weich_messung_neu_gerechnet():
    naehe(E.jsonbar(E.weich_messung()), J["weich"])


def test_weich_zahlen_der_demo_nennen_die_zwei_mal_zwei_werte_der_geschlossenen_form():
    assert [round(x, 3) for x in J["weich"]["zwei_mal_zwei"]] == [3.245, 1.732, 1.21, 1.147]
    assert "| Masse im 2 × 2-Beispiel | 3,245 | 1,732 | 1,210 | 1,147 | 1,1402 |" in README


# ----------------------------------------------------------------------------------------------------------------------------------
# Beschreibung der Tests und Texte
# ----------------------------------------------------------------------------------------------------------------------------------

def test_zahlen_der_orakel_stehen_in_der_readme():
    o = ORAKEL
    for satz in (f"({o['schlupf']} Straßen, 5 Werte von ρ, relativ 1e-9)", f"({o['aufzaehlung']} kleine Aufgaben)", f"({o['kappung']} Instanzen, 5 Werte von ρ)",
                 f"({o['kurve']} Straßen, 6 Werte von m)", f"({o['rho_voll']} Straßen)", f"({o['ausreisser']} Instanzen)", f"({o['kl']} Instanzen, zwei Werte von τ/ε)",
                 f"({o['wasserstein']} Straßen)", f"({o['eindeutig']} Straßen, Spannweite unter 1e-5"):
        assert satz in README, satz
    assert "Abweichung höchstens 2e-8" in README and "25 Straßen" in README and o["rho_voll"] == 25
    quelle = (ROOT / "tests" / "test_oracle_ungleiche_mengen.py").read_text(encoding="utf-8")
    assert "abs=2e-8" in quelle and "< 1e-5" in quelle and "abs(dq / 2 - k.rho_voll) < 1e-2" in quelle


def test_testzahl_der_readme():
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=300)
    zahl = int(re.search(r"(\d+) tests? collected", r.stdout).group(1))
    assert f"**Testzahl:** {zahl} Tests (`pytest --collect-only -q tests`)" in README, zahl


def test_readme_und_app_nennen_keine_planbegriffe_und_keine_umgebungsaussagen():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for text in (README, app):
        for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Saatwert", "trägt", "tragen", "Vormessung", "Mutationstest", "Stück", "V1", "O6", "TODO ", "Plan-"):
            if verboten == "TODO " and text is README:
                continue                                                # der Live-Link-Kommentar bleibt bis zum Deploy
            assert verboten not in text, verboten
    assert not re.search(r"\bSeed\b", re.sub(r"Zufalls-Seed", "", README + app)), "„Seed“ nur als „Zufalls-Seed“"
    assert "Orakel" not in app, "die Orakel-Wörter gehören in Tests und README, nicht in die Anzeigetexte der App"


def test_keine_scheinpruefungen_in_den_tests():
    muster = "or " + "True"
    for datei in (ROOT / "tests").glob("*.py"):
        assert muster not in datei.read_text(encoding="utf-8"), datei.name


def test_readme_nennt_die_wichtigen_aussagen_mit_ihren_voraussetzungen():
    assert "Abschnitt 2" not in README and "Abschnitt 2" not in (ROOT / "app.py").read_text(encoding="utf-8"), "Chizat u. a.: nur der Abstract gelesen"
    assert "hungarian-demo" not in README and "hungarian-demo" not in (ROOT / "app.py").read_text(encoding="utf-8"), "Aussage über die andere Demo nicht belegt"
    for satz in ("Nietert, Goldfeld, Cummings (2022)", "arXiv 2111.01361: nur der Abstract", "Math. Comp. 87, arXiv 1607.05816: nur der Abstract", "Rubner, Tomasi, Guibas (2000)", "Hilfsknoten", "Strafe je vernichtete und je erzeugte Einheit", "scipy.stats.wasserstein_distance",
                 "Die Strafe ρ steht in keinen Daten", "nicht gegen eine Bibliothek geprüft", "Nicht mit echten Mausklicks im Browser durchgeklickt"):
        assert satz in README, satz

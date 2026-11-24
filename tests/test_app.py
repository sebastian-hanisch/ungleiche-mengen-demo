"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine Grenzen und
prüft, dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=) und
`StreamlitAPIException` bei Slidern mit berechneten Grenzen. Dazu: je Preset die erwartete Aussage, Permalink, Randfälle, Worst-Case am Regler-Maximum.

Laufzeit: Die Ausreißer-Reihe ist für die Standardeinstellungen vorgerechnet (tools/vorgerechnet.json); jede andere Einstellung rechnet live (Zwischenspeicher gilt im Prozess).
Die Tests wählen darum eine kleine Ausgangslage (Adresszeile: 10 Kunden, 10 Instanzen in den Reihen 1 und 2, 20 in Reihe 3), wo es nur um das Zusammenspiel der Regler geht."""

import os
import re

import pytest
from streamlit.testing.v1 import AppTest

import ugm_constants as C
import ugm_evaluation as E
import ugm_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle")
KLEIN = dict(n="10", i1="10", i3="20")     # Adresszeile der kleinen Ausgangslage


def _app(**adresse) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for k, v in adresse.items():
        at.query_params[k] = v
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _klein() -> AppTest:
    return _app(**KLEIN)


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def _setze(at, **werte):
    for key, wert in werte.items():
        _regler(at, key).set_value(wert)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (werte, [str(e) for e in at.exception])
    return at


def _texte(at) -> str:
    return " ".join(str(e.value) for kind in ("success", "info", "warning", "markdown", "caption") for e in getattr(at, kind))


def _metrics(at) -> dict:
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_all_sections():
    at = _app()
    titel = " ".join(s.value for s in at.subheader)
    for abschnitt in ("Experiment 1: Normalisieren oder Teilbedienung", "Experiment 2: Die Strafe ρ", "Experiment 3: Ausreißer", "Gesamtmasse bei der Strafe", "Wo die Annahmen enden"):
        assert abschnitt in titel, abschnitt
    assert [e.label for e in at.expander if "Mathematische Formulierung" in e.label] == ["📐 Mathematische Formulierung"]
    assert any("Annahmen, Grenzen und bekannte Probleme" in e.label for e in at.expander)
    assert len(at.metric) == 6 + 4 + 3 and len(at.info) >= 8 and not at.warning
    assert len({p.key for p in at.get("plotly_chart")}) == len(at.get("plotly_chart")) == 7         # jedes Diagramm hat einen eigenen Schlüssel


def test_every_button_click_does_not_raise():
    at = _klein()
    labels = [b.label for b in at.button]
    assert len(labels) == len(C.PRESETS) + 1, labels                      # fünf Presets und der Zufalls-Seed-Button
    for index, label in enumerate(labels):
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    at = _klein()
    schluessel = [s.key for s in at.sidebar.slider]
    assert schluessel == ["n_kunden", "abstand3", "rho3"], schluessel
    for key in schluessel:
        for edge in ("min", "max"):
            regler = _regler(at, key)
            wert = regler.min if edge == "min" else regler.max
            regler.set_value(int(wert) if isinstance(regler.value, int) else float(wert)).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


# ----------------------------------------------------------------------------------------------------------------------------------
# Auswahlregler und Presets
# ----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key,optionen", [
    ("quelle", C.QUELLEN),
    ("ueberhang", (C.UEBERHANG_STUFEN[0], 1.15, C.UEBERHANG_STUFEN[-1])),
    ("rho1", (C.RHO1_STUFEN[0], 0.06, C.RHO1_STUFEN[-1])),
    ("inst1", C.INSTANZEN1_STUFEN[:2]),
    ("rho2", (C.RHO2_STUFEN[0], C.RHO2_STUFEN[-1])),
    ("kurve", (False, True)),
    ("anteil3", C.ANTEIL3_STUFEN),
    ("inst3", C.INSTANZEN3_STUFEN[:2]),
])
def test_optionen_der_auswahlregler_laufen_ohne_fehler(key, optionen):
    at = _klein()
    for option in optionen:
        _setze(at, **{key: option})


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_ohne_fehler_und_zeigt_seine_einstellung(name):
    at = _klein()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key, wert in C.PRESETS[name].items():
        assert at.session_state[key] == wert, (name, key)


def test_preset_von_hand_zeigt_28_drittel_vier_und_57_komma_1_prozent():
    at = _klein()
    at.sidebar.button(key="preset_Von Hand: 2 Lager, 3 Kunden").click().run(timeout=TIMEOUT)
    m = _metrics(at)
    assert m["Bewegungskosten beim Normalisieren"] == "9,33" and m["Bewegungskosten bei Teilbedienung"] == "4,00"
    assert m["Kunden voll / leer / teilweise (Teilbedienung)"] == "2 / 1 / 0" and m["Zielwert mit Strafe, Teilbedienung"] == "6,00"
    deltas = " ".join(str(x.delta) for x in at.metric)
    assert "57,1 % weniger (Basis Normalisieren)" in deltas and "jeder Kunde 66,7 % seiner Nachfrage" in deltas
    text = _texte(at)
    assert "Von Hand: 2 Lager, 3 Kunden: 2 Lager (Bestand 4), 3 Kunden (Nachfrage 6), Überhang R = 1,5" in text
    assert _metrics(at)["Marke ρ_voll = Grenzkosten / 2"] == "0,05"
    # Der Regler ρ von Experiment 2 steht genau auf ρ_voll: „0 von 4“ und „oberhalb von ρ_voll“, kein „ab ρ_voll“
    assert "Bei ρ = 0,05 Straßenlängen bewegt die Teilbedienung 0 von 4 Einheiten (0,0 %)" in text and "wird oberhalb von ρ_voll = 0,05 Straßenlängen bewegt" in text
    kern = [e.value for e in at.info if e.value.startswith("Bei ρ = 0,05 Straßenlängen")]
    assert len(kern) == 1 and "Genau bei ρ = ρ_voll gibt es mehrere gleich gute Pläne" in kern[0] and "ab ρ_voll" not in kern[0]


def test_preset_gleiche_summe_meldet_keine_ersparnis_und_keinen_leeren_kunden():
    at = _klein()
    at.sidebar.button(key="preset_Gleiche Summe").click().run(timeout=TIMEOUT)
    assert not at.exception
    m = _metrics(at)
    assert m["Bewegungskosten beim Normalisieren"] == m["Bewegungskosten bei Teilbedienung"]
    assert "0,0 % weniger" in " ".join(str(x.delta) for x in at.metric) and m["Kunden voll / leer / teilweise (Teilbedienung)"].split(" / ")[1] == "0"
    text = _texte(at)
    assert "dieselbe Summe" in text and "R = 1 und ρ ≥ ρ_voll: beide Rechnungen sind dasselbe Problem" in text and "Teilbedienung kostet im Mittel" not in text
    deltas = [str(x.delta) for x in at.metric]
    assert "0 Einheiten vernichtet oder erzeugt" in " ".join(deltas) and not [d for d in deltas if "e-" in d]       # kein Gleitkomma-Rest wie „2,842e-14“


@pytest.mark.parametrize("rho1", [0.025, 0.04, 0.06, 0.1])
def test_gleiche_summe_mit_kleinem_rho_sagt_nicht_dasselbe_problem(rho1):
    """Codefehler der Prüfung: R = 1 mit ρ unter ρ_voll darf weder in der Kernfrage noch im Verdikt von Experiment 1 „dasselbe Problem“ sagen."""
    at = _app(i1="10", i3="20")                          # 30 Kunden wie in der Prüfung (Zufalls-Seed 35)
    at.sidebar.button(key="preset_Gleiche Summe").click().run(timeout=TIMEOUT)
    _setze(at, rho1=rho1)
    text = _texte(at)
    assert "beide Rechnungen sind dasselbe Problem" not in text
    assert "die beiden Rechnungen sind verschieden" in text and "liegt ρ unter ρ_voll" in text
    at.sidebar.select_slider(key="rho1").set_value(1.0)
    at.run(timeout=TIMEOUT)
    assert "beide Rechnungen sind dasselbe Problem" in _texte(at)


def test_standard_straße_meldet_eine_ersparnis_und_leere_kunden():
    at = _klein()
    text = _texte(at)
    assert "Die Teilbedienung kostet im Mittel" in text and "weniger Bewegungskosten als das Normalisieren" in text
    assert "Normalisieren gibt jedem der 10 Kunden denselben Anteil 76,9 % seiner Nachfrage" in text and "Beide Pläne lassen dieselben" in text


def test_preset_schwache_strafe_bewegt_nur_einen_teil_des_bestands():
    at = _klein()
    at.sidebar.button(key="preset_Schwache Strafe").click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["rho2"] == 0.025
    anteil = float(_metrics(at)["Bewegter Anteil des Bestands"].replace(" %", "").replace(",", "."))
    assert 0.0 < anteil < 100.0 and "bewegt die Teilbedienung" in _texte(at)


def test_preset_ausreisser_zeigt_die_bedienten_ausreisser_der_instanz_der_seitenleiste():
    at = _klein()
    at.sidebar.button(key="preset_Ausreißer 5 %").click().run(timeout=TIMEOUT)
    assert not at.exception
    d = E.ausreisser_instanz(5, 10.0, C.DEFAULTS["seed_input"], C.ZWECK_ANSICHT, 0)
    erwartet = f"Bei ρ = 6 Standardabweichungen werden {E.ausreisser_bedient(d, 6.0)} von 5 Ausreißern bedient"
    assert erwartet in _texte(at)


def test_ohne_ausreisser_meldet_die_app_keinen_anteil():
    at = _klein()
    _setze(at, anteil3=0)
    text = _texte(at)
    assert "Ohne Ausreißer" in text and "gibt es keinen Anteil und keinen Umschlag" in text


def test_kurve_laesst_sich_ausschalten_und_verschwindet():
    at = _klein()
    mit = len(at.get("plotly_chart"))
    _setze(at, kurve=False)
    assert len(at.get("plotly_chart")) == mit - 1


# ----------------------------------------------------------------------------------------------------------------------------------
# Permalink, Randfälle, Worst-Case, Texte
# ----------------------------------------------------------------------------------------------------------------------------------

def test_permalink_uebernimmt_werte_begrenzt_und_eingerastet():
    at = _app(n="9999", r="1.32", p1="0.3", q="unbekannt", p2="0.04", d3="99", p3="-4", a3="4", kv="False", i1="10", i3="20")
    s = at.session_state
    assert s["n_kunden"] == C.N_MAX and s["ueberhang"] == 1.3 and s["rho1"] in (0.25, 0.5) and s["quelle"] == C.DEFAULTS["quelle"] and s["rho2"] in (0.025, 0.05)
    assert s["abstand3"] == C.ABSTAND_MAX and s["rho3"] == C.RHO3_MIN and s["anteil3"] in (3, 5) and s["kurve"] is False


def test_adresszeile_spiegelt_die_einstellungen():
    at = _setze(_klein(), n_kunden=12, ueberhang=1.6)
    assert at.query_params["n"] in (["12"], "12") and at.query_params["r"] in (["1.6"], "1.6") and at.query_params["i1"] in (["10"], "10")


def test_randfaelle_alle_minima_und_seed_0():
    at = _klein()
    _setze(at, quelle=C.QUELLE_STRASSE, ueberhang=1.0, rho1=C.RHO1_STUFEN[0], n_kunden=C.N_MIN, seed_input=0, inst1=C.INSTANZEN1_STUFEN[0], rho2=0.0, kurve=False,
           anteil3=0, abstand3=C.ABSTAND_MIN, rho3=0.0, inst3=C.INSTANZEN3_STUFEN[0])


def test_worst_case_alle_maxima_zugleich_ohne_fehler():
    """Regler-Maximum: 60 Kunden, R = 2, ρ = 1,5, 200 Instanzen, 10 % Ausreißer bei 15 Standardabweichungen, 200 Instanzen mit 100 Quellen und 100 Zielen. Die Zeit wird nicht geprüft."""
    at = _app()
    _setze(at, n_kunden=C.N_MAX, ueberhang=C.UEBERHANG_STUFEN[-1], rho1=C.RHO1_STUFEN[-1], inst1=C.INSTANZEN1_STUFEN[-1], rho2=C.RHO2_STUFEN[-1], anteil3=C.ANTEIL3_STUFEN[-1],
           abstand3=C.ABSTAND_MAX, rho3=C.RHO3_MAX, inst3=C.INSTANZEN3_STUFEN[-1])
    assert len(at.metric) == 13


def test_die_ansicht_nennt_keine_planbegriffe_und_keine_umgebungsaussagen():
    at = _app()
    gesamt = _texte(at) + " " + " ".join(e.label for e in at.expander) + " " + " ".join(s.value for s in at.subheader)
    for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Orakel", "Saatwert", "Sinkhorn-Distanz", "Sinkhorn-Divergenz", "wasserstein", "Stück", "Vormessung"):
        assert verboten not in gesamt, verboten
    assert not re.search(r"\btr[äa]gt\b|\btragen\b", gesamt)
    assert "Zufalls-Seed" in " ".join(w.label for w in at.sidebar.number_input)


def test_jeder_regler_der_seitenleiste_hat_einen_hilfetext():
    at = _app()
    for art in REGLER_ARTEN:
        for w in getattr(at.sidebar, art):
            if w.key in P.SETTING_SPECS and w.key != "seed_input":
                assert w.help, w.key


def test_hilfetexte_verweisen_nur_auf_vorhandene_bloecke_und_nennen_die_bedingung_bei_gleicher_summe():
    at = _app()
    hilfe = at.sidebar.selectbox(key="quelle").help
    assert "„Ergebnis: eine Straße“" in hilfe and "Zeile" not in hilfe and "Die Reihen von Experiment 1 bleiben Zufallsstraßen" in hilfe
    assert "## 🎯 Ergebnis: eine Straße" in " ".join(str(m.value) for m in at.markdown)           # der Block, auf den die Hilfe verweist, existiert
    assert "solange ρ nicht unter ρ_voll liegt" in C.PRESET_HELP["Gleiche Summe"]

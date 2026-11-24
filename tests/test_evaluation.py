"""Verdikte (reine Funktionen in ugm_evaluation.py) an handgesetzten Zeilen: Schwellen, Verzweigungen und jede Zahl im Text; dazu Formatierung und Reihen im Kleinen."""

import numpy as np
import pytest

import ugm_constants as C
import ugm_evaluation as E


def zeile_r(**kw):
    basis = dict(ueberhang=1.3, rho=1.0, n_kunden=30, anzahl=200, ersparnis=[0.331, 0.153, 0.537], leer=[0.196, 0.1, 0.3], voll=[0.7, 0.5, 0.8], teilweise=[0.1, 0.0, 0.2],
                 norm_kosten=[11.6, 9.0, 14.0], teil_kosten=[7.7, 5.0, 10.0], bewegt=[1.0, 1.0, 1.0], mit_leer=200, teurer=0, nicht_voll=0, ersparnis_liste=[0.3], bewegt_nicht_voll=None)
    basis.update(kw)
    return basis


# ----------------------------------------------------------------------------------------------------------------------------------
# Formatierung
# ----------------------------------------------------------------------------------------------------------------------------------

def test_formatierung_mit_dezimalkomma_ohne_minus_null_und_mit_leerzeichen_als_tausendertrenner():
    assert E.de(1.234, 2) == "1,23" and E.de(-0.0001, 2) == "0,00" and E.de(-1e-17, 1) == "0,0" and E.de(-0.5, 1) == "-0,5"
    assert E.de_g(0.00123, 2) == "0,0012" and E.de_g(0, 3) == "0" and E.de_g(1e-9, 2) == "1e-9"
    assert E.ganz(1938) == "1 938" and E.prozent(0.25, 1) == "25,0 %"
    assert E.strl(1.0) == "1 Straßenlänge" and E.strl(0.05) == "0,05 Straßenlängen" and E.strl(0.1) == "0,1 Straßenlängen"
    assert E.mittel_text([0.5, 0.25, 0.75]) == "50,0 % [25,0; 75,0]" and E.mittel_text([1.5, 1.25, 1.75], 2, False) == "1,50 [1,25; 1,75]"
    assert E.mittel_text([-1e-17, -2e-17, 0.0]) == "0,0 % [0,0; 0,0]"


def test_stat_ist_mittel_minimum_maximum_und_jsonbar_gibt_reine_typen():
    assert E.stat([1, 2, 6]) == [3.0, 1.0, 6.0]
    j = E.jsonbar({"a": np.float64(1.5), "b": (np.int64(2), np.array([1.0, 2.0])), "c": np.bool_(True), 3: "x"})
    assert j == {"a": 1.5, "b": [2, [1.0, 2.0]], "c": True, "3": "x"} and type(j["a"]) is float and type(j["c"]) is bool


def test_settings_aus_zustand_wandelt_typen_und_nimmt_standardwerte_fuer_fehlende_schluessel():
    s = E.settings_aus_zustand({"n_kunden": 12.0, "rho1": 1, "kurve": 0, "anteil3": 3.0})
    assert s.n_kunden == 12 and isinstance(s.n_kunden, int) and s.rho1 == 1.0 and isinstance(s.rho1, float) and s.kurve is False and s.anteil3 == 3
    assert s.inst1 == C.DEFAULTS["inst1"] and s.quelle == C.DEFAULTS["quelle"]


# ----------------------------------------------------------------------------------------------------------------------------------
# Verdikte
# ----------------------------------------------------------------------------------------------------------------------------------

def test_verdikt_normalisieren_bei_r_1_ist_gruen_und_sagt_keine_ersparnis():
    art, text = E.verdikt_normalisieren(zeile_r(ueberhang=1.0, ersparnis=[1e-16, -1e-16, 2e-16], leer=[0.0, 0.0, 0.0], mit_leer=0))
    assert art == "success" and "R = 1 und ρ ≥ ρ_voll: beide Rechnungen sind dasselbe Problem" in text and "die Ersparnis ist 0 (Abweichung unter 1e-9" in text and "in 0 von 200 Instanzen" in text


def test_verdikt_normalisieren_bei_r_1_und_rho_unter_rho_voll_sagt_nicht_dasselbe_problem():
    """Codefehler der Prüfung: bei R = 1 stand „dasselbe Problem“ auch dann, wenn die Teilbedienung nicht den ganzen Bestand bewegt (z. B. ρ = 0,025)."""
    art, text = E.verdikt_normalisieren(zeile_r(ueberhang=1.0, rho=0.025, ersparnis=[0.8, 0.5, 0.99], leer=[0.5, 0.2, 0.9], mit_leer=200, nicht_voll=190, bewegt=[0.5, 0.2, 1.0], bewegt_nicht_voll=[0.45, 0.2, 0.9]))
    assert art == "info" and "beide Rechnungen sind dasselbe Problem" not in text
    assert "R = 1, ρ = 0,025 Straßenlängen" in text and "In 190 von 200 Instanzen liegt ρ unter ρ_voll" in text and "in diesen im Mittel 45,0 % bewegt" in text
    assert "größte Abweichung der Ersparnis von 0 ist 0,99" in text and "in 200 von 200 Instanzen ist ein Kunde leer" in text
    art, text = E.verdikt_normalisieren(zeile_r(ueberhang=1.0, ersparnis=[1e-3, -1e-3, 2e-3], nicht_voll=0))          # Abweichung über 1e-9: auch dann kein „dasselbe Problem“
    assert art == "info" and "R = 1 und ρ ≥ ρ_voll: beide" not in text


def test_verdikt_normalisieren_nennt_mittel_spanne_leer_und_gleiche_ungelieferte_menge():
    art, text = E.verdikt_normalisieren(zeile_r())
    assert art == "info"
    assert "R = 1,3, ρ = 1 Straßenlänge, 30 Kunden" in text and "im Mittel 33,1 % weniger" in text and "einzelne Instanzen 15,3 bis 53,7 %" in text
    assert "19,6 % der Kunden leer" in text and "in 200 von 200 Instanzen dieser Reihe mindestens einer" in text and "dieselben Einheiten ungeliefert" in text
    assert "In allen 200 Instanzen bewegt die Teilbedienung den ganzen Bestand" in text and "Teurer als das Normalisieren war die Teilbedienung in 0 Instanzen" in text
    assert text.index("den ganzen Bestand (ρ ≥ ρ_voll)") < text.index("dieselben Einheiten ungeliefert")          # die Bedingung steht vor der Aussage


def test_verdikt_normalisieren_warnt_wenn_rho_unter_rho_voll_liegt():
    art, text = E.verdikt_normalisieren(zeile_r(rho=0.1, nicht_voll=103, bewegt=[0.961, 0.746, 1.0], bewegt_nicht_voll=[0.924, 0.746, 0.999]))
    assert "In 103 von 200 Instanzen liegt ρ unter ρ_voll" in text and "für weniger bewegte Einheiten" in text
    assert "in diesen Instanzen im Mittel 92,4 %, über alle 200 Instanzen im Mittel 96,1 % des Bestands bewegt" in text          # die Bezugsmenge des Mittels steht dabei
    assert "In allen 200 Instanzen bewegt" not in text and "dieselben Einheiten ungeliefert" not in text


def test_verdikt_strafe_nennt_die_stufen_den_umschlag_und_dass_die_strafe_in_keinen_daten_steht():
    r = dict(ueberhang=1.3, n_kunden=30, anzahl=200, rho=[0.025, 0.05, 0.1], bewegt=[[0.4, 0.1, 0.7], [0.726, 0.4, 0.97], [0.962, 0.7, 1.0]], leer=[[0.69, 0.4, 0.9], [0.428, 0.2, 0.6], [0.231, 0.1, 0.4]],
             rho_voll=[0.109, 0.051, 0.261], grenzkosten=[0.218, 0.1, 0.52])
    art, text = E.verdikt_strafe(r)
    assert "ρ = 0,05: im Mittel 72,6 % des Bestands bewegt (42,8 % der Kunden leer)" in text and "ρ = 0,1: im Mittel 96,2 %" in text and "ρ = 0,025" not in text
    assert "0,109 Straßenlängen (einzelne Instanzen 0,051 bis 0,261)" in text and "halben Grenzkosten" in text and "Mittel der Grenzkosten 0,218" in text and "keinen Daten" in text


def test_verdikt_ausreisser_leer_und_gefuellt():
    assert "Ohne Ausreißer (0 %)" in E.verdikt_ausreisser(dict(leer=True))[1]
    r = dict(leer=False, anzahl=200, anzahl_aus=5, anteil_prozent=5, abstand=10.0, anteil_a=[0.688, 0.476, 0.848], anteil_b=[0.747, 0.568, 0.874], anteil_c=[0.632, 0.403, 0.803],
             rho_erste=[4.14, 3.01, 5.82], rho_alle=[5.17, 4.12, 6.75], d_min2=[3.71, 3.01, 4.16], d_max2=[3.82, 3.1, 4.26], erste_ueber_untergrenze=200, alle_ueber_untergrenze=200, abstand_erste=0.43)
    art, text = E.verdikt_ausreisser(r)
    assert "5 von 100 Zielen (5 % der Masse), 10 Standardabweichungen" in text and "68,8 % der Wege der sortierten Zuordnung" in text and "48 bis 85 %" in text
    assert "gibt 74,7 %" in text and "63,2 %" in text and "keine Eigenschaft des Problems, sondern der Definition" in text
    assert "ab ρ = 4,14 bedient (einzelne Instanzen 3,01 bis 5,82), alle ab ρ = 5,17" in text and "in 200 von 200 Instanzen nicht unterschritten" in text and "0,43 Standardabweichungen darüber" in text
    assert "was zum Kleinbeispiel passt" in text and "weil ein bedienter Ausreißer" not in text


def test_verdikt_ausreisser_instanz_zaehlt_bediente_und_nennt_die_marken():
    d = dict(anzahl_aus=5, schwellen=[4.0, 4.5, 5.0, 5.5, 6.0], rho_erste=4.0, rho_alle=6.0, d_min2=3.5)
    assert "werden 0 von 5 Ausreißern bedient" in E.verdikt_ausreisser_instanz(d, 3.0)[1]
    t = E.verdikt_ausreisser_instanz(d, 5.2)[1]
    assert "werden 3 von 5 Ausreißern bedient" in t and "ab ρ = 4,00" in t and "ab ρ = 6,00" in t and "d_min/2 = 3,50" in t
    assert "5 von 5" in E.verdikt_ausreisser_instanz(d, 6.5)[1]
    assert "Ohne Ausreißer" in E.verdikt_ausreisser_instanz(dict(anzahl_aus=0), 3.0)[1]


def test_verdikt_weich_nennt_mittel_spannen_und_die_zaehlungen_ueber_dem_bestand():
    w = dict(anzahl=10, n=30, tau_eps=[0.1, 1.0, 10.0, 100.0], masse=[[9.0, 5.0, 15.0], [1.86, 1.27, 2.48], [1.07, 0.99, 1.12], [1.11, 1.11, 1.12]], ueber_bestand=[10, 10, 9, 10], ueber_beide=[10, 9, 0, 0],
             rho_opt=[0.1, 0.3, 1.0, 3.0], schlupf_ganz=[0, 0, 1, 10], schlupf_leer=[[13.8, 21], [5.7, 8], [0.6, 2], [0.0, 0]])
    art, text = E.verdikt_weich(w)
    assert "bei τ/ε = 1 hat der KL-Plan im Mittel 1,86 Masse (einzelne Instanzen 1,27 bis 2,48), in 9 von 10 Instanzen über beiden Randsummen" in text and "bei τ/ε = 10 im Mittel 1,07, 1 von 10 Instanzen unter dem Bestand 1" in text
    assert "bei τ/ε = 100 im Mittel 1,11, in 10 von 10 Instanzen über dem Bestand" in text and "ab ρ = 3 OPT den ganzen Bestand (10 von 10)" in text and "im Mittel 13,8 von 30 Lagern leer" in text
    assert "nicht gegen eine Bibliothek geprüft" in text


def test_verdikt_ansicht_zweigt_nach_gleicher_summe_ganzem_bestand_und_zu_kleinem_rho():
    gleich = E.ansicht(E.Settings(ueberhang=1.0))
    art, text = E.verdikt_ansicht(gleich)
    assert art == "success" and "dieselbe Summe" in text and "R = 1" in text and "beide Rechnungen sind dasselbe Problem" in text
    voll = E.ansicht(E.Settings())
    art, text = E.verdikt_ansicht(voll)
    assert art == "info" and "denselben Anteil 76,9 %" in text and "bewegt den ganzen Bestand" in text and "dieselben 30 Einheiten ungeliefert" in text
    klein = E.ansicht(E.Settings(rho1=0.025))
    art, text = E.verdikt_ansicht(klein)
    assert "(ρ liegt unter ρ_voll" in text and "ρ ≥ ρ_voll wählen" in text and "bewegt den ganzen Bestand" not in text


@pytest.mark.parametrize("rho1", C.RHO1_STUFEN)
def test_verdikt_ansicht_bei_r_1_sagt_dasselbe_problem_nur_wenn_es_stimmt_je_stufe_von_rho(rho1):
    """Jede Stufe des Reglers ρ bei R = 1 (darunter ρ = 0,025): „dasselbe Problem“ steht genau dann im Text, wenn die Bewegungskosten beider Rechnungen gleich sind."""
    an = E.ansicht(E.Settings(ueberhang=1.0, rho1=rho1))
    art, text = E.verdikt_ansicht(an)
    gleich = abs(an.norm.bewegungskosten - an.teil.bewegungskosten) < 1e-9
    assert ("beide Rechnungen sind dasselbe Problem" in text) == gleich, (rho1, text)
    assert gleich == an.ganzer_bestand and art == ("success" if gleich else "info")
    if not gleich:
        assert "die beiden Rechnungen sind verschieden" in text and "ρ ≥ ρ_voll wählen" in text


def test_verdikt_ansicht_bei_r_1_und_rho_0_025_nennt_die_kleine_bewegte_menge():
    """Randwert aus der Prüfung: Zufalls-Seed 35, R = 1, ρ = 0,025: Bewegungskosten 8,75 gegen 1,10 (verschieden, nicht dasselbe Problem)."""
    an = E.ansicht(E.Settings(ueberhang=1.0, rho1=0.025))
    text = E.verdikt_ansicht(an)[1]
    assert "8,75 gegen 1,1" in text and "ρ = 0,025 Straßenlängen liegt unter ρ_voll" in text and "bewegt nur" in text


def test_menge_zeigt_gleitkomma_reste_als_null():
    assert E.menge(2.842e-14) == "0" and E.menge(-3e-9) == "0" and E.menge(30.0) == "30" and E.menge(0.5) == "0,5" and E.menge(1.2e-6) == "1,2e-6"


def test_verdikt_rho_nennt_bewegte_menge_umschlag_und_dass_die_strafe_gesetzt_werden_muss():
    an = E.ansicht(E.Settings())
    art, text = E.verdikt_rho(an, 0.05)
    assert "Bei ρ = 0,05 Straßenlängen bewegt die Teilbedienung" in text and f"ρ_voll = {E.strl(an.rho_voll_rel)}" in text and "die Strafe selbst steht in keinen Daten, sie muss gesetzt werden" in text
    assert "bewegt die Teilbedienung 0 von 100 Einheiten" in E.verdikt_rho(an, 0.0)[1]
    assert "oberhalb von ρ_voll" in text and "ab ρ_voll" not in text and "Genau bei" not in text


def test_verdikt_rho_beim_handbeispiel_genau_auf_rho_voll_sagt_oberhalb_und_nennt_den_gleichstand():
    """Preset „Von Hand“: der Regler ρ von Experiment 2 steht auf 0,05 = ρ_voll; „0 von 4“ darf nicht neben „ab ρ_voll bewegt“ stehen."""
    an = E.ansicht(E.Settings(quelle=C.QUELLE_HAND, rho1=0.1))
    assert an.rho_voll_rel == pytest.approx(0.05)
    text = E.verdikt_rho(an, 0.05)[1]
    assert "bewegt die Teilbedienung 0 von 4 Einheiten (0,0 %)" in text and "oberhalb von ρ_voll = 0,05 Straßenlängen" in text
    assert "Genau bei ρ = ρ_voll gibt es mehrere gleich gute Pläne" in text and "ab ρ_voll" not in text
    text = E.verdikt_rho(an, 0.075)[1]
    assert "von 4 Einheiten (100,0 %)" in text and "Genau bei" not in text


# ----------------------------------------------------------------------------------------------------------------------------------
# Reihen im Kleinen
# ----------------------------------------------------------------------------------------------------------------------------------

def test_reihe_normalisieren_hat_die_erwarteten_felder_und_ist_reproduzierbar():
    a = E.reihe_normalisieren(1.3, 1.0, 12, 6)
    b = E.reihe_normalisieren(1.3, 1.0, 12, 6)
    assert a == b and a["anzahl"] == 6 and len(a["ersparnis_liste"]) == 6 and a["teurer"] == 0 and 0 < a["ersparnis"][0] < 1 and a["ersparnis"][1] <= a["ersparnis"][0] <= a["ersparnis"][2]
    assert a["voll"][0] + a["leer"][0] + a["teilweise"][0] == pytest.approx(1.0)
    assert E.reihe_normalisieren(1.0, 1.0, 12, 4)["mit_leer"] == 0 and a["bewegt_nicht_voll"] is None
    k = E.reihe_normalisieren(1.3, 0.025, 12, 6)                      # kleines ρ: die Mittel „bewegt“ über alle und über die betroffenen Instanzen unterscheiden sich
    assert k["nicht_voll"] > 0 and k["bewegt_nicht_voll"][0] < 1.0 and k["bewegt"][0] >= k["bewegt_nicht_voll"][0] and k["bewegt_nicht_voll"][2] < 1.0


def test_reihe_strafe_wachsender_anteil_und_rho_voll_unter_dem_raster_ende():
    r = E.reihe_strafe((0.01, 0.1, 0.25), 1.3, 12, 6)
    mittel = [b[0] for b in r["bewegt"]]
    assert mittel == sorted(mittel) and mittel[-1] <= 1.0 + 1e-9 and r["rho_voll"][0] == pytest.approx(r["grenzkosten"][0] / 2)
    assert len(r["raster"]) == len(r["raster_bewegt"]) == 61 and r["raster_bewegt"][0][0] == 0.0


def test_reihe_ausreisser_felder_und_leere_reihe():
    r = E.reihe_ausreisser(5, 10.0, 4)
    assert r["anzahl_aus"] == 5 and not r["leer"] and r["rho_erste"][0] <= r["rho_alle"][0] and r["erste_ueber_untergrenze"] == 4 and len(r["raster_bedient"]) == 101
    assert r["raster_bedient"][0][0] == 0.0 and r["raster_bedient"][-1][0] == 1.0
    assert E.reihe_ausreisser(0, 10.0, 3)["leer"] is True
    assert len(r["rho_alle_liste"]) == 4 and E.anzahl_alle_bedient(r, 0.0) == 0 and E.anzahl_alle_bedient(r, 1e6) == 4
    assert E.anzahl_alle_bedient(r, max(r["rho_alle_liste"]) + 1e-9) == 4 and E.anzahl_alle_bedient(r, max(r["rho_alle_liste"])) == 3          # strikt wie ausreisser_bedient


def test_weich_messung_im_kleinen_zaehlt_masse_und_schlupf():
    w = E.weich_messung(anzahl=2)
    assert w["anzahl"] == 2 and w["nicht_konvergiert"] == 0 and w["masse"][0][0] > w["masse"][1][0] > 1.0 and w["schlupf_bewegt"][-1][0] == pytest.approx(1.0)
    assert w["zwei_mal_zwei"][1] == pytest.approx(1.7324782, abs=1e-6) and w["grenzwert"] == pytest.approx(1.3 ** 0.5)

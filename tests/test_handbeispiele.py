"""Handbeispiele 1 bis 4 (die Rechnungen stehen im Expander „Mathematische Formulierung“ und in der README): jede Zahl hier ist von Hand gerechnet und steht als Bruch im Test.

Die Hilfsknoten-Aufgaben werden in Brüchen über alle aufspannenden Bäume gelöst (`ecken_brueche.py`, unabhängig vom Simplex der Demo); der Simplex der Demo muss denselben Wert geben."""

from fractions import Fraction as F

import numpy as np
import pytest

import ecken_brueche as EB
import ugm_evaluation as E
import ugm_kl as KL
import ugm_teiltransport as T
import ugm_ungleich as U
from conftest import BESTAND, NACHFRAGE, X_KUNDE, X_LAGER

CS = [[F(abs(int(x) - int(y))) for y in X_KUNDE] for x in X_LAGER]


# ----------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 1: Straße mit zwei Lagern und drei Kunden
# ----------------------------------------------------------------------------------------------------------------------------------

def test_hb1_wege(hand):
    assert hand[2].tolist() == [[1, 5, 9], [9, 5, 1]]


def test_hb1_normalisieren_28_drittel_und_jeder_kunde_zwei_drittel(hand):
    a, b, C = hand
    n = U.normalisieren(a, b, C)
    # Nachfrage 4/3 je Kunde; Plan (4/3, 2/3, 0), (0, 2/3, 4/3); Kosten 4/3 · 1 + 2/3 · 5 + 2/3 · 5 + 4/3 · 1 = 28/3
    assert n.bewegungskosten == pytest.approx(28 / 3, abs=1e-12)
    assert n.plan == pytest.approx(np.array([[4 / 3, 2 / 3, 0], [0, 2 / 3, 4 / 3]]), abs=1e-12)
    assert (n.geliefert / b).tolist() == pytest.approx([2 / 3] * 3, abs=1e-12)
    wert, ecken_opt = EB.optimum([F(2), F(2)], [F(4, 3)] * 3, CS)        # dasselbe in Brüchen, ohne Simplex
    assert wert == F(28, 3) and len(ecken_opt) == 1


@pytest.mark.parametrize("rho, zielwert, bewegt", [(F(1, 4), F(5, 2), 0), (F(49, 100), F(49, 10), 0), (F(51, 100), F(502, 100), 4), (F(1), F(6), 4), (F(3), F(10), 4)])
def test_hb1_teilbedienung_zielwerte_in_bruechen_und_mit_dem_simplex(hand, rho, zielwert, bewegt):
    a, b, C = hand
    a2, b2, C2 = EB.hilfsknoten([F(2), F(2)], [F(2)] * 3, CS, rho)
    wert, plaene = EB.optimum(a2, b2, C2)
    assert wert == zielwert
    t = U.teilbedienung(a, b, C, float(rho))
    assert t.zielwert == pytest.approx(float(zielwert), abs=1e-9) and t.bewegt == pytest.approx(bewegt, abs=1e-9)


@pytest.mark.parametrize("rho", [1.0, 3.0])
def test_hb1_der_plan_ist_eindeutig_kunde_2_bleibt_leer_bewegungskosten_4(hand, rho):
    a, b, C = hand
    t = U.teilbedienung(a, b, C, rho)
    assert t.plan == pytest.approx(np.array([[2.0, 0, 0], [0, 0, 2.0]]), abs=1e-12)
    assert t.bewegungskosten == pytest.approx(4.0, abs=1e-12)
    assert U.zaehlung(t.geliefert, b) == (2, 1, 0)
    a2, b2, C2 = EB.hilfsknoten([F(2), F(2)], [F(2)] * 3, CS, F(int(rho)))
    assert len(EB.optimum(a2, b2, C2)[1]) == 1               # eindeutig auch in der Aufzählung


def test_hb1_umschlag_bei_einhalb_gleich_halbe_grenzkosten_der_letzten_einheit(hand):
    a, b, C = hand
    k = T.kurve(a, b, C)
    assert k.steigung.tolist() == pytest.approx([1.0, 1.0]) and k.menge.tolist() == pytest.approx([2.0, 2.0])
    assert k.grenzkosten_letzte == pytest.approx(1.0) and k.rho_voll == pytest.approx(0.5)
    assert T.bewegt_bei(k, 0.49) == 0.0 and T.bewegt_bei(k, 0.51) == pytest.approx(4.0)


def test_hb1_vergleich_vier_siebtel_und_34_drittel_gegen_6(hand):
    a, b, C = hand
    n, t = U.normalisieren(a, b, C, 1.0), U.teilbedienung(a, b, C, 1.0)
    assert 1.0 - t.bewegungskosten / n.bewegungskosten == pytest.approx(4 / 7, abs=1e-12)         # 57,1 % weniger, Basis Normalisieren
    assert n.zielwert == pytest.approx(34 / 3, abs=1e-12) and t.zielwert == pytest.approx(6.0, abs=1e-12)   # Strafe 1 auf die in beiden Plänen gleichen 2 Einheiten


def test_hb1_in_beiden_plaenen_bleiben_dieselben_zwei_einheiten_ungeliefert(hand):
    a, b, C = hand
    n, t = U.normalisieren(a, b, C, 1.0), U.teilbedienung(a, b, C, 1.0)
    assert (b.sum() - n.bewegt) == pytest.approx(2.0) and (b.sum() - t.bewegt) == pytest.approx(2.0)
    assert n.erzeugt == pytest.approx(2.0) and t.erzeugt == pytest.approx(2.0) and n.vernichtet == pytest.approx(0.0, abs=1e-12) and t.vernichtet == pytest.approx(0.0, abs=1e-12)


def test_hb1_die_ansicht_der_app_gibt_dieselben_zahlen():
    an = E.ansicht(E.Settings(quelle="Von Hand: 2 Lager, 3 Kunden", rho1=0.1))      # ρ = 0,1 Straßenlängen der Länge 10 = 1
    assert an.norm.bewegungskosten == pytest.approx(28 / 3) and an.teil.bewegungskosten == pytest.approx(4.0) and an.teil.zielwert == pytest.approx(6.0)
    assert an.ersparnis == pytest.approx(4 / 7) and an.zaehl_teil == (2, 1, 0) and an.zaehl_norm == (0, 0, 3)
    assert an.rho_voll_rel == pytest.approx(0.05)


# ----------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 2: Kappungsformel und Verdrängung
# ----------------------------------------------------------------------------------------------------------------------------------

Q2, Z2 = [F(0), F(5)], [F(4), F(20)]
C2 = [[abs(q - z) for z in Z2] for q in Q2]


def test_hb2_wege_und_ausgeglichene_zuordnungen():
    assert C2 == [[4, 20], [1, 15]]
    assert (C2[0][0] + C2[1][1], C2[0][1] + C2[1][0]) == (19, 21)
    assert F(C2[1][1], 19) == F(15, 19) and min(C2[0][1], C2[1][1]) == 15


@pytest.mark.parametrize("rho", [F(1, 2), F(2), F(8), F(9), F(10), F(30)])
def test_hb2_hilfsknoten_gleich_kappungsformel_gleich_simplex(rho):
    kappe = 2 * rho
    formel = min(min(C2[0][0], kappe) + min(C2[1][1], kappe), min(C2[0][1], kappe) + min(C2[1][0], kappe))
    a2, b2, CH = EB.hilfsknoten([F(1), F(1)], [F(1), F(1)], C2, rho)
    assert EB.optimum(a2, b2, CH)[0] == formel
    t = U.teilbedienung(np.ones(2), np.ones(2), np.array([[4.0, 20.0], [1.0, 15.0]]), float(rho))
    assert t.zielwert == pytest.approx(float(formel), abs=1e-9)


def test_hb2_umschlag_bei_rho_9_nicht_bei_d_min_halbe_7_komma_5():
    for rho, zielwert, bedient in ((8, 17, False), (10, 19, True)):
        a2, b2, CH = EB.hilfsknoten([F(1), F(1)], [F(1), F(1)], C2, F(rho))
        wert, plaene = EB.optimum(a2, b2, CH)
        assert wert == zielwert and [P[1][1] for P in plaene] == [1 if bedient else 0]
    a2, b2, CH = EB.hilfsknoten([F(1), F(1)], [F(1), F(1)], C2, F(9))
    wert, plaene = EB.optimum(a2, b2, CH)
    assert wert == 19 and sorted(P[1][1] for P in plaene) == [0, 1]                    # Gleichstand: zwei optimale Pläne
    assert C2[0][0] - C2[1][0] == 3 and 2 * 9 == 15 + 3                                 # Verdrängung: Zusatzweg 3, Umschlag 2ρ = 15 + 3
    d = E.ausreisser_groessen([0.0, 5.0], [4.0, 20.0], 1)
    assert d["rho_erste"] == pytest.approx(9.0) and d["d_min2"] == pytest.approx(7.5) and d["anteil_a"] == pytest.approx(15 / 19)


# ----------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 3: KL-Strafe, geschlossene Form
# ----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("t", [0.1, 1.0, 10.0, 100.0])
def test_hb3_skalierungsverfahren_gleich_geschlossene_form_auf_1e_9(t):
    a, b, C0 = np.array([0.5, 0.5]), np.array([0.65, 0.65]), np.zeros((2, 2))
    res = KL.kl_strafe(a, b, C0, 1.0, t, tol=1e-14)
    assert res.konvergiert and res.masse == pytest.approx(4.0 * (0.5 * 0.65 / 4.0) ** (t / (2 * t + 1.0)), abs=1e-9)
    assert KL.masse_zwei_mal_zwei(t) == pytest.approx(res.masse, abs=1e-9)


def test_hb3_werte_der_formel_und_grenzfaelle():
    assert [round(KL.masse_zwei_mal_zwei(t), 3) for t in (0.1, 1.0, 4.0, 5.0, 10.0, 100.0)] == [3.245, 1.732, 1.311, 1.278, 1.210, 1.147]
    assert KL.masse_zwei_mal_zwei(1e-9) > 3.99                                  # τ → 0: Summe der Kernwerte 4 (K nicht normiert)
    assert KL.masse_zwei_mal_zwei(1e9) == pytest.approx(1.3 ** 0.5, abs=1e-6)  # τ → ∞: √(1 · 1,3) = 1,1402, weder 1 noch 1,3
    assert KL.masse_zwei_mal_zwei(4.0) > 1.3 > KL.masse_zwei_mal_zwei(5.0)      # über beiden Randsummen bis τ/ε ≈ 4,3
    assert 1.0 < KL.masse_zwei_mal_zwei(100.0) < 1.3                           # auch bei 100 noch über dem Bestand 1


def test_hb3_das_modell_mit_schlupf_bewegt_bei_kosten_null_den_ganzen_bestand_und_lasst_0_3_offen():
    C0 = [[F(0)] * 2 for _ in range(2)]
    for rho in (F(1, 10), F(1), F(10)):
        a2, b2, CH = EB.hilfsknoten([F(1, 2), F(1, 2)], [F(13, 20), F(13, 20)], C0, rho)
        assert EB.optimum(a2, b2, CH)[0] == rho * F(3, 10)
        t = U.teilbedienung(np.array([0.5, 0.5]), np.array([0.65, 0.65]), np.zeros((2, 2)), float(rho))
        assert t.zielwert == pytest.approx(0.3 * float(rho), abs=1e-12) and t.bewegt == pytest.approx(1.0)


def test_hb3_ein_durchlauf_sind_zwei_produkte_und_der_abbruch_aendert_die_masse_nicht_ueber_1e_8():
    a, b = np.array([0.5, 0.5]), np.array([0.65, 0.65])
    res = KL.kl_strafe(a, b, np.zeros((2, 2)), 1.0, 1.0, tol=1e-9)
    assert res.produkte == 2 * res.durchlaeufe and 1 < res.durchlaeufe < 1000
    assert res.masse == pytest.approx(4.0 * (0.5 * 0.65 / 4.0) ** (1.0 / 3.0), abs=1e-8)


# ----------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel 4: der Ausreißeranteil hängt am gewählten Plan
# ----------------------------------------------------------------------------------------------------------------------------------

def test_hb4_zwei_optimale_zuordnungen_mit_anteil_ein_halb_drei_viertel_und_fuenf_achtel_der_mischung():
    ys = [F(2), F(3)]
    C4 = [[abs(x - y) for y in ys] for x in (F(0), F(1))]
    assert C4 == [[2, 3], [1, 2]]
    assert len(EB.ecken([F(1), F(1)], [F(1), F(1)])) == 2
    wert, plaene = EB.optimum([F(1), F(1)], [F(1), F(1)], C4)
    assert wert == 4 and len(plaene) == 2
    sortiert = ((F(1), F(0)), (F(0), F(1)))
    gekreuzt = ((F(0), F(1)), (F(1), F(0)))
    assert set(plaene) == {sortiert, gekreuzt}
    anteil = lambda P: sum(C4[i][1] * P[i][1] for i in range(2)) / 4         # Wege zum Ausreißer (Ziel 3) durch die Gesamtkosten
    assert (anteil(sortiert), anteil(gekreuzt)) == (F(1, 2), F(3, 4))
    mitte = tuple(tuple((sortiert[i][j] + gekreuzt[i][j]) / 2 for j in range(2)) for i in range(2))
    assert EB.kosten(mitte, C4) == 4 and anteil(mitte) == F(5, 8)


def test_hb4_die_drei_fassungen_der_demo_geben_ein_halb_dreiviertel_oder_ein_halb_und_ein_viertel():
    d = E.ausreisser_groessen([0.0, 1.0], [2.0, 3.0], 1)
    assert d["anteil_a"] == pytest.approx(0.5)                                   # sortierte Zuordnung
    assert d["anteil_c"] == pytest.approx(0.25)                                  # vom Plan unabhängig: Integral oberhalb der Schnittstelle 2
    assert d["anteil_b"] in (pytest.approx(0.5), pytest.approx(0.75))            # eine der beiden optimalen Zuordnungen, welche, ist Sache des Verfahrens
    assert len({round(d["anteil_a"], 9), round(d["anteil_c"], 9)}) == 2


def test_hb4_integral_von_f_minus_g_ist_zwei_und_oberhalb_der_schnittstelle_ein_viertel():
    xs, ys = [F(0), F(1)], [F(2), F(3)]
    z = sorted(set(xs) | set(ys))
    gesamt = oben = F(0)
    for lo, hi in zip(z[:-1], z[1:]):
        stueck = abs(F(sum(1 for x in xs if x <= lo), 2) - F(sum(1 for y in ys if y <= lo), 2)) * (hi - lo)
        gesamt += stueck
        oben += stueck if lo >= 2 else 0
    assert gesamt == 2 and oben / gesamt == F(1, 4)

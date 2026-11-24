"""Randfälle: leere Seiten, ein Lager und ein Kunde, ρ = 0, ρ sehr groß, kleine Massen, gleich weite Kunden, gleiche Summen, ungültige Eingaben."""

import numpy as np
import pytest

import ugm_teiltransport as T
import ugm_ungleich as U


def test_bestand_null_oder_nachfrage_null_bedeutet_nichts_zu_tun_und_nur_strafe():
    C = np.array([[1.0, 2.0], [3.0, 1.0]])
    t = U.teilbedienung(np.zeros(2), np.array([2.0, 3.0]), C, 0.4)
    assert t.bewegt == 0.0 and t.erzeugt == pytest.approx(5.0) and t.vernichtet == 0.0 and t.zielwert == pytest.approx(2.0)
    t = U.teilbedienung(np.array([2.0, 3.0]), np.zeros(2), C, 0.4)
    assert t.bewegt == 0.0 and t.vernichtet == pytest.approx(5.0) and t.zielwert == pytest.approx(2.0)
    t = U.teilbedienung(np.zeros(2), np.zeros(2), C, 0.4)
    assert t.bewegt == 0.0 and t.zielwert == 0.0
    n = U.normalisieren(np.zeros(2), np.array([2.0, 3.0]), C)
    assert n.bewegt == 0.0 and n.bewegungskosten == 0.0
    k = T.kurve(np.zeros(2), np.array([2.0, 3.0]), C)
    assert k.schritte == 0 and k.rho_voll == 0.0 and T.bewegt_bei(k, 5.0) == 0.0


def test_ein_lager_ein_kunde_die_strafe_entscheidet_bei_rho_gleich_weg_halbe():
    C = np.array([[2.0]])
    for rho, bewegt in ((0.5, 0.0), (0.99, 0.0), (1.01, 3.0), (10.0, 3.0)):
        t = U.teilbedienung(np.array([3.0]), np.array([5.0]), C, rho)
        assert t.bewegt == pytest.approx(bewegt, abs=1e-9), rho
        assert t.zielwert == pytest.approx(min(rho * 8.0, 2.0 * 3.0 + rho * 2.0), abs=1e-9)
    k = T.kurve(np.array([3.0]), np.array([5.0]), C)
    assert k.rho_voll == pytest.approx(1.0) and k.bewegt_gesamt == pytest.approx(3.0)


def test_rho_null_bewegt_nichts_bei_positiven_wegen_und_rho_sehr_gross_bewegt_den_ganzen_bestand():
    C = np.array([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]])
    a, b = np.array([2.0, 2.0]), np.array([2.0, 2.0, 3.0])
    t0 = U.teilbedienung(a, b, C, 0.0)
    assert t0.zielwert == pytest.approx(0.0, abs=1e-12) and t0.bewegt == pytest.approx(0.0, abs=1e-12)
    tg = U.teilbedienung(a, b, C, 1e6)
    assert tg.bewegt == pytest.approx(4.0) and tg.zielwert == pytest.approx(tg.bewegungskosten + 1e6 * 3.0, rel=1e-12)


def test_massen_von_0_01_bis_100_geben_dieselben_verhaeltnisse():
    C = np.array([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]])
    a, b = np.array([2.0, 2.0]), np.array([2.0, 2.0, 3.0])
    ref = U.teilbedienung(a, b, C, 0.7)
    for f in (1e-2, 1.0, 100.0):
        t = U.teilbedienung(a * f, b * f, C, 0.7)
        assert t.zielwert == pytest.approx(ref.zielwert * f, rel=1e-6) and t.bewegt == pytest.approx(ref.bewegt * f, rel=1e-6)


def test_alle_kunden_gleich_weit_gleichstand_zielwert_ist_eindeutig_die_wahl_der_kunden_nicht():
    C = np.full((1, 4), 0.5)
    a, b = np.array([2.0]), np.array([1.0, 1.0, 1.0, 1.0])
    t = U.teilbedienung(a, b, C, 1.0)           # 2ρ = 2 > 0,5: bewegt 2 Einheiten
    assert t.bewegt == pytest.approx(2.0) and t.zielwert == pytest.approx(1.0 + 1.0 * (0.0 + 2.0))
    assert t.geliefert.sum() == pytest.approx(2.0) and (t.geliefert <= b + 1e-12).all()      # welche Kunden, ist offen; die Summe ist bestimmt


def test_gleiche_gesamtmassen_geben_normalisieren_gleich_teilbedienung(hand):
    a, b, C = hand
    b4 = np.array([4 / 3] * 3)
    n = U.normalisieren(a, b4, C)
    rho_voll = T.kurve(a, b4, C).rho_voll
    t = U.teilbedienung(a, b4, C, 1.01 * rho_voll)             # ab ρ_voll; darunter bewegt die Teilbedienung auch bei gleicher Summe weniger
    assert rho_voll > 1.0 and n.bewegungskosten == pytest.approx(t.bewegungskosten) and n.zielwert == pytest.approx(U.normalisieren(a, b4, C, 1.01 * rho_voll).zielwert)
    assert U.teilbedienung(a, b4, C, 1.0).bewegt < 4.0           # ρ = 1 liegt unter ρ_voll: nicht der ganze Bestand


def test_zaehlung_voll_leer_teilweise_mit_den_grenzen():
    assert U.zaehlung([2.0, 0.0, 1.0, 1e-8, 2.0 - 1e-8], [2.0, 2.0, 2.0, 2.0, 2.0]) == (2, 2, 1)
    assert U.zaehlung([0.0, 0.0], [0.0, 1.0]) == (0, 2, 0)          # Nachfrage 0 und nichts geliefert: leer


def test_ungueltige_eingaben_werden_abgelehnt():
    C = np.array([[1.0]])
    with pytest.raises(ValueError):
        U.teilbedienung(np.array([1.0]), np.array([1.0]), C, -0.1)
    with pytest.raises(ValueError):
        T.kurve(np.array([-1.0]), np.array([1.0]), C)
    with pytest.raises(ValueError):
        T.kurve(np.array([1.0, 2.0]), np.array([1.0]), C)
    with pytest.raises(ValueError):
        T.kosten_bei_menge(T.kurve(np.array([1.0]), np.array([1.0]), C), 2.0)


def test_kappung_ist_min_von_weg_und_zwei_rho():
    assert U.kappung(np.array([[1.0, 5.0]]), 1.0).tolist() == [[1.0, 2.0]]

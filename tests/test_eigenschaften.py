"""Eigenschaftstests (je 15 bis 50 Instanzen, feste Zufalls-Seeds): Sätze, die für jede Straße gelten müssen. Orakel-Vergleiche stehen in test_oracle_ungleiche_mengen.py."""

import numpy as np
import pytest

import ugm_evaluation as E
import ugm_scenario as SC
import ugm_teiltransport as T
import ugm_ungleich as U


def strassen(anzahl, ueberhang=1.3, n_kunden=14, zweck=11, seed=5):
    for i in range(anzahl):
        yield SC.make_strasse(n_kunden + i % 6, ueberhang, seed, zweck, i)


def test_bei_gleicher_summe_ist_die_teilbedienung_gleich_dem_normalisieren_auf_20_strassen():
    for s in strassen(20, ueberhang=1.0):
        n, t = U.normalisieren(s.bestand, s.nachfrage, s.kosten), U.teilbedienung(s.bestand, s.nachfrage, s.kosten, 1.0)
        assert t.bewegungskosten == pytest.approx(n.bewegungskosten, rel=1e-9, abs=1e-9)
        assert n.erzeugt == pytest.approx(0.0, abs=1e-9) and t.vernichtet == pytest.approx(0.0, abs=1e-9)


def test_ab_rho_voll_sind_die_bewegungskosten_der_teilbedienung_nie_hoeher_als_beim_normalisieren_auf_50_strassen():
    """Satz: der normalisierte Plan bewegt den ganzen Bestand; die Teilbedienung wählt unter allen Plänen, die den ganzen Bestand bewegen, den billigsten."""
    for i in range(50):
        s = SC.make_strasse(12 + i % 9, 1.0 + 0.1 * (i % 11), 5, 12, i)
        k = T.kurve(s.bestand, s.nachfrage, s.kosten)
        n = U.normalisieren(s.bestand, s.nachfrage, s.kosten)
        for rho in (k.rho_voll * 1.001, 1.0):
            t = U.teilbedienung(s.bestand, s.nachfrage, s.kosten, rho)
            assert t.bewegt == pytest.approx(s.bestand.sum(), abs=1e-7), (i, rho)
            assert t.bewegungskosten <= n.bewegungskosten + 1e-9, (i, rho)


def test_abschaetzung_der_bewegungskosten_gilt_fuer_jedes_rho_auch_unter_rho_voll():
    """Satz: der normalisierte Plan ist für das Modell mit Schlupf zulässig (R ≥ 1), also Bewegungskosten(Teil) ≤ Bewegungskosten(Norm) − 2ρ (Bestand − bewegte Menge)."""
    unter = 0
    for i, s in enumerate(list(strassen(20, ueberhang=1.0)) + list(strassen(20, ueberhang=1.3)) + list(strassen(10, ueberhang=2.0))):
        n = U.normalisieren(s.bestand, s.nachfrage, s.kosten)
        for rho in (0.0, 0.01, 0.025, 0.06, 0.1, 0.3, 1.0):
            t = U.teilbedienung(s.bestand, s.nachfrage, s.kosten, rho)
            assert t.bewegungskosten <= n.bewegungskosten - 2 * rho * (s.bestand.sum() - t.bewegt) + 1e-9, (i, rho)
            unter += t.bewegt < s.bestand.sum() - 1e-7
    assert unter > 50                                                  # die Probe liegt wirklich oft unter ρ_voll


def test_der_zielwert_der_teilbedienung_ist_bei_jedem_rho_hoechstens_der_des_normalisierens():
    for i, s in enumerate(strassen(30)):
        for rho in (0.0, 0.02, 0.1, 0.3, 1.0):
            n = U.normalisieren(s.bestand, s.nachfrage, s.kosten, rho)
            t = U.teilbedienung(s.bestand, s.nachfrage, s.kosten, rho)
            assert t.zielwert <= n.zielwert + 1e-9, (i, rho)


def test_mehr_rho_bewegt_nie_weniger_masse_und_der_zielwert_ist_konkav_in_rho():
    raster = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4]          # gleichmäßig genug für die zweiten Differenzen unten
    gleich = [0.0 + 0.02 * j for j in range(16)]
    for i, s in enumerate(strassen(20)):
        bew = [U.teilbedienung(s.bestand, s.nachfrage, s.kosten, r).bewegt for r in raster]
        assert all(b2 >= b1 - 1e-7 for b1, b2 in zip(bew, bew[1:])), (i, bew)
        f = np.array([U.teilbedienung(s.bestand, s.nachfrage, s.kosten, r).zielwert for r in gleich])
        assert (f[2:] - 2 * f[1:-1] + f[:-2] <= 1e-9).all(), i                                 # konkav: Minimum von Geraden in ρ


def test_zielwert_gleich_kurve_plus_strafe_der_kurve_ist_ein_zweiter_unabhaengiger_rechenweg_im_haus():
    for i, s in enumerate(strassen(25)):
        k = T.kurve(s.bestand, s.nachfrage, s.kosten)
        for rho in (0.0, 0.03, 0.07, 0.15, 1.0):
            m = T.bewegt_bei(k, rho)
            erwartet = T.kosten_bei_menge(k, m) + rho * (s.bestand.sum() + s.nachfrage.sum() - 2 * m)
            t = U.teilbedienung(s.bestand, s.nachfrage, s.kosten, rho)
            assert t.zielwert == pytest.approx(erwartet, rel=1e-9, abs=1e-9), (i, rho)
            assert T.plan_bei(k, rho).sum() == pytest.approx(m, abs=1e-9)
            assert (T.plan_bei(k, rho).sum(axis=1) <= s.bestand + 1e-9).all() and (T.geliefert_bei(k, rho) <= s.nachfrage + 1e-9).all()


def test_steigungen_der_kurve_sind_nicht_fallend_und_die_mengen_summieren_sich_zum_kleineren_gesamt():
    for s in strassen(30):
        k = T.kurve(s.bestand, s.nachfrage, s.kosten)
        assert (np.diff(k.steigung) >= -1e-9).all() and (k.menge > 0).all()
        assert k.bewegt_gesamt == pytest.approx(min(s.bestand.sum(), s.nachfrage.sum()), abs=1e-9)
        assert k.kosten_gesamt == pytest.approx(U.teilbedienung(s.bestand, s.nachfrage, s.kosten, 1e6).bewegungskosten, rel=1e-9, abs=1e-9)


def test_kosten_und_rho_mit_demselben_faktor_aendern_den_zielwert_proportional_und_die_lieferungen_nicht():
    for i, s in enumerate(strassen(15)):
        for rho in (0.05, 0.2):
            t = U.teilbedienung(s.bestand, s.nachfrage, s.kosten, rho)
            t7 = U.teilbedienung(s.bestand, s.nachfrage, 7.0 * s.kosten, 7.0 * rho)
            assert t7.zielwert == pytest.approx(7.0 * t.zielwert, rel=1e-9, abs=1e-9) and t7.bewegt == pytest.approx(t.bewegt, abs=1e-7), (i, rho)


def test_umnummerieren_der_lager_und_kunden_aendert_den_zielwert_nicht():
    for i, s in enumerate(strassen(15)):
        g = SC.instanz_rng(9, 9, i)
        pl, pk = list(range(len(s.bestand))), list(range(len(s.nachfrage)))
        g.shuffle(pl)
        g.shuffle(pk)
        C2 = s.kosten[np.ix_(pl, pk)]
        for rho in (0.05, 1.0):
            a = U.teilbedienung(s.bestand, s.nachfrage, s.kosten, rho)
            b = U.teilbedienung(s.bestand[pl], s.nachfrage[pk], C2, rho)
            assert b.zielwert == pytest.approx(a.zielwert, rel=1e-9, abs=1e-9) and b.bewegt == pytest.approx(a.bewegt, abs=1e-7)


def test_ein_zusaetzlicher_kunde_ohne_nachfrage_aendert_nichts():
    for i, s in enumerate(strassen(15)):
        Cn = np.hstack([s.kosten, np.full((len(s.bestand), 1), 0.3)])
        nn = np.concatenate([s.nachfrage, [0.0]])
        for rho in (0.05, 1.0):
            a = U.teilbedienung(s.bestand, s.nachfrage, s.kosten, rho)
            b = U.teilbedienung(s.bestand, nn, Cn, rho)
            assert b.zielwert == pytest.approx(a.zielwert, rel=1e-9, abs=1e-9) and b.geliefert[-1] == pytest.approx(0.0, abs=1e-12)


def test_ab_rho_voll_wird_immer_das_kleinere_von_bestand_und_nachfrage_bewegt_auch_bei_bestand_ueber_nachfrage():
    for i in range(30):
        s = SC.make_strasse(12 + i % 5, 0.6 + 0.1 * (i % 15), 5, 13, i)
        k = T.kurve(s.bestand, s.nachfrage, s.kosten)
        t = U.teilbedienung(s.bestand, s.nachfrage, s.kosten, k.rho_voll * 1.001 + 1e-9)
        assert t.bewegt == pytest.approx(min(s.bestand.sum(), s.nachfrage.sum()), abs=1e-7), i


def test_normalisieren_gibt_jedem_kunden_denselben_anteil_eins_durch_r_der_nachfrage():
    for s in strassen(15):
        n = U.normalisieren(s.bestand, s.nachfrage, s.kosten)
        assert n.geliefert / s.nachfrage == pytest.approx(np.full(len(s.nachfrage), 1.0 / s.ueberhang), abs=1e-9)
        assert U.zaehlung(n.geliefert, s.nachfrage)[1] == 0


def test_der_umschlag_des_ersten_ausreissers_liegt_nie_unter_d_min_halbe_in_40_instanzen():
    for i in range(40):
        d = E.ausreisser_instanz(2 + i % 3, 6.0 + i % 5, 5, 14, i, n=30)
        assert d["rho_erste"] >= d["d_min2"] - 1e-12 and d["rho_alle"] >= d["d_max2"] - 1e-12 and d["rho_erste"] <= d["rho_alle"]


def test_ausreisser_bedient_zaehlt_ab_den_schwellen():
    d = E.ausreisser_instanz(3, 8.0, 5, 14, 0, n=30)
    assert E.ausreisser_bedient(d, 0.0) == 0 and E.ausreisser_bedient(d, d["rho_erste"] - 1e-9) == 0
    assert E.ausreisser_bedient(d, d["rho_alle"] + 1e-9) == 3 and E.ausreisser_bedient(d, 1e6) == 3
    assert [E.ausreisser_bedient(d, s + 1e-9) for s in d["schwellen"]] == [1, 2, 3]


@pytest.mark.parametrize("ueberhang", [1.0, 1.15, 1.3, 1.6, 2.0])
def test_ab_rho_gleich_einhalb_wird_in_den_reihen_der_ganze_bestand_bewegt_bei_10_30_und_60_kunden(ueberhang):
    """Messung, kein Satz: ρ_voll liegt in allen Reihen (je 40 Instanzen) unter 0,5 (größter Wert der Messung 0,375 bei 100 Instanzen je Zelle); die Hilfe des Reglers nennt ρ = 0,5."""
    for n in (10, 30, 60):
        for i in range(40):
            s = SC.make_strasse(n, ueberhang, 2026, 2, i)
            assert T.kurve(s.bestand, s.nachfrage, s.kosten).rho_voll < 0.5, (ueberhang, n, i)

"""Orakel für das Modell mit Strafe, die Teiltransport-Kurve und die KL-Strafe: jeweils ein ANDERER Rechenweg als der der Demo.

- Modell mit Strafe: `scipy.optimize.linprog` (HiGHS) mit Schlupfvariablen; die Aufzählung aller Ecken in Brüchen über aufspannende Bäume (`ecken_brueche.py`) für kleine Aufgaben;
  bei gleichen Gesamtmassen die Kappungsformel (gewöhnliche Zuordnung mit den Kosten min(c, 2ρ), `scipy.optimize.linear_sum_assignment`).
- Teiltransport-Kurve: das Linearprogramm mit der Nebenbedingung „bewegte Menge = m“ (`linprog`); der Umschlag ρ_voll gegen eine Bisektion über ρ mit dem Modell mit Strafe
  und gegen den Differenzenquotienten der Kurve (Toleranz 1e-2, er hängt an der Auflösung).
- Ausreißer: ρ_erste und ρ_alle gegen Bisektion mit der Kappungsformel.
- KL-Strafe: die Stationaritätsbedingung der Zielfunktion (C + ε ln P + τ ln(Zeilensumme / a) + τ ln(Spaltensumme / b) = 0 in jeder Zelle), unabhängig von der Iteration.
- Das Normalisieren: `scipy.stats.wasserstein_distance` normiert die Gewichte stillschweigend und gibt die Bewegungskosten des Normalisierens durch die Gesamtmasse (Versionsvermerk im Test).
Welche Ecke bei Gleichstand herauskommt, wird nie bitgleich verglichen: geprüft werden Zielwerte und Kundenmengen. Fehlt scipy im Test-venv, überspringt sich der Test (`pytest.importorskip`); alle Läufe zusammen unter 30 Sekunden."""

from fractions import Fraction as F

import numpy as np
import pytest

import ecken_brueche as EB
import ugm_kl as KL
import ugm_scenario as SC
import ugm_teiltransport as T
import ugm_ungleich as U
from orakel_zahlen import ORAKEL

scipy_opt = pytest.importorskip("scipy.optimize")


def lp_schlupf(a, b, Cm, rho, skala=1.0):
    """min <C,P> + ρ (vernichtet + erzeugt) mit Schlupfvariablen, unabhängig vom Simplex der Demo."""
    n, m = Cm.shape
    c = np.concatenate([Cm.ravel(), np.full(n + m, rho)])
    Ar = np.hstack([np.kron(np.eye(n), np.ones((1, m))), np.eye(n), np.zeros((n, m))])
    Ac = np.hstack([np.kron(np.ones((1, n)), np.eye(m)), np.zeros((m, n)), np.eye(m)])
    r = scipy_opt.linprog(c, A_eq=np.vstack([Ar, Ac]), b_eq=np.concatenate([a, b]) * skala, bounds=(0, None), method="highs")
    assert r.status == 0, r.message
    return r.fun / skala, r.x[: n * m].reshape(n, m) / skala


def lp_menge(a, b, Cm, menge):
    """Kleinste Bewegungskosten für genau `menge` bewegte Einheiten (P1 ≤ a, Pᵀ1 ≤ b, ΣP = menge)."""
    n, m = Cm.shape
    Aub = np.vstack([np.kron(np.eye(n), np.ones((1, m))), np.kron(np.ones((1, n)), np.eye(m))])
    r = scipy_opt.linprog(Cm.ravel(), A_ub=Aub, b_ub=np.concatenate([a, b]), A_eq=np.ones((1, n * m)), b_eq=[menge], bounds=(0, None), method="highs")
    assert r.status == 0, r.message
    return float(r.fun)


def strassen(anzahl, n_kunden=15, ueberhang=1.3, zweck=9):
    for i in range(anzahl):
        yield SC.make_strasse(n_kunden + i % 7, ueberhang + 0.1 * (i % 5), 3, zweck, i)


def test_modell_mit_strafe_gleich_linprog_mit_schlupfvariablen_auf_40_strassen():
    for i, s in enumerate(strassen(ORAKEL["schlupf"])):
        for rho in (0.0, 0.03, 0.1, 0.4, 2.0):
            t = U.teilbedienung(s.bestand, s.nachfrage, s.kosten, rho)
            ref, _ = lp_schlupf(s.bestand, s.nachfrage, s.kosten, rho)
            assert t.zielwert == pytest.approx(ref, rel=1e-9, abs=1e-9), (i, rho)
            assert (t.plan.sum(axis=1) <= s.bestand + 1e-9).all() and (t.geliefert <= s.nachfrage + 1e-9).all() and (t.plan >= 0).all()


def test_aufzaehlung_aller_ecken_in_bruechen_gleich_simplex_auf_30_kleinen_aufgaben():
    for k in range(ORAKEL["aufzaehlung"]):
        g = SC.instanz_rng(7, 8, k)
        n, m = 2 + g.below(2), 3
        a = [g.below(4) + 1 for _ in range(n)]
        b = [g.below(4) + 1 for _ in range(m)]
        Cm = [[g.below(6) for _ in range(m)] for _ in range(n)]
        rho = F(g.below(9) + 1, 2)
        ah, bh, CH = EB.hilfsknoten([F(x) for x in a], [F(x) for x in b], [[F(x) for x in r] for r in Cm], rho)
        wert, _ = EB.optimum(ah, bh, CH)
        t = U.teilbedienung(np.array(a, float), np.array(b, float), np.array(Cm, float), float(rho))
        assert t.zielwert == pytest.approx(float(wert), abs=1e-9), (k, a, b, Cm, rho)


def test_kappungsformel_bei_gleichen_gesamtmassen_gleich_linear_sum_assignment_auf_20_instanzen():
    lsa = scipy_opt.linear_sum_assignment
    for i in range(ORAKEL["kappung"]):
        xa, xb = SC.make_ausreisser(2, 6.0, 4, 9, i, n=18)
        Cm = np.abs(xa[:, None] - xb[None, :])
        for rho in (0.2, 0.7, 1.5, 3.0, 8.0):
            r, c = lsa(np.minimum(Cm, 2 * rho))
            kappe = float(np.minimum(Cm, 2 * rho)[r, c].sum())
            t = U.teilbedienung(np.ones(18), np.ones(18), Cm, rho)
            assert t.zielwert == pytest.approx(kappe, rel=1e-9, abs=1e-9), (i, rho)
            assert t.zielwert == pytest.approx(lp_schlupf(np.ones(18), np.ones(18), Cm, rho)[0], rel=1e-9, abs=1e-9)


def test_teiltransport_kurve_gleich_linprog_mit_bewegter_menge_gleich_m_auf_25_strassen():
    for i, s in enumerate(strassen(ORAKEL["kurve"], n_kunden=10)):
        k = T.kurve(s.bestand, s.nachfrage, s.kosten)
        gesamt = k.bewegt_gesamt
        assert gesamt == pytest.approx(min(s.bestand.sum(), s.nachfrage.sum()), abs=1e-9)
        for anteil in (0.0, 0.2, 0.5, 0.8, 0.95, 1.0):
            m = anteil * gesamt
            assert T.kosten_bei_menge(k, m) == pytest.approx(lp_menge(s.bestand, s.nachfrage, s.kosten, m), rel=1e-9, abs=1e-9), (i, anteil)


def test_rho_voll_gleich_bisektion_und_gleich_halbe_grenzkosten_als_differenzenquotient_auf_25_strassen():
    for i, s in enumerate(strassen(ORAKEL["rho_voll"], n_kunden=12)):
        k = T.kurve(s.bestand, s.nachfrage, s.kosten)
        lo, hi = 0.0, 1.0
        for _ in range(34):
            mitte = 0.5 * (lo + hi)
            if U.teilbedienung(s.bestand, s.nachfrage, s.kosten, mitte).bewegt >= s.bestand.sum() - 1e-6:
                hi = mitte
            else:
                lo = mitte
        assert hi == pytest.approx(k.rho_voll, abs=2e-8), i
        gesamt = k.bewegt_gesamt
        dq = (lp_menge(s.bestand, s.nachfrage, s.kosten, gesamt) - lp_menge(s.bestand, s.nachfrage, s.kosten, gesamt - 0.1)) / 0.1
        assert abs(dq / 2 - k.rho_voll) < 1e-2, (i, dq / 2, k.rho_voll)            # Toleranz 1e-2: der Differenzenquotient hängt an der Auflösung 0,1


def test_ausreisser_umschlaege_gleich_bisektion_mit_der_kappungsformel_auf_12_instanzen():
    lsa = scipy_opt.linear_sum_assignment
    from ugm_evaluation import ausreisser_groessen
    n, k_aus = 30, 2

    def bedient(Cm, rho):
        r, c = lsa(np.minimum(Cm, 2 * rho))
        return int(sum(1 for i, j in zip(r, c) if j >= n - k_aus and Cm[i, j] < 2 * rho - 1e-9))

    for i in range(ORAKEL["ausreisser"]):
        xa, xb = SC.make_ausreisser(k_aus, 8.0, 6, 9, i, n=n)
        Cm = np.abs(xa[:, None] - xb[None, :])
        d = ausreisser_groessen(xa, xb, k_aus)
        for ziel, erwartet in ((1, d["rho_erste"]), (k_aus, d["rho_alle"])):
            lo, hi = 0.0, 30.0
            for _ in range(36):
                mitte = 0.5 * (lo + hi)
                if bedient(Cm, mitte) >= ziel:
                    hi = mitte
                else:
                    lo = mitte
            assert hi == pytest.approx(erwartet, abs=1e-6), (i, ziel)
        assert d["rho_erste"] >= d["d_min2"] - 1e-12 and d["rho_alle"] >= d["d_max2"] - 1e-12          # Untergrenze der Kappungsformel


def test_wer_leer_bleibt_ist_bestimmt_gelieferte_menge_je_kunde_ueber_alle_optimalen_plaene():
    for i, s in enumerate(strassen(ORAKEL["eindeutig"], n_kunden=12, ueberhang=1.3)):
        n, m = s.kosten.shape
        rho = 1.0
        t = U.teilbedienung(s.bestand, s.nachfrage, s.kosten, rho)
        c = np.concatenate([s.kosten.ravel(), np.full(n + m, rho)])
        Ar = np.hstack([np.kron(np.eye(n), np.ones((1, m))), np.eye(n), np.zeros((n, m))])
        Ac = np.hstack([np.kron(np.ones((1, n)), np.eye(m)), np.zeros((m, n)), np.eye(m)])
        A = np.vstack([Ar, Ac])
        b_eq = np.concatenate([s.bestand, s.nachfrage])
        for j in range(m):
            wert = np.zeros(n * m + n + m)
            wert[[i2 * m + j for i2 in range(n)]] = 1.0
            ergebnisse = []
            for vz in (1.0, -1.0):
                r = scipy_opt.linprog(vz * wert, A_ub=c[None, :], b_ub=[t.zielwert + 1e-9], A_eq=A, b_eq=b_eq, bounds=(0, None), method="highs")
                assert r.status == 0, r.message
                ergebnisse.append(vz * r.fun)
            assert ergebnisse[1] - ergebnisse[0] < 1e-5, (i, j, ergebnisse)          # gemessen: größte Spannweite 1,9e-6 (HiGHS-Toleranz)


def test_kl_strafe_erfuellt_die_stationaritaetsbedingung_in_jeder_zelle():
    for i in range(ORAKEL["kl"]):
        a, b, Cm = SC.make_weich(5, 9, i, n=7, ueberhang=1.3)
        for tau_eps in (0.5, 5.0):
            eps = 0.05
            res = KL.kl_strafe(a, b, Cm, eps, tau_eps * eps, tol=1e-13)
            assert res.konvergiert
            P = res.plan
            rest = Cm + eps * np.log(P) + tau_eps * eps * np.log(P.sum(axis=1) / a)[:, None] + tau_eps * eps * np.log(P.sum(axis=0) / b)[None, :]
            assert np.abs(rest).max() < 1e-6, (i, tau_eps, np.abs(rest).max())


def test_wasserstein_distance_von_scipy_normiert_die_gewichte_stillschweigend():
    stats = pytest.importorskip("scipy.stats")
    import scipy
    x_lager, x_kunde = np.array([0.0, 10.0]), np.array([1.0, 5.0, 9.0])
    w1 = stats.wasserstein_distance(x_lager, x_kunde, [2, 2], [2, 2, 2])
    w2 = stats.wasserstein_distance(x_lager, x_kunde, [20, 20], [1, 1, 1])
    assert w1 == pytest.approx((28 / 3) / 4, abs=1e-12) and w2 == pytest.approx(w1, abs=1e-12), f"scipy {scipy.__version__}"
    for i, s in enumerate(strassen(ORAKEL["wasserstein"], n_kunden=12)):
        n = U.normalisieren(s.bestand, s.nachfrage, s.kosten)
        w = stats.wasserstein_distance(s.x_lager, s.x_kunden, s.bestand, s.nachfrage)
        assert n.bewegungskosten / s.bestand.sum() == pytest.approx(w, rel=1e-9, abs=1e-12), i

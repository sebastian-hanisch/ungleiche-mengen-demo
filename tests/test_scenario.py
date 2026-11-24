"""Szenario: gleiche Eingabe gleiche Instanz (auch in einem zweiten Prozess mit anderem Hash-Seed), Aufbau der Straße, der Ausreißer und der 30 × 30-Instanzen, Kopien der Quellen."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import ugm_constants as C
import ugm_scenario as SC

ROOT = Path(__file__).resolve().parent.parent


def test_gleiche_eingabe_gleiche_strasse_andere_eingabe_andere_strasse():
    a = SC.make_strasse(30, 1.3, 35, 1, 0)
    b = SC.make_strasse(30, 1.3, 35, 1, 0)
    assert np.array_equal(a.x_kunden, b.x_kunden) and np.array_equal(a.nachfrage, b.nachfrage)
    for andere in (SC.make_strasse(30, 1.3, 36, 1, 0), SC.make_strasse(30, 1.3, 35, 2, 0), SC.make_strasse(30, 1.3, 35, 1, 1)):
        assert not np.array_equal(a.x_kunden, andere.x_kunden)


def test_strasse_hat_drei_lager_bestand_100_sortierte_kunden_und_nachfrage_r_mal_100():
    for R in (1.0, 1.3, 2.0):
        s = SC.make_strasse(30, R, 5, 1, 3)
        assert s.x_lager.tolist() == [0.25, 0.5, 0.75] and s.bestand.tolist() == [40.0, 30.0, 30.0] and s.bestand.sum() == 100.0
        assert (np.diff(s.x_kunden) >= 0).all() and s.x_kunden.min() >= 0 and s.x_kunden.max() < 1
        assert s.nachfrage.sum() == pytest.approx(100.0 * R, rel=1e-12) and s.ueberhang == pytest.approx(R, rel=1e-12)
        assert s.kosten.shape == (3, 30) and s.kosten.max() <= 0.75 and s.kosten.min() >= 0
        roh = s.nachfrage / s.nachfrage.min()                 # ganzzahlige Nachfrage 1 bis 8: Verhältnisse zum kleinsten Wert sind Brüche mit Nenner höchstens 8
        assert (roh >= 1 - 1e-12).all() and roh.max() <= 8.0 + 1e-9
    with pytest.raises(ValueError):
        SC.make_strasse(0, 1.3, 5, 1, 0)


def test_handbeispiel_hat_zwei_lager_drei_kunden_und_laenge_10():
    h = SC.handbeispiel()
    assert h.x_lager.tolist() == [0.0, 10.0] and h.x_kunden.tolist() == [1.0, 5.0, 9.0] and h.laenge == 10.0
    assert h.kosten.tolist() == [[1.0, 5.0, 9.0], [9.0, 5.0, 1.0]] and h.ueberhang == pytest.approx(1.5)


def test_ausreisser_instanz_hat_n_quellen_n_ziele_und_die_letzten_k_liegen_beim_abstand():
    xa, xb = SC.make_ausreisser(5, 10.0, 5, 4, 0)
    assert len(xa) == len(xb) == 100
    assert np.abs(xb[95:] - 10.0).max() < 0.5 and np.abs(xb[:95]).max() < 5.0 and np.abs(xa).max() < 5.0
    assert np.array_equal(xa, SC.make_ausreisser(5, 10.0, 5, 4, 0)[0])
    xa0, xb0 = SC.make_ausreisser(0, 10.0, 5, 4, 0)
    assert len(xb0) == 100 and np.abs(xb0).max() < 5.0
    with pytest.raises(ValueError):
        SC.make_ausreisser(101, 10.0, 5, 4, 0)


def test_ausreisser_gesamt_verteilung_normal_mittelwert_null_und_streuung_eins_ueber_viele_instanzen():
    punkte = np.concatenate([SC.make_ausreisser(0, 10.0, 7, 4, i)[0] for i in range(60)])
    assert abs(punkte.mean()) < 0.06 and abs(punkte.std() - 1.0) < 0.05          # 6 000 Normalzahlen: Standardfehler 0,013 beziehungsweise 0,009


def test_weiche_instanz_hat_summen_eins_und_1_komma_3_und_groesste_kosten_eins():
    a, b, K = SC.make_weich(5, 5, 0)
    assert a.shape == b.shape == (30,) and a.sum() == pytest.approx(1.0, abs=1e-14) and b.sum() == pytest.approx(1.3, abs=1e-14)
    assert K.max() == 1.0 and K.min() >= 0 and K.shape == (30, 30) and (a > 0).all() and (b > 0).all()


def test_rng_ist_eine_bytegleiche_kopie_der_vorlage_im_website_repo_falls_vorhanden():
    vorlage = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not vorlage.exists():
        pytest.skip("Vorlage nicht vorhanden (Demo allein ausgecheckt)")
    assert (ROOT / "ugm_rng.py").read_bytes() == vorlage.read_bytes()


def test_simplex_ist_eine_kopie_der_quelle_in_zuordnen_transportieren_demo_falls_vorhanden():
    quelle = ROOT.parent / "zuordnen-transportieren-demo" / "ztr_simplex.py"
    if not quelle.exists():
        pytest.skip("Quelle nicht vorhanden (Demo allein ausgecheckt)")
    assert (ROOT / "ugm_simplex.py").read_text(encoding="utf-8") == quelle.read_text(encoding="utf-8").replace("ztr_", "ugm_")


def test_splitmix64_bekannte_ausgaben_fuer_seed_0():
    from ugm_rng import SplitMix64

    g = SplitMix64(0)
    assert [g.next_u64() for _ in range(4)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F, 0xF88BB8A8724C81EC]


def test_zweiter_prozess_mit_anderem_hashseed_rechnet_dieselbe_strasse_und_dieselbe_reihe():
    code = ("import sys; sys.path.insert(0, r'%s'); sys.dont_write_bytecode = True\n"
            "import ugm_scenario as SC, ugm_evaluation as E\n"
            "s = SC.make_strasse(20, 1.3, 5, 1, 0)\n"
            "r = E.reihe_normalisieren(1.3, 1.0, 20, 5)\n"
            "print(repr((float(s.nachfrage.sum()), float(s.x_kunden.sum()), r['ersparnis'], r['mit_leer'])))" % str(ROOT))
    ausgaben = []
    for hashseed in ("1", "2"):
        r = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": hashseed, "PATH": "", "SYSTEMROOT": "C:\\Windows", "OMP_NUM_THREADS": "1"}, timeout=120)
        assert r.returncode == 0, r.stderr
        ausgaben.append(r.stdout.strip())
    assert ausgaben[0] == ausgaben[1] and ausgaben[0]


def test_konstanten_sind_stimmig():
    assert sum(C.BESTAND) == C.BESTAND_SUMME == 100.0 and len(C.X_LAGER) == len(C.BESTAND) == 3
    assert C.UEBERHANG_WEICH == 1.3 and C.N_WEICH == 30 and C.N_AUSREISSER == 100 and C.SERIEN_SEED == 2026

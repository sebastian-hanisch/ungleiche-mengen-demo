"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzen und einrasten; jedes Preset lässt sich rechnen."""

import pytest

import ugm_constants as C
import ugm_evaluation as E
import ugm_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.SETTING_SPECS), name          # jedes Preset legt ALLE Regler fest
        assert C.PRESET_HELP[name]
    assert list(C.PRESETS) == ["Straße: Nachfrage 130 gegen Bestand 100", "Von Hand: 2 Lager, 3 Kunden", "Gleiche Summe", "Schwache Strafe", "Ausreißer 5 %"]


def test_preset_values_are_inside_the_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, wert in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert wert in spec.options, (name, key, wert)
            else:
                assert spec.lo <= wert <= spec.hi, (name, key, wert)


def test_no_two_presets_are_identical_and_the_default_is_exactly_one_preset():
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)
    assert [n for n, p in C.PRESETS.items() if p == C.DEFAULTS] == ["Straße: Nachfrage 130 gegen Bestand 100"]
    assert E.Settings() == E.settings_aus_zustand(C.DEFAULTS)


def test_setting_specs_cover_every_preset_key_and_every_default():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) == set(C.DEFAULTS)
    assert sorted(s.url_param for s in P.SETTING_SPECS.values()) == sorted({"q", "r", "p1", "n", "seed", "i1", "p2", "kv", "a3", "d3", "p3", "i3"})       # keine doppelten Adressparameter


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laesst_sich_rechnen_und_zeigt_die_erwartete_einstellung(name):
    s = E.settings_aus_zustand(C.PRESETS[name])
    an = E.ansicht(s)
    assert an.teil.zielwert <= an.norm.zielwert + 1e-9 and an.rho > 0
    if name == "Von Hand: 2 Lager, 3 Kunden":
        assert an.strasse.name == C.QUELLE_HAND and an.ersparnis == pytest.approx(4 / 7) and an.rho == pytest.approx(1.0)
    if name == "Gleiche Summe":
        assert an.strasse.ueberhang == pytest.approx(1.0) and abs(an.ersparnis) < 1e-9
    if name == "Schwache Strafe":
        assert s.rho2 == 0.025
    if name == "Ausreißer 5 %":
        assert (s.anteil3, s.abstand3, s.rho3) == (5, 10.0, 6.0)


def test_einrasten_und_begrenzen_des_permalinks():
    sp = P.SETTING_SPECS
    assert P._eingerastet(sp["n_kunden"], 500) == C.N_MAX and P._eingerastet(sp["n_kunden"], -3) == C.N_MIN
    assert P._eingerastet(sp["ueberhang"], 1.32) == 1.3 and P._eingerastet(sp["ueberhang"], 99.0) == 2.0
    assert P._eingerastet(sp["rho2"], 0.04) in (0.05, 0.025) and P._eingerastet(sp["rho2"], 7.0) == 0.3
    assert P._eingerastet(sp["abstand3"], 99.0) == 15.0 and P._eingerastet(sp["rho3"], -2.0) == 0.0 and P._eingerastet(sp["rho3"], 3.14) == 3.1
    assert P._eingerastet(sp["quelle"], C.QUELLE_HAND) == C.QUELLE_HAND and P._eingerastet(sp["quelle"], "unbekannt") is None
    assert P._eingerastet(sp["inst1"], 120) in (100, 200) and P._eingerastet(sp["anteil3"], 4) in (3, 5)
    assert P._eingerastet(sp["kurve"], True) is True and P._eingerastet(sp["kurve"], False) is False
    assert P._wahrheit("True") is True and P._wahrheit("false") is False and P._wahrheit("1") is True

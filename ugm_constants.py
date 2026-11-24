"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und Hilfetexte.

Rechenaufwand am Regler-Maximum (alles numpy, Zeiten auf dem Entwicklungsrechner, ein Kern; auf der Cloud wahrscheinlich 2- bis 3-mal länger):
- Experiment 1 löst je Instanz zwei Transportprobleme (Normalisieren: 3 × n, Teilbedienung über Hilfsknoten: 4 × (n + 1)); bei 60 Kunden und 200 Instanzen einige Sekunden.
- Experiment 2 rechnet je Instanz einmal die Teiltransport-Kurve (kürzeste erweiternde Wege) und liest daraus jedes ρ ab.
- Experiment 3 rechnet je Instanz eine Kurve mit 100 Quellen und 100 Zielen (100 Wege); 200 Instanzen brauchen einige zehn Sekunden (Spinner, Zwischenspeicher).
- Die Reihen für die Standardeinstellungen liegen vorgerechnet in tools/vorgerechnet.json."""

QUELLE_STRASSE = "Straße (Zufall)"
QUELLE_HAND = "Von Hand: 2 Lager, 3 Kunden"
QUELLEN = (QUELLE_STRASSE, QUELLE_HAND)

X_LAGER = (0.25, 0.5, 0.75)
BESTAND = (40.0, 30.0, 30.0)
BESTAND_SUMME = 100.0

UEBERHANG_STUFEN = (1.0, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0)
RHO1_STUFEN = (0.025, 0.04, 0.06, 0.1, 0.25, 0.5, 1.0, 1.5)
INSTANZEN1_STUFEN = (10, 20, 50, 100, 200)
RHO2_STUFEN = (0.0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3)
ANTEIL3_STUFEN = (0, 1, 2, 3, 5, 7, 10)         # Prozent der Ziele
INSTANZEN3_STUFEN = (20, 50, 100, 200)
RHO_TABELLE = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25)   # Reihe von Experiment 2
UEBERHANG_TABELLE = (1.0, 1.15, 1.3, 1.6, 2.0)              # Reihe von Experiment 1

N_MIN, N_MAX, DEFAULT_N = 10, 60, 30            # Zahl der Kunden
ABSTAND_MIN, ABSTAND_MAX = 5.0, 15.0            # Abstand der Ausreißer in Standardabweichungen
RHO3_MIN, RHO3_MAX = 0.0, 10.0                  # ρ in Standardabweichungen
SEED_MAX, DEFAULT_SEED = 999999, 35
N_AUSREISSER = 100                              # Quellen und Ziele des Ausreißer-Experiments
SIGMA_AUSREISSER = 0.1                          # Streuung der Ausreißer um ihren Abstand
N_WEICH = 30                                    # Lager und Kunden im Absatz zur Gesamtmasse
UEBERHANG_WEICH = 1.3                           # Summe der Nachfrage im Absatz zur Gesamtmasse (Summe des Bestands 1)
ANZAHL_WEICH = 10                               # Instanzen des Absatzes zur Gesamtmasse
TAU_EPS = (0.1, 1.0, 10.0, 100.0)               # τ/ε der KL-Strafe
RHO_OPT = (0.1, 0.3, 1.0, 3.0)                  # ρ/OPT des Modells mit Schlupf
EPS_OPT = 0.1                                   # ε = 0,1 · OPT
TOL_KL = 1e-9                                   # Abbruch der KL-Strafe: größte relative Änderung von u und v
MAX_KL = 400000                                 # Durchläufe der KL-Strafe, höchstens

SERIEN_SEED = 2026                              # Zufalls-Seed der Reihen: nie der Regler „Zufalls-Seed“
ZWECK_ANSICHT, ZWECK_REIHE1, ZWECK_STRAFE, ZWECK_AUSREISSER, ZWECK_WEICH = 1, 2, 3, 4, 5   # getrennte Zufallsströme

COLOR_LAGER = "#2F6B65"
COLOR_KUNDE = "#c0504d"
COLOR_NORM = "#4c78a8"
COLOR_TEIL = "#e08a1e"
COLOR_BAND = "rgba(76,120,168,0.18)"
COLOR_GRAU = "rgba(128,128,128,0.55)"

DEFAULTS = {
    "quelle": QUELLE_STRASSE,
    "ueberhang": 1.3,
    "rho1": 1.0,
    "n_kunden": DEFAULT_N,
    "seed_input": DEFAULT_SEED,
    "inst1": 200,
    "rho2": 0.05,
    "kurve": True,
    "anteil3": 5,
    "abstand3": 10.0,
    "rho3": 3.0,
    "inst3": 200,
}

# Presets: jede Zeile legt ALLE Regler fest (Schlüssel wie in DEFAULTS); `aktives_preset` vergleicht jeden Regler.
PRESETS = {
    "Straße: Nachfrage 130 gegen Bestand 100": dict(DEFAULTS),
    "Von Hand: 2 Lager, 3 Kunden": dict(DEFAULTS, quelle=QUELLE_HAND, rho1=0.1),
    "Gleiche Summe": dict(DEFAULTS, ueberhang=1.0),
    "Schwache Strafe": dict(DEFAULTS, rho2=0.025),
    "Ausreißer 5 %": dict(DEFAULTS, anteil3=5, abstand3=10.0, rho3=6.0),
}
PRESET_HELP = {
    "Straße: Nachfrage 130 gegen Bestand 100": "Drei Lager mit zusammen 100 Einheiten, 30 Kunden mit zusammen 130 Einheiten Nachfrage: die Voreinstellung. Ansicht oben und Experimente 1 und 2.",
    "Von Hand: 2 Lager, 3 Kunden": "Zwei Lager bei 0 und 10 (Bestand 2 und 2), drei Kunden bei 1, 5 und 9 (Nachfrage je 2) auf einer Straße der Länge 10; ρ = 0,1 Straßenlängen, also 1. Die Handrechnung steht im Expander „Mathematische Formulierung“.",
    "Gleiche Summe": "Bestand und Nachfrage sind beide 100 (R = 1): Normalisieren und Teilbedienung sind dasselbe Problem, solange ρ nicht unter ρ_voll liegt (Experiment 1 mit ρ = 1 Straßenlänge); die Ersparnis ist dann 0.",
    "Schwache Strafe": "Experiment 2 mit ρ = 0,025 Straßenlängen: die Strafe ist so klein, dass die Teilbedienung nur einen Teil des Bestands bewegt.",
    "Ausreißer 5 %": "Experiment 3: 5 von 100 Zielen liegen 10 Standardabweichungen weit weg; bei ρ = 6 Standardabweichungen werden in der Reihe von Experiment 3 in 179 von 200 Instanzen alle fünf bedient (Mittel von ρ_alle: 5,17).",
}

"""Szenario: Straße mit Lagern und Kunden, Ausreißer-Instanzen und 30 × 30-Instanzen aus dem Hausgenerator SplitMix64 (`ugm_rng.py`).

Alles ist deterministisch aus (Zufalls-Seed, Zweck, Index): gleiche Eingabe, gleiche Instanz, auf jeder Plattform.
Straße (Länge 1): drei Lager bei 1/4, 1/2, 3/4 mit Beständen 40, 30, 30 (Summe 100); die Kunden liegen gleichverteilt auf [0, 1] (sortiert); ihre Nachfrage ist
ganzzahlig 1 bis 8 und wird auf R · 100 skaliert, R = Nachfrage / Bestand. Die Kosten sind die Wege |x − y| (höchstens 1).
Ausreißer: n Quellen und n Ziele, standardnormal; k der Ziele liegen stattdessen bei „Abstand“ + 0,1 · Normalzahl (in Standardabweichungen); jeder Punkt hat die Masse 1.
Dirichlet(1)-Gewichte sind n unabhängige Exponentialzahlen, durch ihre Summe geteilt."""

from dataclasses import dataclass

import numpy as np

import ugm_constants as C
from ugm_rng import MASK, SplitMix64


@dataclass(frozen=True)
class Strasse:
    x_lager: np.ndarray
    bestand: np.ndarray
    x_kunden: np.ndarray
    nachfrage: np.ndarray
    laenge: float = 1.0
    name: str = ""

    @property
    def kosten(self) -> np.ndarray:
        return np.abs(self.x_lager[:, None] - self.x_kunden[None, :])

    @property
    def ueberhang(self) -> float:
        """R = Nachfrage / Bestand."""
        return float(self.nachfrage.sum() / self.bestand.sum())


def instanz_rng(seed: int, zweck: int, index: int) -> SplitMix64:
    """Eigener Zufallsstrom je (Seed, Zweck, Index); die ersten Zahlen werden verworfen, damit benachbarte Indizes unabhängig wirken."""
    g = SplitMix64((int(seed) * 1_000_003 + int(zweck) * 7_919 + int(index) * 104_729 + 12_345) & MASK)
    g.next_u64()
    g.next_u64()
    return g


def dirichlet(g: SplitMix64, n: int, summe: float = 1.0) -> np.ndarray:
    e = np.array([g.expovariate(1.0) for _ in range(n)])
    return summe * e / e.sum()


def make_strasse(n_kunden: int, ueberhang: float, seed: int, zweck: int, index: int) -> Strasse:
    """Die Straße der Demo. Reihenfolge der Ziehungen: erst die Orte der Kunden, dann ihre Nachfragen."""
    if n_kunden < 1:
        raise ValueError("mindestens ein Kunde")
    g = instanz_rng(seed, zweck, index)
    x = np.sort(np.array([g.random() for _ in range(n_kunden)]))
    roh = np.array([g.below(8) + 1 for _ in range(n_kunden)], dtype=float)
    nachfrage = roh / roh.sum() * C.BESTAND_SUMME * ueberhang
    return Strasse(np.array(C.X_LAGER), np.array(C.BESTAND), x, nachfrage, 1.0, C.QUELLE_STRASSE)


def handbeispiel() -> Strasse:
    """Handbeispiel 1: Lager bei 0 und 10 (Bestand 2, 2), Kunden bei 1, 5, 9 (Nachfrage 2, 2, 2); die Straße hat die Länge 10."""
    return Strasse(np.array([0.0, 10.0]), np.array([2.0, 2.0]), np.array([1.0, 5.0, 9.0]), np.array([2.0, 2.0, 2.0]), 10.0, C.QUELLE_HAND)


def make_ausreisser(anzahl_aus: int, abstand: float, seed: int, zweck: int, index: int, n: int = C.N_AUSREISSER):
    """Quellen xa (n Punkte) und Ziele xb (n Punkte, die letzten `anzahl_aus` sind die Ausreißer bei `abstand` ± 0,1). Gibt (xa, xb) zurück."""
    if not 0 <= anzahl_aus <= n:
        raise ValueError("Zahl der Ausreißer außerhalb des Bereichs")
    g = instanz_rng(seed, zweck, index)
    xa = np.array([g.normal() for _ in range(n)])
    normal = np.array([g.normal() for _ in range(n - anzahl_aus)])
    aus = np.array([abstand + 0.1 * g.normal() for _ in range(anzahl_aus)])
    return xa, np.concatenate([normal, aus])


def make_weich(seed: int, zweck: int, index: int, n: int = C.N_WEICH, ueberhang: float = C.UEBERHANG_WEICH):
    """30 × 30-Instanz der weichen Strafe: Punkte im Einheitsquadrat, Kosten = quadrierter Abstand / größter Wert, a ~ Dirichlet(1) mit Summe 1,
    b ~ Dirichlet(1) mit Summe `ueberhang` (unabhängig gezogen). Reihenfolge: Orte der Lager, Orte der Kunden, a, b. Gibt (a, b, Kosten) zurück."""
    g = instanz_rng(seed, zweck, index)
    X = np.array([[g.random(), g.random()] for _ in range(n)])
    Y = np.array([[g.random(), g.random()] for _ in range(n)])
    a = dirichlet(g, n, 1.0)
    b = dirichlet(g, n, ueberhang)
    q = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return a, b, q / q.max()

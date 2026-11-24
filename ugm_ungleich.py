"""Ungleiche Mengen: Normalisieren und Teilbedienung (nur numpy und Standardbibliothek, ohne Streamlit).

Aufgabe: n Lager mit Beständen a_i, m Kunden mit Nachfragen b_j, Kosten c_ij (hier Wege auf einer Straße). Die Summen von Bestand und Nachfrage sind verschieden.
Zwei Wege, damit umzugehen:
  Normalisieren   Nachfrage mit Σa/Σb multiplizieren (beide Seiten auf dieselbe Summe), dann das gewöhnliche Transportproblem lösen. Jeder Kunde bekommt denselben Anteil
                  seiner Nachfrage.
  Teilbedienung   Strafe ρ je vernichtete Einheit Bestand (nicht bewegt) und je erzeugte Einheit Nachfrage (nicht geliefert): min <C, P> + ρ (Σ(a − P1) + Σ(b − Pᵀ1)).
                  Gerechnet über Hilfsknoten: ein Schein-Kunde Z mit der Nachfrage Σa (Kosten Lager → Z gleich ρ) und ein Schein-Lager S mit dem Bestand Σb (Kosten S → Kunde
                  gleich ρ, S → Z gleich 0). Das ist ein ausgeglichenes Transportproblem, das der Transportsimplex (`ugm_simplex.py`, Kopie aus der Demo „Vom Zuordnen zum
                  Transportieren“) löst.
Die Strafe gilt je vernichtete und je erzeugte Einheit; eine bewegte Einheit spart also 2ρ. Daraus folgt die Kappung: ein Weg, der länger als 2ρ ist, wird nie bedient."""

from dataclasses import dataclass

import numpy as np

from ugm_simplex import loese_transport

LEER = 1e-7          # gelieferte Menge höchstens so viel: der Kunde ist leer
REGEL = "steilste"   # Eintrittsregel des Simplex (die schnellste der vier; der Zielwert hängt nicht an ihr)


@dataclass(frozen=True)
class Bedienung:
    plan: np.ndarray            # n × m, bewegte Mengen
    geliefert: np.ndarray       # je Kunde
    bewegt: float               # Summe des Plans
    bewegungskosten: float      # <C, P> ohne Strafe
    vernichtet: float           # nicht bewegter Bestand
    erzeugt: float              # nicht gelieferte Nachfrage
    zielwert: float             # Bewegungskosten + ρ (vernichtet + erzeugt)
    rho: float


def hilfsknoten(bestand, nachfrage, C, rho):
    """Das ausgeglichene Transportproblem der Hilfsknoten: Zeilen = Lager + Schein-Lager S, Spalten = Kunden + Schein-Kunde Z. Gibt (a2, b2, C2) zurück."""
    a = np.asarray(bestand, float)
    b = np.asarray(nachfrage, float)
    C = np.asarray(C, float)
    n, m = C.shape
    a2 = np.concatenate([a, [b.sum()]])
    b2 = np.concatenate([b, [a.sum()]])
    C2 = np.zeros((n + 1, m + 1))
    C2[:n, :m] = C
    C2[:n, m] = rho
    C2[n, :m] = rho
    C2[n, m] = 0.0
    return a2, b2, C2


def teilbedienung(bestand, nachfrage, C, rho, regel: str = REGEL) -> Bedienung:
    """Das Modell mit Strafe ρ je vernichtete und je erzeugte Einheit, über die Hilfsknoten gelöst."""
    a = np.asarray(bestand, float)
    b = np.asarray(nachfrage, float)
    C = np.asarray(C, float)
    n, m = C.shape
    if rho < 0:
        raise ValueError("ρ darf nicht negativ sein")
    if a.sum() <= 0 or b.sum() <= 0:
        return _leer(a, b, C, rho)
    a2, b2, C2 = hilfsknoten(a, b, C, rho)
    sol = loese_transport(a2, b2, C2, regel)
    P = sol.plan[:n, :m].copy()
    return _bedienung(P, a, b, C, rho)


def normalisieren(bestand, nachfrage, C, rho: float = 0.0, regel: str = REGEL) -> Bedienung:
    """Beide Seiten auf dieselbe Summe bringen (die Nachfrage mal Σa/Σb), dann das gewöhnliche Transportproblem lösen. `rho` dient nur dazu, den Zielwert mit Strafe auf die in
    diesem Plan nicht gelieferten Einheiten zu bilden (der Plan hängt nicht an ρ)."""
    a = np.asarray(bestand, float)
    b = np.asarray(nachfrage, float)
    C = np.asarray(C, float)
    if a.sum() <= 0 or b.sum() <= 0:
        return _leer(a, b, C, rho)
    b_norm = b * (a.sum() / b.sum())
    sol = loese_transport(a, b_norm, C, regel)
    return _bedienung(sol.plan.copy(), a, b, C, rho)


def _leer(a, b, C, rho) -> Bedienung:
    P = np.zeros(C.shape)
    return _bedienung(P, a, b, C, rho)


def _bedienung(P, a, b, C, rho) -> Bedienung:
    geliefert = P.sum(axis=0)
    bewegt = float(P.sum())
    bk = float((C * P).sum())
    vern = float(a.sum() - bewegt)
    erz = float(b.sum() - bewegt)
    return Bedienung(P, geliefert, bewegt, bk, vern, erz, bk + rho * (vern + erz), float(rho))


def zaehlung(geliefert, nachfrage):
    """(voll, leer, teilweise): Kunden mit der ganzen Nachfrage, mit höchstens LEER und dazwischen."""
    g = np.asarray(geliefert, float)
    b = np.asarray(nachfrage, float)
    leer = int((g <= LEER).sum())
    voll = int(((g >= b - LEER) & (g > LEER)).sum())
    return voll, leer, len(g) - voll - leer


def kappung(C, rho):
    """Gekappte Kosten min(c, 2ρ): bei gleichen Gesamtmassen ist das Modell mit Strafe ein gewöhnliches Transportproblem mit diesen Kosten."""
    return np.minimum(np.asarray(C, float), 2.0 * rho)

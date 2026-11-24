"""Teiltransport-Kurve durch sukzessive kürzeste Wege (nur numpy, ohne Streamlit).

Aufgabe: Wie viel kostet es, genau m Einheiten vom Bestand zu den Kunden zu bewegen (m höchstens min(Bestand, Nachfrage)), ohne Strafe? Das Verfahren der
kürzesten erweiternden Wege (englisch successive shortest paths) bewegt zuerst die billigste Einheit, dann die nächstbilligste unter Umlenken früherer Lieferungen
und so weiter. Die Kosten sind eine konvexe, stückweise lineare Funktion g(m); jeder Schritt k hat eine Steigung s_k (die Grenzkosten je Einheit auf diesem Stück)
und eine Menge t_k.

Folgerung für das Modell mit Strafe ρ je vernichtete und je erzeugte Einheit: Der Zielwert ist g(m) + ρ (Bestand + Nachfrage − 2m), die Steigung der Strafe also
2ρ je bewegter Einheit. Optimal ist, genau die Schritte mit Steigung s_k < 2ρ zu gehen (bei s_k = 2ρ ist beides optimal). Der Umschlag ρ_voll, bei dem der ganze
Bestand bewegt wird, ist deshalb die Hälfte der Grenzkosten der letzten Einheit: ρ_voll = s_letzte / 2. Eine Bisektion über ρ ist dafür nicht nötig.

Das Verfahren folgt dem Lehrbuch (Ahuja, Magnanti, Orlin: Network Flows, Kapitel über kürzeste erweiternde Wege). Kürzeste Wege im Restnetz: ein Weg beginnt bei einem Lager
mit Restbestand, geht vorwärts zu einem Kunden (Kosten c_ij) und rückwärts über Zellen mit Fluss (Kosten −c_ij) zu einem Lager und endet bei einem Kunden mit Restnachfrage.
Es gibt keinen negativen Kreis, weil der Fluss nach jedem Schritt für seine Menge kostenminimal ist; die Wegesuche (Bellman-Ford, dichte Matrizen in numpy) endet deshalb."""

from dataclasses import dataclass

import numpy as np

TOL_REST = 1e-12     # Restmengen und Flüsse darunter gelten als null
TOL_WEG = 1e-12      # Verbesserungen der Weglänge darunter zählen nicht (verhindert Endlosschleifen durch Rundung)


@dataclass(frozen=True)
class Kurve:
    steigung: np.ndarray      # s_k: Kosten je Einheit im Schritt k, nicht fallend
    menge: np.ndarray         # t_k: bewegte Menge im Schritt k
    start: np.ndarray         # Lager, bei dem der Weg beginnt
    ende: np.ndarray          # Kunde, bei dem der Weg endet (nur dessen Lieferung wächst im Schritt)
    wege: tuple               # je Schritt die Zellen des Weges: Liste von (i, j, +1 oder -1)
    n: int
    m: int

    @property
    def schritte(self) -> int:
        return len(self.steigung)

    @property
    def bewegt_gesamt(self) -> float:
        return float(self.menge.sum())

    @property
    def kosten_gesamt(self) -> float:
        return float((self.steigung * self.menge).sum())

    @property
    def grenzkosten_letzte(self) -> float:
        """Grenzkosten der letzten bewegten Einheit des ganzen Bestands (0, wenn nichts zu bewegen ist)."""
        return float(self.steigung[-1]) if self.schritte else 0.0

    @property
    def rho_voll(self) -> float:
        """Kleinste Strafe, bei der der ganze (kleinere) Bestand bewegt wird: die Hälfte der Grenzkosten der letzten Einheit."""
        return self.grenzkosten_letzte / 2.0


def kurve(bestand, nachfrage, C) -> Kurve:
    """Die Teiltransport-Kurve durch kürzeste erweiternde Wege. bestand (n), nachfrage (m), C (n × m) mit C ≥ 0 oder beliebig (keine negativen Kreise bei Start ohne Fluss)."""
    a = np.array(bestand, float)
    b = np.array(nachfrage, float)
    C = np.asarray(C, float)
    n, m = C.shape
    if a.shape != (n,) or b.shape != (m,):
        raise ValueError("bestand (n), nachfrage (m) und C (n x m) müssen zueinander passen")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("Mengen dürfen nicht negativ sein")
    P = np.zeros((n, m))
    ra, rb = a.copy(), b.copy()
    s_liste, t_liste, i_liste, j_liste, wege = [], [], [], [], []
    schranke = 50 * (n + m) + 1000
    while (ra > TOL_REST).any() and (rb > TOL_REST).any():
        if len(s_liste) > schranke:
            raise RuntimeError("Teiltransport-Kurve: keine Konvergenz")
        dL = np.where(ra > TOL_REST, 0.0, np.inf)
        vorL = np.full(n, -1)                  # Vorgänger eines Lagers: der Kunde, von dem der Weg rückwärts kam (-1: Startlager)
        vorK = np.full(m, -1)                  # Vorgänger eines Kunden: das Lager, von dem der Weg vorwärts kam
        dK = np.full(m, np.inf)
        fluss = P > TOL_REST
        for _ in range(n + m + 2):
            # vorwärts: Lager -> Kunde
            kand = dL[:, None] + C
            i_best = np.argmin(kand, axis=0)
            neu = kand[i_best, np.arange(m)]
            besser = neu < dK - TOL_WEG
            if besser.any():
                dK = np.where(besser, neu, dK)
                vorK = np.where(besser, i_best, vorK)
            # rückwärts: Kunde -> Lager über Zellen mit Fluss
            rueck = np.where(fluss, dK[None, :] - C, np.inf)
            j_best = np.argmin(rueck, axis=1)
            neu_l = rueck[np.arange(n), j_best]
            besser_l = neu_l < dL - TOL_WEG
            if besser_l.any():
                dL = np.where(besser_l, neu_l, dL)
                vorL = np.where(besser_l, j_best, vorL)
            if not besser.any() and not besser_l.any():
                break
        ziel = np.where(rb > TOL_REST, dK, np.inf)
        j_end = int(np.argmin(ziel))
        if not np.isfinite(ziel[j_end]):
            break
        # Weg zurückverfolgen: j_end <- vorK <- Lager (<- vorL Kunde) ...
        zellen = []
        j = j_end
        while True:
            i = int(vorK[j])
            zellen.append((i, j, +1))
            if vorL[i] < 0:
                i_start = i
                break
            j = int(vorL[i])
            zellen.append((i, j, -1))
        t = min(ra[i_start], rb[j_end], *[P[i, j] for (i, j, vz) in zellen if vz < 0])
        if t <= TOL_REST:
            raise RuntimeError("Teiltransport-Kurve: Schritt ohne Menge")
        for (i, j, vz) in zellen:
            P[i, j] += vz * t
        P[P < TOL_REST] = 0.0
        ra[i_start] -= t
        rb[j_end] -= t
        s_liste.append(float(ziel[j_end]))
        t_liste.append(float(t))
        i_liste.append(i_start)
        j_liste.append(j_end)
        wege.append(tuple(zellen))
    return Kurve(np.array(s_liste), np.array(t_liste), np.array(i_liste, dtype=int), np.array(j_liste, dtype=int), tuple(wege), n, m)


def schritte_bei(k: Kurve, rho: float, tol: float = 1e-12) -> int:
    """Zahl der Schritte, die beim Preis ρ gegangen werden: alle mit Steigung s_k < 2ρ (bei Gleichheit ist beides optimal; hier gilt „nicht gegangen“)."""
    return int(np.searchsorted(k.steigung, 2.0 * rho - tol, side="left")) if k.schritte else 0


def bewegt_bei(k: Kurve, rho: float) -> float:
    """Bewegte Menge beim Preis ρ."""
    return float(k.menge[: schritte_bei(k, rho)].sum())


def geliefert_bei(k: Kurve, rho: float) -> np.ndarray:
    """Gelieferte Menge je Kunde beim Preis ρ (nur das Ende eines Weges ändert die Lieferung)."""
    s = schritte_bei(k, rho)
    g = np.zeros(k.m)
    np.add.at(g, k.ende[:s], k.menge[:s])
    return g


def plan_bei(k: Kurve, rho: float) -> np.ndarray:
    """Ein optimaler Plan beim Preis ρ (n × m), durch Wiederholen der Wege."""
    P = np.zeros((k.n, k.m))
    for t, weg in zip(k.menge[: schritte_bei(k, rho)], k.wege):
        for (i, j, vz) in weg:
            P[i, j] += vz * t
    P[P < TOL_REST] = 0.0
    return P


def kosten_bei_menge(k: Kurve, menge: float) -> float:
    """g(m): die kleinsten Kosten, genau `menge` Einheiten zu bewegen (stückweise linear)."""
    if menge < 0 or menge > k.bewegt_gesamt + 1e-9:
        raise ValueError("Menge außerhalb des Bereichs der Kurve")
    rest = menge
    summe = 0.0
    for s, t in zip(k.steigung, k.menge):
        d = min(rest, t)
        summe += s * d
        rest -= d
        if rest <= 1e-15:
            break
    return summe


def stuetzstellen(k: Kurve):
    """(Mengen, Kosten) an den Knickstellen der Kurve, beginnend bei (0, 0)."""
    mengen = np.concatenate([[0.0], np.cumsum(k.menge)])
    kosten = np.concatenate([[0.0], np.cumsum(k.steigung * k.menge)])
    return mengen, kosten

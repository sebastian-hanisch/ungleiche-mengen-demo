"""Hilfsmodul der Tests: Optimalwert eines Transportproblems in Brüchen, unabhängig von jedem Löser.

Die Ecken des Transportpolytops sind die Flüsse auf den aufspannenden Bäumen des vollständigen bipartiten Graphen K_{m,n}; wer sie alle aufzählt und die zulässigen nimmt,
findet das Optimum ohne Simplex, HiGHS oder Netzwerkbibliothek. Alles in `fractions.Fraction` (exakt). Nur für kleine Aufgaben (bis etwa 3 × 4)."""

import itertools
from fractions import Fraction as F


def baeume(m, n):
    """Alle aufspannenden Bäume von K_{m,n} als Kantenmengen (i, j)."""
    kanten = [(i, j) for i in range(m) for j in range(n)]
    for teil in itertools.combinations(kanten, m + n - 1):
        elt = list(range(m + n))

        def wurzel(x):
            while elt[x] != x:
                elt[x] = elt[elt[x]]
                x = elt[x]
            return x

        ok = True
        for (i, j) in teil:
            ri, rj = wurzel(i), wurzel(m + j)
            if ri == rj:
                ok = False
                break
            elt[ri] = rj
        if ok:
            yield teil


def baum_fluss(teil, a, b):
    """Der Fluss auf dem Baum `teil` mit den Zeilensummen a und den Spaltensummen b (Blätter abschälen)."""
    m, n = len(a), len(b)
    ra = [F(x) for x in a]
    rb = [F(x) for x in b]
    kanten = set(teil)
    P = [[F(0)] * n for _ in range(m)]
    while kanten:
        gefunden = False
        for i in range(m):
            es = [e for e in kanten if e[0] == i]
            if len(es) == 1:
                (_, j) = es[0]
                P[i][j] = ra[i]
                rb[j] -= ra[i]
                ra[i] = F(0)
                kanten.remove(es[0])
                gefunden = True
                break
        if gefunden:
            continue
        for j in range(n):
            es = [e for e in kanten if e[1] == j]
            if len(es) == 1:
                (i, _) = es[0]
                P[i][j] = rb[j]
                ra[i] -= rb[j]
                rb[j] = F(0)
                kanten.remove(es[0])
                gefunden = True
                break
        assert gefunden
    return P


def ecken(a, b):
    """Alle zulässigen Ecken: {Plan (Tupel von Tupeln): Liste der Bäume}."""
    m, n = len(a), len(b)
    out = {}
    for t in baeume(m, n):
        P = baum_fluss(t, a, b)
        if all(x >= 0 for r in P for x in r):
            out.setdefault(tuple(tuple(r) for r in P), []).append(t)
    return out


def kosten(P, C):
    return sum(P[i][j] * C[i][j] for i in range(len(C)) for j in range(len(C[0])))


def optimum(a, b, C):
    """(Optimalwert, Liste aller optimalen Ecken) in Brüchen."""
    ec = ecken(a, b)
    w = {P: kosten(P, C) for P in ec}
    mn = min(w.values())
    return mn, [P for P, x in w.items() if x == mn]


def hilfsknoten(bestand, nachfrage, C, rho):
    """Ausgeglichene Fassung des Modells mit Strafe: Zeilen = Lager + Schein-Lager, Spalten = Kunden + Schein-Kunde (Brüche)."""
    n, m = len(bestand), len(nachfrage)
    a = list(bestand) + [sum(nachfrage)]
    b = list(nachfrage) + [sum(bestand)]
    CH = [[C[i][j] for j in range(m)] + [rho] for i in range(n)] + [[rho] * m + [F(0)]]
    return a, b, CH

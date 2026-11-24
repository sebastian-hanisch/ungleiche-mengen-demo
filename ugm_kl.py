"""Weiche Randsummen: Skalierungsverfahren mit KL-Strafe (nur numpy, ohne Streamlit).

Aufgabe: minimiere <C, P> + ε KL(P | 1) + τ KL(P1 | a) + τ KL(Pᵀ1 | b) über alle P ≥ 0; die Randsummen müssen nicht stimmen, sie werden mit dem Gewicht τ bestraft
(KL ist die verallgemeinerte Kullback-Leibler-Abweichung Σ x ln(x/y) − x + y). Die Lösung hat die Gestalt P = diag(u) K diag(v) mit dem **nicht normierten** Kern
K = exp(−C/ε); die Skalierungen kommen aus der Iteration
    u ← (a / (K v))^λ,   v ← (b / (Kᵀ u))^λ,   λ = τ / (τ + ε).
Für τ → ∞ (λ → 1) ist das die gewöhnliche Sinkhorn-Iteration mit festen Randsummen, für τ → 0 bleiben u und v bei 1 und P = K.
Quelle: Chizat, Peyré, Schmitzer, Vialard (2018), Scaling algorithms for unbalanced optimal transport problems (nur der Abstract gelesen); das Verfahren ist hier
selbst nachgebaut und nicht gegen eine Bibliothek geprüft; die Stationaritätsbedingung der Zielfunktion prüft ein Test.

Zählung: ein Durchlauf sind zwei Matrix-Vektor-Produkte (K v und Kᵀ u). Abbruch: die größte relative Änderung von u und v unter TOL (float64)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KLErgebnis:
    plan: np.ndarray
    u: np.ndarray
    v: np.ndarray
    durchlaeufe: int
    konvergiert: bool

    @property
    def produkte(self) -> int:
        return 2 * self.durchlaeufe

    @property
    def masse(self) -> float:
        return float(self.plan.sum())


def kl_strafe(a, b, C, eps: float, tau: float, tol: float = 1e-9, max_iter: int = 400000) -> KLErgebnis:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    C = np.asarray(C, float)
    if eps <= 0 or tau <= 0:
        raise ValueError("ε und τ müssen positiv sein")
    if C.shape != (len(a), len(b)) or (a <= 0).any() or (b <= 0).any():
        raise ValueError("a (n), b (m) und C (n x m) müssen zueinander passen und positiv sein")
    K = np.exp(-C / eps)
    KT = np.ascontiguousarray(K.T)
    lam = tau / (tau + eps)
    u = np.ones(len(a))
    v = np.ones(len(b))
    d = np.inf
    k = 0
    while k < max_iter:
        un = (a / (K @ v)) ** lam
        vn = (b / (KT @ un)) ** lam
        d = max(float(np.max(np.abs(un / u - 1.0))), float(np.max(np.abs(vn / v - 1.0))))
        u, v = un, vn
        k += 1
        if d < tol:
            break
    plan = u[:, None] * K * v[None, :]
    return KLErgebnis(plan, u, v, k, bool(d < tol))


def masse_zwei_mal_zwei(tau_eps: float, a_i: float = 0.5, b_j: float = 0.65) -> float:
    """Geschlossene Form des 2 × 2-Beispiels (alle Kosten 0, a = (a_i, a_i), b = (b_j, b_j)): Masse = 4 (a_i b_j / 4)^(τ/(2τ + ε)); τ/ε = `tau_eps`."""
    return 4.0 * (a_i * b_j / 4.0) ** (tau_eps / (2.0 * tau_eps + 1.0))

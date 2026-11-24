"""Gemeinsame Fixtures: das Handbeispiel 1, an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ordner der Straße des Handbeispiels 1: Lager bei 0 und 10 (Bestand 2, 2), Kunden bei 1, 5, 9 (Nachfrage 2, 2, 2).
X_LAGER = np.array([0.0, 10.0])
X_KUNDE = np.array([1.0, 5.0, 9.0])
BESTAND = np.array([2.0, 2.0])
NACHFRAGE = np.array([2.0, 2.0, 2.0])


@pytest.fixture
def hand():
    """(Bestand, Nachfrage, Wege) des Handbeispiels 1: Wege (1, 5, 9) und (9, 5, 1)."""
    return BESTAND.copy(), NACHFRAGE.copy(), np.abs(X_LAGER[:, None] - X_KUNDE[None, :])

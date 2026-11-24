"""Kennzahlen, Reihen und Verdikte der Demo (reine Funktionen, ohne Streamlit).

Aufbau: `ansicht` (eine Straße: Normalisieren gegen Teilbedienung), die Reihen je Experiment (`reihe_normalisieren`, `reihe_strafe`, `reihe_ausreisser`), der Absatz zur
Gesamtmasse bei der Strafe (`weich_messung`) und die Verdikte `verdikt_*` als Funktionen (Art, Text), die die App mit `getattr(st, Art)(Text)` anzeigt.
Jede Zahl eines Verdikts stammt aus der Reihe, die es bekommt.

Zählweise: Instanzen (Zähleinheit der Reihen), Kunden (voll, leer, teilweise), Einheiten Masse, Bewegungskosten in Einheiten mal Weg ohne Strafe, Zielwert mit Strafe;
für die KL-Strafe Matrix-Vektor-Produkte (ein Durchlauf = zwei Produkte). Skala der Strafe: ρ in Straßenlängen (Experimente 1 und 2) beziehungsweise in Standardabweichungen
(Experiment 3) beziehungsweise ρ/OPT beim Vergleich mit der KL-Strafe."""

from dataclasses import dataclass

import numpy as np

import ugm_constants as C
import ugm_teiltransport as T
from ugm_kl import kl_strafe, masse_zwei_mal_zwei
from ugm_scenario import Strasse, handbeispiel, make_ausreisser, make_strasse, make_weich
from ugm_simplex import loese_transport
from ugm_ungleich import LEER, Bedienung, normalisieren, teilbedienung, zaehlung


# ----------------------------------------------------------------------------------------------------------------------------------------
# Einstellungen und Formatierung
# ----------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    quelle: str = C.DEFAULTS["quelle"]
    ueberhang: float = C.DEFAULTS["ueberhang"]
    rho1: float = C.DEFAULTS["rho1"]
    n_kunden: int = C.DEFAULTS["n_kunden"]
    seed: int = C.DEFAULTS["seed_input"]
    inst1: int = C.DEFAULTS["inst1"]
    rho2: float = C.DEFAULTS["rho2"]
    kurve: bool = C.DEFAULTS["kurve"]
    anteil3: int = C.DEFAULTS["anteil3"]
    abstand3: float = C.DEFAULTS["abstand3"]
    rho3: float = C.DEFAULTS["rho3"]
    inst3: int = C.DEFAULTS["inst3"]


_ZUSTAND_ZU_FELD = {"quelle": "quelle", "ueberhang": "ueberhang", "rho1": "rho1", "n_kunden": "n_kunden", "seed_input": "seed", "inst1": "inst1", "rho2": "rho2",
                    "kurve": "kurve", "anteil3": "anteil3", "abstand3": "abstand3", "rho3": "rho3", "inst3": "inst3"}


def settings_aus_zustand(zustand) -> Settings:
    """Settings aus dem Session State (Schlüssel der Regler); fehlende Schlüssel nehmen den Standardwert."""
    felder = {feld: zustand[key] for key, feld in _ZUSTAND_ZU_FELD.items() if key in zustand}
    for feld in ("n_kunden", "seed", "inst1", "anteil3", "inst3"):
        if feld in felder:
            felder[feld] = int(felder[feld])
    for feld in ("ueberhang", "rho1", "rho2", "abstand3", "rho3"):
        if feld in felder:
            felder[feld] = float(felder[feld])
    if "kurve" in felder:
        felder["kurve"] = bool(felder["kurve"])
    return Settings(**felder)


def de(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    text = f"{x:.{stellen}f}"
    if float(text) == 0.0:
        text = text.lstrip("-")                  # kein „−0,0“
    return text.replace(".", ",")


def de_g(x: float, signifikant: int = 3) -> str:
    """Zahl mit Dezimalkomma und höchstens `signifikant` gültigen Stellen (ohne Exponent, wo es lesbar bleibt)."""
    if x == 0:
        return "0"
    text = f"{x:.{signifikant}g}"
    if "e" in text:
        mant, exp = text.split("e")
        return f"{mant.replace('.', ',')}e{int(exp)}"
    return text.replace(".", ",")


def menge(x: float, signifikant: int = 4) -> str:
    """Menge in Einheiten als Text; Gleitkomma-Reste unter der Leer-Schwelle (1e-7) stehen als „0“, nicht als „2,842e-14“."""
    return de_g(0.0 if abs(x) < LEER else x, signifikant)


def strl(x: float) -> str:
    """ρ oder ein Weg in Straßenlängen als Text: „1 Straßenlänge“, „0,05 Straßenlängen“."""
    return de_g(x, 3) + (" Straßenlänge" if abs(x - 1.0) < 1e-12 else " Straßenlängen")


def ganz(x: float) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(round(x)):,}".replace(",", " ")


def prozent(x: float, stellen: int = 1) -> str:
    return de(100.0 * x, stellen) + " %"


def stat(werte):
    """[Mittel, Minimum, Maximum] einer Liste (als Liste, damit die Werte unverändert durch JSON gehen)."""
    v = np.asarray(list(werte), float)
    return [float(v.mean()), float(v.min()), float(v.max())]


def jsonbar(x):
    """Wandelt Ergebnisse in reine Python-Typen um (Listen, Zahlen, Texte): dieselbe Gestalt wie nach dem Laden aus der JSON-Datei."""
    if isinstance(x, dict):
        return {str(k): jsonbar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonbar(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonbar(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def mittel_text(b, stellen: int = 1, prozentwert: bool = True) -> str:
    """Mittel [Minimum; Maximum] als Text (Prozent: die Liste enthält Anteile 0 bis 1)."""
    if prozentwert:
        return f"{de(100 * b[0], stellen)} % [{de(100 * b[1], stellen)}; {de(100 * b[2], stellen)}]"
    return f"{de(b[0], stellen)} [{de(b[1], stellen)}; {de(b[2], stellen)}]"


# ----------------------------------------------------------------------------------------------------------------------------------------
# Eine Straße: Normalisieren gegen Teilbedienung
# ----------------------------------------------------------------------------------------------------------------------------------------

def strasse_der_ansicht(s: Settings) -> Strasse:
    if s.quelle == C.QUELLE_HAND:
        return handbeispiel()
    return make_strasse(s.n_kunden, s.ueberhang, s.seed, C.ZWECK_ANSICHT, 0)


@dataclass(frozen=True)
class Ansicht:
    strasse: Strasse
    rho: float                 # in den Einheiten der Straße (ρ in Straßenlängen mal Länge)
    norm: Bedienung
    teil: Bedienung
    kurve: T.Kurve
    zaehl_norm: tuple          # (voll, leer, teilweise)
    zaehl_teil: tuple

    @property
    def ersparnis(self):
        """1 − Bewegungskosten Teilbedienung / Bewegungskosten Normalisieren (Basis Normalisieren); None, wenn Normalisieren nichts kostet."""
        if self.norm.bewegungskosten <= 0:
            return None
        return 1.0 - self.teil.bewegungskosten / self.norm.bewegungskosten

    @property
    def rho_voll_rel(self) -> float:
        """ρ_voll in Straßenlängen."""
        return self.kurve.rho_voll / self.strasse.laenge

    @property
    def ganzer_bestand(self) -> bool:
        return self.teil.bewegt >= min(self.strasse.bestand.sum(), self.strasse.nachfrage.sum()) - 1e-7


def ansicht(s: Settings) -> Ansicht:
    st = strasse_der_ansicht(s)
    rho = s.rho1 * st.laenge
    norm = normalisieren(st.bestand, st.nachfrage, st.kosten, rho)
    teil = teilbedienung(st.bestand, st.nachfrage, st.kosten, rho)
    kurve = T.kurve(st.bestand, st.nachfrage, st.kosten)
    return Ansicht(st, rho, norm, teil, kurve, zaehlung(norm.geliefert, st.nachfrage), zaehlung(teil.geliefert, st.nachfrage))


def ungeliefert(st: Strasse, b: Bedienung) -> float:
    return float(st.nachfrage.sum() - b.bewegt)


def verdikt_ansicht(an: Ansicht):
    """Kernfrage „Was macht das Normalisieren mit den Kunden?“ für die Straße der Seitenleiste (Art, Text)."""
    st = an.strasse
    R = st.ueberhang
    if abs(R - 1.0) < 1e-9:
        if an.ganzer_bestand:
            return "success", (f"Bestand und Nachfrage haben dieselbe Summe ({de_g(st.bestand.sum(), 4)} Einheiten, R = 1) und ρ liegt nicht unter ρ_voll: Normalisieren ändert nichts, "
                               f"beide Rechnungen sind dasselbe Problem. Die Bewegungskosten sind {de(an.norm.bewegungskosten, 2)} gegen {de(an.teil.bewegungskosten, 2)} (eine Instanz).")
        return "info", (f"Bestand und Nachfrage haben dieselbe Summe ({de_g(st.bestand.sum(), 4)} Einheiten, R = 1), aber ρ = {strl(an.rho / st.laenge)} liegt unter ρ_voll = {de_g(an.rho_voll_rel, 3)}: "
                        f"die Teilbedienung bewegt nur {de_g(an.teil.bewegt, 4)} von {de_g(st.bestand.sum(), 4)} Einheiten, die beiden Rechnungen sind verschieden. "
                        f"Die Bewegungskosten sind {de(an.norm.bewegungskosten, 2)} gegen {de(an.teil.bewegungskosten, 2)} (eine Instanz) und gelten für verschieden viele bewegte Einheiten; ρ ≥ ρ_voll wählen.")
    teile = [f"Normalisieren gibt jedem der {len(st.nachfrage)} Kunden denselben Anteil {prozent(1.0 / R, 1)} seiner Nachfrage (Bestand {de_g(st.bestand.sum(), 4)}, Nachfrage "
             f"{de_g(st.nachfrage.sum(), 4)}, R = {de_g(R, 3)}); die Bewegungskosten sind {de(an.norm.bewegungskosten, 2)}."]
    v, l, t = an.zaehl_teil
    if an.ganzer_bestand:
        e = an.ersparnis
        vergleich = (f"{prozent(abs(e), 1)} {'weniger' if e >= 0 else 'mehr'} als beim Normalisieren (Basis Normalisieren)" if e is not None else "")
        teile.append(f"Die Teilbedienung mit ρ = {strl(an.rho / st.laenge)} bewegt den ganzen Bestand und kostet {de(an.teil.bewegungskosten, 2)} Bewegungskosten, {vergleich}; "
                     f"{v} Kunden bekommen alles, {l} nichts und {t} einen Teil.")
        teile.append(f"Beide Pläne lassen dieselben {de_g(ungeliefert(st, an.teil), 4)} Einheiten ungeliefert; die Normierung ist keine falsche Rechnung, sie legt fest, wer wenig bekommt "
                     "(eine Instanz, Bewegungskosten ohne Strafe).")
    else:
        teile.append(f"Die Teilbedienung bewegt bei ρ = {strl(an.rho / st.laenge)} nur {de_g(an.teil.bewegt, 4)} von {de_g(st.bestand.sum(), 4)} Einheiten (ρ liegt unter "
                     f"ρ_voll = {de_g(an.rho_voll_rel, 3)}); {v} Kunden bekommen alles, {l} nichts und {t} einen Teil. Der Kostenvergleich mit dem Normalisieren gilt dann für weniger bewegte Einheiten "
                     "und sagt wenig; ρ ≥ ρ_voll wählen.")
    return "info", " ".join(teile)


def verdikt_rho(an: Ansicht, rho2_rel: float):
    """Kernfrage „Was legt die Strafe ρ fest, und woher käme sie?“ für die Straße der Seitenleiste bei dem ρ von Experiment 2 (Art, Text)."""
    st = an.strasse
    rho = rho2_rel * st.laenge
    bewegt = T.bewegt_bei(an.kurve, rho)
    gesamt = an.kurve.bewegt_gesamt
    gleichstand = abs(rho - an.kurve.rho_voll) <= 1e-9 * max(1.0, an.kurve.rho_voll)
    zusatz = " Genau bei ρ = ρ_voll gibt es mehrere gleich gute Pläne; gezeigt wird einer davon." if gleichstand else ""
    return "info", (f"Bei ρ = {strl(rho2_rel)} bewegt die Teilbedienung {de_g(bewegt, 4)} von {de_g(gesamt, 4)} Einheiten ({prozent(bewegt / gesamt, 1) if gesamt > 0 else '–'}). "
                    f"Der ganze Bestand wird oberhalb von ρ_voll = {strl(an.rho_voll_rel)} bewegt, das ist die Hälfte der Grenzkosten der letzten bewegten Einheit "
                    f"({strl(an.kurve.grenzkosten_letzte / st.laenge)}).{zusatz} Die Daten legen nur diesen Umschlag fest; die Strafe selbst steht in keinen Daten, sie muss gesetzt werden (eine Instanz).")


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: Normalisieren oder Teilbedienung, über viele Instanzen
# ----------------------------------------------------------------------------------------------------------------------------------------

def reihe_normalisieren(ueberhang: float, rho_rel: float, n_kunden: int, anzahl: int, seed: int = C.SERIEN_SEED):
    """Je Instanz: Bewegungskosten beider Rechnungen, Ersparnis (1 − Teilbedienung / Normalisieren), Kunden leer. Straße der Länge 1, ρ in Straßenlängen.
    Zufallsstrom je Instanz aus (seed, Zweck, Index), unabhängig vom Regler „Zufalls-Seed“."""
    ers, leer, voll, teilw, nk, tk, bew, mit_leer, teurer, nicht_voll = [], [], [], [], [], [], [], 0, 0, 0
    bew_nv = []                                  # bewegter Anteil des Bestands nur in den Instanzen mit ρ unter ρ_voll
    for i in range(anzahl):
        st = make_strasse(n_kunden, ueberhang, seed, C.ZWECK_REIHE1, i)
        norm = normalisieren(st.bestand, st.nachfrage, st.kosten, rho_rel)
        teil = teilbedienung(st.bestand, st.nachfrage, st.kosten, rho_rel)
        v, l, t = zaehlung(teil.geliefert, st.nachfrage)
        ers.append(1.0 - teil.bewegungskosten / norm.bewegungskosten if norm.bewegungskosten > 0 else 0.0)
        leer.append(l / n_kunden)
        voll.append(v / n_kunden)
        teilw.append(t / n_kunden)
        nk.append(norm.bewegungskosten)
        tk.append(teil.bewegungskosten)
        bew.append(teil.bewegt / st.bestand.sum())
        mit_leer += l > 0
        teurer += teil.bewegungskosten > norm.bewegungskosten + 1e-9
        if teil.bewegt < min(st.bestand.sum(), st.nachfrage.sum()) - 1e-7:
            nicht_voll += 1
            bew_nv.append(teil.bewegt / st.bestand.sum())
    return dict(ueberhang=float(ueberhang), rho=float(rho_rel), n_kunden=int(n_kunden), anzahl=int(anzahl), ersparnis=stat(ers), leer=stat(leer), voll=stat(voll), teilweise=stat(teilw),
                norm_kosten=stat(nk), teil_kosten=stat(tk), bewegt=stat(bew), mit_leer=int(mit_leer), teurer=int(teurer), nicht_voll=int(nicht_voll), ersparnis_liste=[float(x) for x in ers],
                bewegt_nicht_voll=stat(bew_nv) if bew_nv else None)


def reihen_ueberhang(rho_rel: float, n_kunden: int, anzahl: int, ueberhaenge=C.UEBERHANG_TABELLE):
    """Die Reihe von Experiment 1 für jeden Überhang R der Tabelle."""
    return [reihe_normalisieren(R, rho_rel, n_kunden, anzahl) for R in ueberhaenge]


def verdikt_normalisieren(r: dict):
    """Verdikt zu Experiment 1 aus der Reihe `reihe_normalisieren` (Art, Text)."""
    R, k, n = r["ueberhang"], r["anzahl"], r["n_kunden"]
    if abs(R - 1.0) < 1e-9:
        abw = max(abs(r["ersparnis"][1]), abs(r["ersparnis"][2]))
        if r["nicht_voll"] == 0 and abw < 1e-9:
            return "success", (f"R = 1 und ρ ≥ ρ_voll: beide Rechnungen sind dasselbe Problem; die Ersparnis ist 0 (Abweichung unter 1e-9, Rundung), in {r['mit_leer']} von {k} Instanzen ist ein Kunde leer. "
                               "Die Normierung ist nur bei ungleicher Summe eine Annahme.")
        bew = r.get("bewegt_nicht_voll")
        dort = f" (in diesen im Mittel {prozent(bew[0], 1)} bewegt)" if bew else ""
        return "info", (f"R = 1, ρ = {strl(r['rho'])}: die Summen sind gleich, aber die Rechnungen sind nicht überall dasselbe Problem. In {r['nicht_voll']} von {k} Instanzen liegt ρ unter ρ_voll: "
                        f"dort bewegt die Teilbedienung nicht den ganzen Bestand{dort}, und der Kostenvergleich gilt für weniger bewegte Einheiten. Die Ersparnis ist im Mittel {prozent(r['ersparnis'][0], 1)}, "
                        f"die größte Abweichung der Ersparnis von 0 ist {de_g(abw, 2)}; in {r['mit_leer']} von {k} Instanzen ist ein Kunde leer.")
    e = r["ersparnis"]
    satz1 = (f"R = {de_g(R, 3)}, ρ = {strl(r['rho'])}, {n} Kunden: Die Teilbedienung kostet im Mittel {prozent(e[0], 1)} weniger Bewegungskosten als das Normalisieren "
             f"(einzelne Instanzen {de(100 * e[1], 1)} bis {de(100 * e[2], 1)} %, Mittel über {k} Instanzen, Basis Normalisieren). Im Mittel bleiben {prozent(r['leer'][0], 1)} der Kunden leer, "
             f"in {r['mit_leer']} von {k} Instanzen dieser Reihe mindestens einer.")
    if r["nicht_voll"] == 0:
        satz2 = (f"In allen {k} Instanzen bewegt die Teilbedienung den ganzen Bestand (ρ ≥ ρ_voll); dann lassen beide Pläne dieselben Einheiten ungeliefert (die Differenz von Nachfrage und Bestand): "
                 f"die Normierung ist nicht falsch, sie legt fest, wer wenig bekommt. Teurer als das Normalisieren war die Teilbedienung in {r['teurer']} Instanzen.")
    else:
        bew = r.get("bewegt_nicht_voll")
        dort = f"in diesen Instanzen im Mittel {prozent(bew[0], 1)}, über alle {k} Instanzen im Mittel {prozent(r['bewegt'][0], 1)} des Bestands bewegt" if bew else f"{prozent(r['bewegt'][0], 1)} im Mittel"
        satz2 = (f"In {r['nicht_voll']} von {k} Instanzen liegt ρ unter ρ_voll: dort bewegt die Teilbedienung nicht den ganzen Bestand ({dort}), "
                 "und der Kostenvergleich gilt für weniger bewegte Einheiten.")
    return "info", satz1 + " " + satz2


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: die Strafe ρ
# ----------------------------------------------------------------------------------------------------------------------------------------

RHO_RASTER = tuple(round(0.005 * i, 3) for i in range(0, 61))      # ρ von 0 bis 0,3 Straßenlängen für die Kurve des Diagramms


def reihe_strafe(rho_liste, ueberhang: float, n_kunden: int, anzahl: int, seed: int = C.SERIEN_SEED):
    """Je Instanz die Teiltransport-Kurve (kürzeste erweiternde Wege); daraus bewegter Anteil des Bestands und leere Kunden für jedes ρ der Liste, dazu ρ_voll je Instanz
    und die Kurve des mittleren bewegten Anteils auf dem ρ-Raster (Straße der Länge 1)."""
    bew = {r: [] for r in rho_liste}
    leer = {r: [] for r in rho_liste}
    raster = {r: [] for r in RHO_RASTER}
    rho_voll, grenz = [], []
    for i in range(anzahl):
        st = make_strasse(n_kunden, ueberhang, seed, C.ZWECK_STRAFE, i)
        k = T.kurve(st.bestand, st.nachfrage, st.kosten)
        gesamt = st.bestand.sum()
        for r in rho_liste:
            bew[r].append(T.bewegt_bei(k, r) / gesamt)
            leer[r].append(float((T.geliefert_bei(k, r) <= LEER).mean()))
        for r in RHO_RASTER:
            raster[r].append(T.bewegt_bei(k, r) / gesamt)
        rho_voll.append(k.rho_voll)
        grenz.append(k.grenzkosten_letzte)
    return dict(ueberhang=float(ueberhang), n_kunden=int(n_kunden), anzahl=int(anzahl), rho=[float(r) for r in rho_liste], bewegt=[stat(bew[r]) for r in rho_liste],
                leer=[stat(leer[r]) for r in rho_liste], rho_voll=stat(rho_voll), grenzkosten=stat(grenz), raster=[float(r) for r in RHO_RASTER],
                raster_bewegt=[stat(raster[r]) for r in RHO_RASTER], rho_voll_liste=[float(x) for x in rho_voll])


def verdikt_strafe(r: dict):
    """Verdikt zu Experiment 2 aus der Reihe `reihe_strafe` (Art, Text)."""
    k = r["anzahl"]

    def bei(x):
        return r["bewegt"][min(range(len(r["rho"])), key=lambda i: abs(r["rho"][i] - x))], min(range(len(r["rho"])), key=lambda i: abs(r["rho"][i] - x))

    teile = []
    for x in (0.05, 0.1):
        if any(abs(rr - x) < 1e-9 for rr in r["rho"]):
            b, i = bei(x)
            teile.append(f"ρ = {de_g(x, 3)}: im Mittel {prozent(b[0], 1)} des Bestands bewegt ({prozent(r['leer'][i][0], 1)} der Kunden leer)")
    satz1 = ("Mittel über " + f"{k} Instanzen, R = {de_g(r['ueberhang'], 3)}, {r['n_kunden']} Kunden: " + "; ".join(teile) + ".") if teile else ""
    rv = r["rho_voll"]
    satz2 = (f"Der Umschlag ρ_voll (der ganze Bestand wird bewegt) liegt im Mittel bei {de(rv[0], 3)} Straßenlängen (einzelne Instanzen {de(rv[1], 3)} bis {de(rv[2], 3)}) und ist gleich der "
             f"halben Grenzkosten der letzten bewegten Einheit (Mittel der Grenzkosten {de(r['grenzkosten'][0], 3)}). Die Strafe selbst steht in keinen Daten, sie muss gesetzt werden.")
    return "info", (satz1 + " " + satz2).strip()


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Ausreißer
# ----------------------------------------------------------------------------------------------------------------------------------------

def ausreisser_groessen(xa, xb, anzahl_aus: int) -> dict:
    """Die Größen des Experiments für gegebene Punkte: n Quellen xa und n Ziele xb mit Masse 1; die letzten `anzahl_aus` Ziele (nach der Reihenfolge von xb) sind die Ausreißer.
    Anteil der Ausreißer an den Wegen (a) der sortierten Zuordnung, (b) der Zuordnung, die das Verfahren der kürzesten Wege am Ende liefert, (c) am Integral von |F − G|
    oberhalb der Schnittstelle (größter Nicht-Ausreißer-Punkt beider Seiten); ρ_erste und ρ_alle (ab dem der erste und der letzte Ausreißer bedient wird), d_min/2 und d_max/2."""
    xa = np.asarray(xa, float)
    xb = np.asarray(xb, float)
    n = len(xa)
    kosten = np.abs(xa[:, None] - xb[None, :])
    kurve = T.kurve(np.ones(n), np.ones(len(xb)), kosten)
    erste = len(xb) - anzahl_aus
    d = dict(xa=xa, xb=xb, kosten=kosten, kurve=kurve, anzahl_aus=anzahl_aus)
    if anzahl_aus == 0:
        d.update(anteil_a=None, anteil_b=None, anteil_c=None, rho_erste=None, rho_alle=None, d_min2=None, d_max2=None, schwellen=[])
        return d
    sa, sb = np.sort(xa), np.sort(xb)
    wege = np.abs(sa - sb)
    d["anteil_a"] = float(wege[len(sb) - anzahl_aus:].sum() / wege.sum())
    plan = T.plan_bei(kurve, 1e9)
    teuer = kosten * plan
    d["anteil_b"] = float(teuer[:, erste:].sum() / teuer.sum())
    schnitt = max(float(xa.max()), float(xb[:erste].max()) if erste > 0 else float(xa.max()))
    d["anteil_c"] = float(np.maximum(xb[erste:] - schnitt, 0.0).sum() / wege.sum())
    # Schwellen: ρ, ab dem jeder Ausreißer die erste Menge bekommt (Schritt, an dessen Ende er liegt, halbe Steigung)
    schwellen = []
    gesehen = set()
    for idx in range(kurve.schritte):
        j = int(kurve.ende[idx])
        if j >= erste and j not in gesehen:
            gesehen.add(j)
            schwellen.append(float(kurve.steigung[idx] / 2.0))
    schwellen.sort()
    d["schwellen"] = schwellen
    d["rho_erste"] = schwellen[0]
    d["rho_alle"] = schwellen[-1]
    nah = kosten[:, erste:].min(axis=0)
    d["d_min2"] = float(nah.min() / 2.0)
    d["d_max2"] = float(nah.max() / 2.0)
    return d


def ausreisser_instanz(anzahl_aus: int, abstand: float, seed: int, zweck: int, index: int, n: int = C.N_AUSREISSER):
    """Eine Ausreißer-Instanz der Demo (Punkte aus dem Hausgenerator) mit den Größen von `ausreisser_groessen`."""
    xa, xb = make_ausreisser(anzahl_aus, abstand, seed, zweck, index, n)
    return ausreisser_groessen(xa, xb, anzahl_aus)


def ausreisser_bedient(d: dict, rho: float) -> int:
    """Zahl der Ausreißer, die beim Preis ρ bedient werden."""
    return int(sum(1 for s in d["schwellen"] if s < rho))


RHO3_RASTER = tuple(round(0.1 * i, 1) for i in range(0, 101))      # ρ von 0 bis 10 Standardabweichungen


def reihe_ausreisser(anteil_prozent: int, abstand: float, anzahl: int, seed: int = C.SERIEN_SEED):
    """Über `anzahl` Instanzen: die drei Fassungen des Ausreißeranteils, ρ_erste, ρ_alle, d_min/2, d_max/2 und der Anteil bedienter Ausreißer gegen ρ."""
    k_aus = int(round(anteil_prozent * C.N_AUSREISSER / 100))
    a, b, c, rf, ra, dmin, dmax, anteile = [], [], [], [], [], [], [], {r: [] for r in RHO3_RASTER}
    for i in range(anzahl):
        d = ausreisser_instanz(k_aus, abstand, seed, C.ZWECK_AUSREISSER, i)
        if k_aus == 0:
            continue
        a.append(d["anteil_a"])
        b.append(d["anteil_b"])
        c.append(d["anteil_c"])
        rf.append(d["rho_erste"])
        ra.append(d["rho_alle"])
        dmin.append(d["d_min2"])
        dmax.append(d["d_max2"])
        for r in RHO3_RASTER:
            anteile[r].append(ausreisser_bedient(d, r) / k_aus)
    out = dict(anteil_prozent=int(anteil_prozent), anzahl_aus=k_aus, abstand=float(abstand), anzahl=int(anzahl), raster=[float(r) for r in RHO3_RASTER])
    if k_aus == 0:
        return dict(out, leer=True)
    out.update(leer=False, anteil_a=stat(a), anteil_b=stat(b), anteil_c=stat(c), rho_erste=stat(rf), rho_alle=stat(ra), d_min2=stat(dmin), d_max2=stat(dmax),
               erste_ueber_untergrenze=int(sum(1 for x, y in zip(rf, dmin) if x >= y - 1e-9)), alle_ueber_untergrenze=int(sum(1 for x, y in zip(ra, dmax) if x >= y - 1e-9)),
               abstand_erste=float(np.mean(np.asarray(rf) - np.asarray(dmin))), raster_bedient=[stat(anteile[r]) for r in RHO3_RASTER],
               rho_alle_liste=[float(x) for x in ra])
    return out


def anzahl_alle_bedient(r: dict, rho: float) -> int:
    """In wie vielen Instanzen der Reihe `reihe_ausreisser` werden beim Preis ρ alle Ausreißer bedient (ρ_alle < ρ, dieselbe Schwelle wie `ausreisser_bedient`)."""
    return int(sum(1 for x in r["rho_alle_liste"] if x < rho))


def verdikt_ausreisser_instanz(d: dict, rho: float):
    """Kernfrage „Wann wird ein Ausreißer mitgenommen?“ für die Instanz der Seitenleiste bei dem ρ von Experiment 3 (Art, Text)."""
    k = d["anzahl_aus"]
    if k == 0:
        return "info", "Ohne Ausreißer gibt es nichts mitzunehmen: Quellen und Ziele sind beide standardnormal, die Zuordnung bewegt alles."
    bed = ausreisser_bedient(d, rho)
    return "info", (f"Bei ρ = {de_g(rho, 3)} Standardabweichungen werden {bed} von {k} Ausreißern bedient. Der erste wird ab ρ = {de(d['rho_erste'], 2)} bedient, der letzte ab ρ = {de(d['rho_alle'], 2)}; "
                    f"die Untergrenze aus dem kürzesten Weg ist d_min/2 = {de(d['d_min2'], 2)} (eine Instanz, Wege |x − y|, gleiche Gesamtmassen).")


def verdikt_ausreisser(r: dict):
    """Verdikt zu Experiment 3 aus der Reihe `reihe_ausreisser` (Art, Text)."""
    if r["leer"]:
        return "info", "Ohne Ausreißer (0 %) gibt es keinen Anteil und keinen Umschlag; ein Ausreißeranteil über 0 wählen."
    k, n = r["anzahl"], r["anzahl_aus"]
    a, b, c = r["anteil_a"], r["anteil_b"], r["anteil_c"]
    satz1 = (f"{n} von {C.N_AUSREISSER} Zielen ({r['anteil_prozent']} % der Masse), {de_g(r['abstand'], 3)} Standardabweichungen entfernt: Sie machen im Mittel {prozent(a[0], 1)} der Wege der sortierten "
             f"Zuordnung aus (einzelne Instanzen {de(100 * a[1], 0)} bis {de(100 * a[2], 0)} %). Eine andere optimale Zuordnung (die des Verfahrens der kürzesten Wege) gibt {prozent(b[0], 1)}, der vom "
             f"Plan unabhängige Anteil am Integral von |F − G| oberhalb der Schnittstelle {prozent(c[0], 1)}: bei Gleichstand ist der Anteil keine Eigenschaft des Problems, sondern der Definition.")
    satz2 = (f"Der erste Ausreißer wird im Mittel ab ρ = {de(r['rho_erste'][0], 2)} bedient (einzelne Instanzen {de(r['rho_erste'][1], 2)} bis {de(r['rho_erste'][2], 2)}), alle ab ρ = {de(r['rho_alle'][0], 2)}. "
             f"Die Untergrenze d_min/2 (halber kürzester Weg) liegt im Mittel bei {de(r['d_min2'][0], 2)} und wird in {r['erste_ueber_untergrenze']} von {k} Instanzen nicht unterschritten; der Umschlag liegt im Mittel "
             f"{de(r['abstand_erste'], 2)} Standardabweichungen darüber, was zum Kleinbeispiel passt: ein bedienter Ausreißer verdrängt eine Quelle von ihrem Nachbarn. Voraussetzung: gleiche Gesamtmassen, Wege |x − y|.")
    return "info", satz1 + " " + satz2


# ----------------------------------------------------------------------------------------------------------------------------------------
# Absatz: Gesamtmasse bei der Strafe
# ----------------------------------------------------------------------------------------------------------------------------------------

def weich_messung(anzahl: int = C.ANZAHL_WEICH, seed: int = C.SERIEN_SEED):
    """30 × 30-Instanzen mit Σa = 1 und Σb = 1,3 (Dirichlet(1), quadrierter Abstand / größter Wert): Gesamtmasse der KL-Strafe für τ/ε = 0,1 / 1 / 10 / 100 bei ε = 0,1 · OPT (OPT des
    auf Summe 1 normierten Problems), dazu das Modell mit Schlupf für ρ/OPT = 0,1 / 0,3 / 1 / 3 (bewegte Masse, leere Lager)."""
    masse = {t: [] for t in C.TAU_EPS}
    produkte = {t: [] for t in C.TAU_EPS}
    nicht = 0
    bewegt = {r: [] for r in C.RHO_OPT}
    leer = {r: [] for r in C.RHO_OPT}
    for i in range(anzahl):
        a, b, kosten = make_weich(seed, C.ZWECK_WEICH, i)
        opt = loese_transport(a, b * (a.sum() / b.sum()), kosten, "steilste").kosten
        eps = C.EPS_OPT * opt
        for t in C.TAU_EPS:
            res = kl_strafe(a, b, kosten, eps, t * eps, tol=C.TOL_KL, max_iter=C.MAX_KL)
            masse[t].append(res.masse)
            produkte[t].append(res.produkte)
            nicht += not res.konvergiert
        for r in C.RHO_OPT:
            sol = teilbedienung(a, b, kosten, r * opt)
            bewegt[r].append(sol.bewegt)
            leer[r].append(int((sol.plan.sum(axis=1) < 1e-9).sum()))
    return dict(anzahl=int(anzahl), n=C.N_WEICH, tau_eps=list(C.TAU_EPS), masse=[stat(masse[t]) for t in C.TAU_EPS], produkte_median=[float(np.median(produkte[t])) for t in C.TAU_EPS],
                ueber_bestand=[int(sum(1 for x in masse[t] if x > 1.0)) for t in C.TAU_EPS], ueber_beide=[int(sum(1 for x in masse[t] if x > C.UEBERHANG_WEICH)) for t in C.TAU_EPS],
                nicht_konvergiert=int(nicht), rho_opt=list(C.RHO_OPT), schlupf_bewegt=[stat(bewegt[r]) for r in C.RHO_OPT],
                schlupf_ganz=[int(sum(1 for x in bewegt[r] if x >= 1.0 - 1e-7)) for r in C.RHO_OPT], schlupf_leer=[[float(np.mean(leer[r])), int(max(leer[r]))] for r in C.RHO_OPT],
                zwei_mal_zwei=[masse_zwei_mal_zwei(t) for t in C.TAU_EPS], grenzwert=float((1.0 * C.UEBERHANG_WEICH) ** 0.5))


def verdikt_weich(w: dict):
    """Verdikt-Zeile zur Gesamtmasse bei der Strafe aus `weich_messung` (Art, Text)."""
    i1, i10, i100 = w["tau_eps"].index(1.0), w["tau_eps"].index(10.0), w["tau_eps"].index(100.0)
    k = w["anzahl"]
    m1, m10, m100 = w["masse"][i1], w["masse"][i10], w["masse"][i100]
    unter = k - w["ueber_bestand"][i10]
    j3 = w["rho_opt"].index(3.0)
    j01 = w["rho_opt"].index(0.1)
    return "info", (f"Σa = 1, Σb = {de_g(C.UEBERHANG_WEICH, 2)}: bei τ/ε = 1 hat der KL-Plan im Mittel {de(m1[0], 2)} Masse (einzelne Instanzen {de(m1[1], 2)} bis {de(m1[2], 2)}), in {w['ueber_beide'][i1]} von {k} Instanzen über "
                    f"beiden Randsummen; bei τ/ε = 10 im Mittel {de(m10[0], 2)}, {unter} von {k} Instanzen unter dem Bestand 1; bei τ/ε = 100 im Mittel {de(m100[0], 2)}, in {w['ueber_bestand'][i100]} von {k} Instanzen "
                    f"über dem Bestand (Gewichte Dirichlet(1), {w['n']} × {w['n']}, je {k} Instanzen, Verfahren nach Chizat u. a. nachgebaut, nicht gegen eine Bibliothek geprüft). Das Modell mit Schlupf bewegt ab ρ = 3 OPT "
                    f"den ganzen Bestand ({w['schlupf_ganz'][j3]} von {k}) und lässt bei ρ = 0,1 OPT im Mittel {de(w['schlupf_leer'][j01][0], 1)} von {w['n']} Lagern leer.")


# ----------------------------------------------------------------------------------------------------------------------------------------
# Vorrechnung
# ----------------------------------------------------------------------------------------------------------------------------------------

def vorzurechnen():
    """Alle Einträge von tools/vorgerechnet.json als Liste (Name, Parameter, Aufruf): die Ausreißer-Reihe der Standardeinstellungen."""
    d = C.DEFAULTS
    return [("reihe_ausreisser", dict(anteil_prozent=int(d["anteil3"]), abstand=float(d["abstand3"]), anzahl=int(d["inst3"])),
             lambda: reihe_ausreisser(d["anteil3"], d["abstand3"], d["inst3"]))]

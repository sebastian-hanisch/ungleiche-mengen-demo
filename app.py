"""Angebot ungleich Nachfrage: Normalisieren oder Teilbedienung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Bestand und Nachfrage haben selten dieselbe Summe. Das Normalisieren zwingt jedem Kunden denselben Anteil seiner Nachfrage auf; die Teilbedienung mit einer Strafe ρ je nicht gelieferte
Einheit lässt Bestand und Nachfrage verschieden und sagt, was mit dem Rest geschieht. Drei Experimente (Normalisieren oder Teilbedienung, die Strafe ρ, Ausreißer) und ein Absatz zur Gesamtmasse
bei der Strafe. Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ugm_constants as C
import ugm_evaluation as E
import ugm_teiltransport as T
from ugm_layout import dunkles_design, kopf_css
from ugm_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from ugm_visualization import (
    build_ausreisser,
    build_bewegt_rho,
    build_ersparnis,
    build_gesamtmasse,
    build_kurve,
    build_strasse,
)
from ugm_vorgerechnet import hole

st.set_page_config(page_title="Angebot ungleich Nachfrage – Sebastian Hanisch", layout="wide")

SPINNER = "Die Reihe wird gerechnet (bei den Standardeinstellungen liegt sie vorgerechnet bereit) ..."
UEBERHANG2 = 1.3        # Überhang R der Reihe von Experiment 2


@st.cache_data(show_spinner=False)
def _ansicht(settings):
    return E.ansicht(settings)


@st.cache_data(show_spinner=SPINNER)
def _reihe1(ueberhang, rho, n_kunden, anzahl):
    return E.reihe_normalisieren(ueberhang, rho, n_kunden, anzahl)


@st.cache_data(show_spinner=SPINNER)
def _tabelle1(rho, n_kunden, anzahl):
    return E.reihen_ueberhang(rho, n_kunden, anzahl)


@st.cache_data(show_spinner=SPINNER)
def _reihe2(n_kunden, anzahl):
    return E.reihe_strafe(C.RHO_TABELLE, UEBERHANG2, n_kunden, anzahl)


@st.cache_data(show_spinner=SPINNER)
def _reihe3(anteil, abstand, anzahl):
    fertig = hole("reihe_ausreisser", anteil_prozent=int(anteil), abstand=float(abstand), anzahl=int(anzahl))
    return fertig if fertig is not None else E.reihe_ausreisser(anteil, abstand, anzahl)


@st.cache_data(show_spinner=False)
def _instanz3(anteil, abstand, seed):
    d = E.ausreisser_instanz(int(round(anteil * C.N_AUSREISSER / 100)), abstand, seed, C.ZWECK_ANSICHT, 0)
    return {k: d[k] for k in ("anzahl_aus", "anteil_a", "anteil_b", "anteil_c", "rho_erste", "rho_alle", "d_min2", "d_max2", "schwellen")}


@st.cache_data(show_spinner=False)
def _weich():
    return E.weich_messung()


def md_tabelle(kopf, zeilen):
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)] + ["| " + " | ".join(z) + " |" for z in zeilen])


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in ugm_layout.py).
st.html(kopf_css(dunkles_design()))
# Breite Tabellen bekommen einen eigenen waagerechten Rollbalken, damit auf schmalen Bildschirmen nicht die ganze Seite quer rollt.
st.html('<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>')
with st.container(key="kopf"):
    st.title("⚖️ Angebot ungleich Nachfrage: Normalisieren oder Teilbedienung")
st.markdown(
    """
Das gewöhnliche **Transportproblem** (englisch transportation problem) verlangt, dass **Bestand und Nachfrage in der Summe gleich** sind. In der Praxis sind sie es selten: 100 Einheiten im Lager, 130 Einheiten Nachfrage.
Eine stille Antwort ist das **Normalisieren** (englisch normalization): beide Seiten werden auf dieselbe Summe gebracht, und jeder Kunde bekommt denselben Anteil seiner Nachfrage. Die Alternative ist die
**Teilbedienung** (englisch partial transport, auch unausgeglichener Transport, englisch unbalanced transport): Bestand und Nachfrage dürfen verschieden bleiben, und eine **Strafe ρ** je nicht bewegte
und je nicht gelieferte Einheit sagt, was mit dem Rest geschieht. Die Experimente prüfen eine Annahme: *Beide Seiten auf Summe 1 zu normieren ist eine Formalie.*

Weiter unten: **❓ Kernfragen**, die drei **🔬 Experimente**, der Absatz **📏 Gesamtmasse bei der Strafe**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und
**📐 Mathematische Formulierung** (mit den Handrechnungen in Brüchen). Alle Aussagen gelten in diesem Modell: eine Straße, Kosten = Weg, Strafe je nicht bewegte und je nicht gelieferte Einheit.
"""
)

with st.expander("So funktioniert die Teilbedienung", expanded=True):
    st.markdown(
        """
1. **Aufgabe.** Lager mit Beständen $a_i$, Kunden mit Nachfragen $b_j$, Wege $c_{ij}$ auf einer Straße. Die Summen von $a$ und $b$ sind verschieden; der Überhang ist $R = \\sum b / \\sum a$.
2. **Normalisieren.** Die Nachfrage wird mit $1/R$ multipliziert, dann ist die Summe gleich und das gewöhnliche Transportproblem löst die Aufgabe. **Jeder Kunde bekommt denselben Anteil $1/R$** seiner Nachfrage.
3. **Teilbedienung.** Gesucht ist der Plan $P$ mit den kleinsten Bewegungskosten plus Strafe: $\\rho$ je Einheit Bestand, die nicht bewegt wird (**vernichtet**), und je Einheit Nachfrage, die nicht geliefert wird (**erzeugt**).
   Gerechnet wird über zwei **Hilfsknoten** (englisch dummy nodes): ein Schein-Kunde nimmt den unbewegten Bestand zum Preis $\\rho$ auf, ein Schein-Lager liefert die ungedeckte Nachfrage zum Preis $\\rho$; das ist ein gewöhnliches Transportproblem.
4. **Kappung.** Eine bewegte Einheit spart zweimal $\\rho$ (sie vermeidet eine vernichtete und eine erzeugte Einheit). Ein Weg, der länger als $2\\rho$ ist, wird deshalb nie bedient.
5. **Umschlag.** Bewegt man genau $m$ Einheiten, steigen die Kosten $g(m)$ mit wachsender Steigung (die **Teiltransport-Kurve** ist konvex). Der ganze Bestand wird bewegt, sobald $2\\rho$ die Grenzkosten der letzten Einheit erreicht: $\\rho_{\\text{voll}}$ ist
   die **Hälfte der Grenzkosten der letzten bewegten Einheit**. Die Daten legen nur diesen Umschlag fest, nicht den Wert der Strafe.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🧩 Straße und Überhang")
    st.selectbox("Straße oben", C.QUELLEN, key="quelle",
                 help="Die Ansicht „Ergebnis: eine Straße“ und die Kurve in Experiment 2 zeigen das Handbeispiel mit zwei Lagern und drei Kunden oder eine Zufallsstraße (3 Lager, Bestand 100). "
                      "Die Reihen von Experiment 1 bleiben Zufallsstraßen. Beim Handbeispiel gelten Überhang, Kundenzahl und Zufalls-Seed nicht.")
    st.select_slider("Überhang R = Nachfrage / Bestand", options=C.UEBERHANG_STUFEN, key="ueberhang", format_func=lambda x: E.de_g(x, 3),
                     help="Gesamtnachfrage geteilt durch den Gesamtbestand 100. R = 1: gleiche Summe. Wirkt auf die Straße oben und auf Experiment 1.")
    st.slider("Zahl der Kunden", *bounds("n_kunden"), key="n_kunden", help="Kunden der Zufallsstraße, gleichverteilt auf der Straße; Nachfrage 1 bis 8 je Kunde, auf die Gesamtnachfrage R · 100 skaliert.")
    st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Straße generieren", width="stretch", on_click=randomize_seed)
    st.header("🔬 Experiment 1: Normalisieren oder Teilbedienung")
    st.select_slider("Strafe ρ (Straßenlängen)", options=C.RHO1_STUFEN, key="rho1", format_func=lambda x: E.de_g(x, 3),
                     help="Strafe je nicht bewegte und je nicht gelieferte Einheit, in Straßenlängen (Länge der Straße = 1). Ab ρ = 0,5 ist 2ρ nicht kleiner als der längste Weg; in den Reihen wird dann der ganze Bestand bewegt. Wirkt auf die Ansicht oben und Experiment 1.")
    st.select_slider("Instanzen der Reihen", options=C.INSTANZEN1_STUFEN, key="inst1", help="So viele Straßen in den Reihen der Experimente 1 und 2 (Zähleinheit ist die Instanz; Zufalls-Seed der Reihen 2026).")
    st.header("🔬 Experiment 2: Die Strafe ρ")
    st.select_slider("Strafe ρ (Straßenlängen)", options=C.RHO2_STUFEN, key="rho2", format_func=lambda x: E.de_g(x, 3),
                     help="Strafe, bei der die Teilbedienung die Straße oben löst und die Kurve gelesen wird. Die Reihe darunter zeigt den bewegten Anteil über alle Instanzen für feste Stufen.")
    st.toggle("Teiltransport-Kurve zeigen", key="kurve", help="Zeigt die kleinsten Kosten für genau m bewegte Einheiten und die Gerade der Steigung 2ρ durch den Punkt, den die Strafe wählt.")
    st.header("🔬 Experiment 3: Ausreißer")
    st.select_slider("Ausreißeranteil in % der Ziele", options=C.ANTEIL3_STUFEN, key="anteil3", help="Von 100 Zielen liegen so viele weit weg bei „Abstand“ (je Ziel Masse 1, also gleicher Prozentwert der Masse). Alle übrigen Punkte sind standardnormal.")
    st.slider("Abstand der Ausreißer (Standardabweichungen)", *bounds("abstand3"), key="abstand3", step=0.5, help="Ort der Ausreißer in Standardabweichungen; sie streuen mit 0,1 darum.")
    st.slider("Strafe ρ (Standardabweichungen)", *bounds("rho3"), key="rho3", step=0.1, help="Strafe, bei der die Instanz der Seitenleiste gezeigt wird; die Reihe darunter läuft über alle ρ.")
    st.select_slider("Instanzen der Reihe", options=C.INSTANZEN3_STUFEN, key="inst3", help="So viele Instanzen mit je 100 Quellen und 100 Zielen (Zähleinheit ist die Instanz).")

sync_query_params({k: st.session_state[k] for k in C.DEFAULTS})

settings = E.settings_aus_zustand(st.session_state)
hand = settings.quelle == C.QUELLE_HAND

an = _ansicht(settings)
st_ = an.strasse
einheit_bk = "Einheiten mal Weg"

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: eine Straße")
rho_rel = an.rho / st_.laenge
st.caption(f"{st_.name}: {len(st_.bestand)} Lager (Bestand {E.de_g(float(st_.bestand.sum()), 4)}), {len(st_.nachfrage)} Kunden (Nachfrage {E.de_g(float(st_.nachfrage.sum()), 4)}), Überhang R = {E.de_g(st_.ueberhang, 3)}; "
           f"Straße der Länge {E.de_g(st_.laenge, 3)}; ρ = {E.strl(rho_rel)} (in den Einheiten der Wege: {E.de_g(an.rho, 3)}). Bewegungskosten in {einheit_bk}, ohne Strafe.")
v_n, l_n, t_n = an.zaehl_norm
v_t, l_t, t_t = an.zaehl_teil
ers = an.ersparnis
m1, m2, m3 = st.columns(3)
m1.metric("Bewegungskosten beim Normalisieren", E.de(an.norm.bewegungskosten, 2), delta=f"bewegt {E.de_g(an.norm.bewegt, 4)} Einheiten, jeder Kunde {E.prozent(1.0 / st_.ueberhang, 1)} seiner Nachfrage", delta_color="off")
m2.metric("Bewegungskosten bei Teilbedienung", E.de(an.teil.bewegungskosten, 2),
          delta=(f"{E.prozent(abs(ers), 1)} {'weniger' if ers >= 0 else 'mehr'} (Basis Normalisieren)" if ers is not None else "Normalisieren kostet 0"), delta_color="off")
m3.metric("Bewegte Einheiten (Teilbedienung)", E.de_g(an.teil.bewegt, 4), delta=f"von {E.de_g(float(st_.bestand.sum()), 4)} Bestand", delta_color="off")
m4, m5, m6 = st.columns(3)
m4.metric("Kunden voll / leer / teilweise (Teilbedienung)", f"{v_t} / {l_t} / {t_t}", delta=f"Normalisieren: {v_n} / {l_n} / {t_n}", delta_color="off")
m5.metric("Zielwert mit Strafe, Normalisieren", E.de(an.norm.zielwert, 2), delta=f"{E.menge(an.norm.vernichtet + an.norm.erzeugt)} Einheiten vernichtet oder erzeugt", delta_color="off")
m6.metric("Zielwert mit Strafe, Teilbedienung", E.de(an.teil.zielwert, 2), delta=f"{E.menge(an.teil.vernichtet + an.teil.erzeugt)} Einheiten vernichtet oder erzeugt", delta_color="off")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_strasse(st_, an.norm.geliefert, titel="Normalisieren: jeder Kunde bekommt denselben Anteil", farbe="#4c78a8"), width="stretch", key="fig_strasse_norm")
with c2:
    st.plotly_chart(build_strasse(st_, an.teil.geliefert, titel=f"Teilbedienung, ρ = {E.strl(rho_rel)}", farbe="#e08a1e"), width="stretch", key="fig_strasse_teil")
st.caption("Grau: Nachfrage des Kunden, farbig: gelieferte Menge; Dreiecke: Lager mit Bestand. Beim Normalisieren bekommt jeder Kunde denselben Anteil seiner Nachfrage; bei der Teilbedienung bekommen manche Kunden alles, andere nichts oder einen Teil (Zählung oben).")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**Was macht das Normalisieren mit den Kunden?**")
kind, text = E.verdikt_ansicht(an)
getattr(st, kind)(text)
st.markdown("**Was legt die Strafe ρ fest, und woher käme sie?**")
kind, text = E.verdikt_rho(an, settings.rho2)
getattr(st, kind)(text)
st.markdown("**Wann wird ein Ausreißer mitgenommen?**")
inst3 = _instanz3(settings.anteil3, settings.abstand3, settings.seed)
kind, text = E.verdikt_ausreisser_instanz(inst3, settings.rho3)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Normalisieren oder Teilbedienung")
r1 = _reihe1(settings.ueberhang, settings.rho1, settings.n_kunden, settings.inst1)
tab1 = _tabelle1(settings.rho1, settings.n_kunden, settings.inst1)
st.caption(f"Reihe über {r1['anzahl']} Zufallsstraßen (3 Lager, Bestand 100, {r1['n_kunden']} Kunden, Zufalls-Seed der Reihen {C.SERIEN_SEED}), ρ = {E.strl(r1['rho'])}; Bewegungskosten ohne Strafe, "
           "Ersparnis = 1 − Teilbedienung / Normalisieren (Basis Normalisieren). „Leer“ heißt: höchstens 1e-7 Einheiten geliefert. Das Handbeispiel und die einzelne Straße stehen oben.")
st.plotly_chart(build_ersparnis(r1), width="stretch", key="fig_ersparnis")
st.markdown("**Ersparnis und leere Kunden nach Überhang R** (Mittel [Minimum; Maximum] über die Instanzen):")
st.markdown(md_tabelle(
    ["R", "Ersparnis", "Kunden leer (Mittel)", "Instanzen mit leerem Kunden", "Bewegungskosten Normalisieren / Teilbedienung (Mittel)"],
    [[E.de_g(z["ueberhang"], 3), E.mittel_text(z["ersparnis"]), E.prozent(z["leer"][0], 1), f"{z['mit_leer']} von {z['anzahl']}",
      f"{E.de(z['norm_kosten'][0], 2)} / {E.de(z['teil_kosten'][0], 2)}"] for z in tab1]))
kind, text = E.verdikt_normalisieren(r1)
getattr(st, kind)(text)
st.info("**Gegenstimme.** Die Teilbedienung braucht einen Parameter ρ, den das Normalisieren nicht hat; sein Wert ist eine Modellentscheidung, kein Messwert (Experiment 2). Wo beide Verteilungen Wahrscheinlichkeiten sind (Summe 1 per Definition), "
        "ist Normalisieren richtig; die Frage nach der Normierung entsteht nur bei Mengen. Masse zu verwerfen (Ausreißer, Experiment 3) kann auch bei Wahrscheinlichkeiten sinnvoll sein.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Die Strafe ρ")
r2 = _reihe2(settings.n_kunden, settings.inst1)
bewegt_rho = T.bewegt_bei(an.kurve, settings.rho2 * st_.laenge)
st.caption(f"Reihe über {r2['anzahl']} Zufallsstraßen (R = {E.de_g(r2['ueberhang'], 3)}, {r2['n_kunden']} Kunden, Zufalls-Seed der Reihen {C.SERIEN_SEED}); die Kurve zeigt die Straße oben. "
           "Die Teiltransport-Kurve entsteht durch kürzeste erweiternde Wege: zuerst die billigste Einheit, dann die nächste unter Umlenken früherer Lieferungen.")
rho2_roh = settings.rho2 * st_.laenge
voll2, leer2, teil2 = E.zaehlung(T.geliefert_bei(an.kurve, rho2_roh), st_.nachfrage)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Bewegter Anteil des Bestands", E.prozent(bewegt_rho / float(st_.bestand.sum()), 1), delta=f"{E.de_g(bewegt_rho, 4)} Einheiten bei ρ = {E.de_g(settings.rho2, 3)}", delta_color="off")
k2.metric("Kunden voll / leer / teilweise", f"{voll2} / {leer2} / {teil2}", delta="Instanz oben", delta_color="off")
k3.metric("Grenzkosten der letzten bewegten Einheit", E.de_g(an.kurve.grenzkosten_letzte / st_.laenge, 3), delta="in Straßenlängen je Einheit", delta_color="off")
k4.metric("Marke ρ_voll = Grenzkosten / 2", E.de_g(an.rho_voll_rel, 3), delta="in Straßenlängen", delta_color="off")
if settings.kurve:
    st.plotly_chart(build_kurve(an.kurve, rho2_roh, st_.laenge), width="stretch", key="fig_kurve")
    st.caption("Die Kurve ist konvex und stückweise linear; die Gerade hat die Steigung 2ρ (eine bewegte Einheit spart zweimal ρ). Der Berührpunkt ist die Menge, die die Strafe bewegt.")
st.plotly_chart(build_bewegt_rho(r2, settings.rho2), width="stretch", key="fig_bewegt_rho")
st.markdown(md_tabelle(
    ["ρ (Straßenlängen)", "bewegter Anteil des Bestands", "Kunden leer"],
    [[E.de_g(rho, 3), E.mittel_text(b), E.prozent(le[0], 1)] for rho, b, le in zip(r2["rho"], r2["bewegt"], r2["leer"])]))
kind, text = E.verdikt_strafe(r2)
getattr(st, kind)(text)
st.caption("Die steigenden Grenzkosten im Graphen zeigt die Demo ssp-demo (kürzeste erweiternde Wege); Schattenpreise und Zukauf: lp-dualitaet-demo.")

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Ausreißer")
r3 = _reihe3(settings.anteil3, settings.abstand3, settings.inst3)
st.caption(f"Reihe über {r3['anzahl']} Instanzen: 100 Quellen und 100 Ziele, Masse 1 je Punkt, alle standardnormal bis auf {r3['anzahl_aus']} Ziele bei {E.de_g(r3['abstand'], 3)} (± 0,1); beide Gesamtmassen gleich, Wege |x − y|, "
           f"Zufalls-Seed der Reihen {C.SERIEN_SEED}. ρ in Standardabweichungen.")
if r3["leer"]:
    st.info("Ohne Ausreißer (0 %) gibt es keinen Anteil und keinen Umschlag; ein Ausreißeranteil über 0 wählen.")
else:
    e1, e2, e3 = st.columns(3)
    e1.metric("Anteil an den Wegen der sortierten Zuordnung", E.prozent(r3["anteil_a"][0], 1), delta=f"{E.de(100 * r3['anteil_a'][1], 0)} bis {E.de(100 * r3['anteil_a'][2], 0)} % je Instanz", delta_color="off")
    e2.metric("Anteil bei einer anderen optimalen Zuordnung", E.prozent(r3["anteil_b"][0], 1), delta="Zuordnung des Verfahrens der kürzesten Wege", delta_color="off")
    e3.metric("Anteil am Integral von |F − G| oberhalb der Schnittstelle", E.prozent(r3["anteil_c"][0], 1), delta="unabhängig vom gewählten Plan", delta_color="off")
    st.plotly_chart(build_ausreisser(r3, settings.rho3), width="stretch", key="fig_ausreisser")
    st.markdown(md_tabelle(
        ["Größe", "Mittel [Minimum; Maximum] über die Instanzen"],
        [["erster Ausreißer wird bedient ab ρ_erste", E.mittel_text(r3["rho_erste"], 2, False)],
         ["alle Ausreißer werden bedient ab ρ_alle", E.mittel_text(r3["rho_alle"], 2, False)],
         ["Untergrenze d_min/2 (halber kürzester Weg zu einem Ausreißer)", E.mittel_text(r3["d_min2"], 2, False)],
         ["Untergrenze d_max/2 für ρ_alle (halber größter der kürzesten Wege je Ausreißer)", E.mittel_text(r3["d_max2"], 2, False)]]))
kind, text = E.verdikt_ausreisser(r3)
getattr(st, kind)(text)
st.markdown("**Kleinbeispiel von Hand.** Quellen bei 0 und 5, Ziele bei 4 und 20 (der Ausreißer), Masse 1 je Punkt. Der kürzeste Weg zum Ausreißer ist 15, die Untergrenze also 7,5. Bei ρ = 8 ist der Weg kürzer als 2ρ = 16 und wird doch nicht bedient: "
            "wer den Ausreißer bedient, schickt die Quelle bei 5 weg, das Ziel bei 4 bekommt dann die Quelle bei 0 (Weg 4 statt 1, Zusatzweg 3). Der Umschlag liegt bei 2ρ = 15 + 3, also **bei ρ = 9 statt 7,5**; bei ρ = 9 sind beide Pläne gleich teuer.")
st.markdown("**Warum der Anteil eine Definition braucht.** Quellen bei 0 und 1, Ziele bei 2 und 3 (das Ziel bei 3 ist der Ausreißer): beide Zuordnungen kosten 4 (2 + 2 und 3 + 1), beide sind optimal. Der Ausreißer macht 2 von 4 aus (½) bei der sortierten Zuordnung, "
            "3 von 4 (¾) bei der gekreuzten, und oberhalb der Schnittstelle bei 2 liegt ¼ des Integrals von |F − G|: dieselben Kosten, drei Zahlen.")

st.markdown("---")

# --- Gesamtmasse bei der Strafe ------------------------------------------------------------------------------------------------------------

st.subheader("📏 Gesamtmasse bei der Strafe (feste Zahlen, ohne Regler)")
w = _weich()
st.markdown(
    """
Statt die Randsummen hart zu fordern, kann man sie auch **weich** bestrafen: das Skalierungsverfahren mit **KL-Strafe** (Kullback-Leibler; englisch unbalanced Sinkhorn, nach Chizat u. a.) minimiert
$\\langle C,P\\rangle + \\varepsilon\\,\\mathrm{KL}(P\\,|\\,1) + \\tau\\,\\mathrm{KL}(P\\mathbf 1\\,|\\,a) + \\tau\\,\\mathrm{KL}(P^{\\top}\\mathbf 1\\,|\\,b)$ mit dem Gewicht $\\tau$ der Randsummen. Ihre Lage zu den beiden Summen **hängt von $\\tau/\\varepsilon$ ab**: bei schwacher Strafe liegt die Masse weit über beiden,
bei starker Strafe im $2 \\times 2$-Beispiel zwischen ihnen (ab $\\tau/\\varepsilon \\approx 4{,}3$), in den Messungen unten bei $\\tau/\\varepsilon = 100$ über dem Bestand. Der Kern $K = e^{-C/\\varepsilon}$ ist nicht normiert, für $\\tau \\to 0$ ist die Masse die Summe aller Kernwerte, für $\\tau \\to \\infty$ im $2 \\times 2$-Beispiel $\\sqrt{\\sum a \\cdot \\sum b}$.
Das Modell mit Schlupf (scharfe Lösung des linearen Programms) bewegt dagegen höchstens den Bestand.
    """
)
st.plotly_chart(build_gesamtmasse(w), width="stretch", key="fig_gesamtmasse")
st.markdown("**Geschlossene Form im 2 × 2-Beispiel** (alle Kosten 0, $\\sum a = 1$ mit je ½, $\\sum b = 1{,}3$ mit je 0,65): Masse $= 4 \\cdot 0{,}08125^{\\,\\tau/(2\\tau+\\varepsilon)}$.")
st.markdown(md_tabelle(["τ/ε"] + [E.de_g(t, 3) for t in w["tau_eps"]] + ["τ/ε → ∞"],
                       [["Masse im 2 × 2-Beispiel"] + [E.de(x, 3) for x in w["zwei_mal_zwei"]] + [E.de(w["grenzwert"], 4)],
                        [f"Masse, {w['n']} × {w['n']}, Mittel [Minimum; Maximum]"] + [E.mittel_text(b, 2, False) for b in w["masse"]] + ["–"],
                        ["Matrix-Vektor-Produkte (Median)"] + [E.ganz(p) for p in w["produkte_median"]] + ["–"]]))
st.markdown(md_tabelle(["ρ/OPT (Modell mit Schlupf)"] + [E.de_g(r, 2) for r in w["rho_opt"]],
                       [["bewegte Masse, Mittel [Minimum; Maximum]"] + [E.mittel_text(b, 2, False) for b in w["schlupf_bewegt"]],
                        ["leere Lager (Mittel, größte Zahl)"] + [f"{E.de(le[0], 1)} ({le[1]})" for le in w["schlupf_leer"]]]))
kind, text = E.verdikt_weich(w)
getattr(st, kind)(text)
st.caption(f"Messung: {w['anzahl']} Instanzen je Zelle, Gewichte Dirichlet(1), ε = 0,1 OPT (OPT des auf Summe 1 normierten Problems), Abbruch bei relativer Änderung von u und v unter 1e-9 (float64), "
           "Matrix-Vektor-Produkte: ein Durchlauf = zwei. Konvergenz und Zahl der Durchläufe des Verfahrens behandelt die Demo zum Matrix-Abgleichen (Furness, IPF) in derselben Reihe; ρ und τ sind verschiedene Parameter, keine Entsprechung behauptet.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme dieser Demo | Was passiert, wenn sie nicht gilt | Verweise |
|---|---|---|
| **Angebot und Nachfrage sind in der Summe gleich.** | Bei ungleicher Summe legt das Normalisieren fest, wer wenig bekommt (Experiment 1); die Teilbedienung braucht eine Strafe ρ, die in keinen Daten steht (Experiment 2); Ausreißer werden erst ab einer Strafe oberhalb von d/2 mitgenommen (Experiment 3). Wo beide Seiten Wahrscheinlichkeiten sind, ist Normalisieren richtig (Masse zu verwerfen, Experiment 3, kann dort trotzdem sinnvoll sein). | `network-flow-demo` (ungleiche Summe als Modellierungs-Kniff mit Notbeschaffung), `auction-algorithm-demo` (Verzicht-Option mit so großer Strafe, dass erst die Zahl der Paare zählt, dann die Kosten), `lp-dualitaet-demo` (Zukaufspreis) |
| **Eine Strafe ρ je nicht bewegte und je nicht gelieferte Einheit.** | Getrennte Strafen für Fehlmenge (Nachfrage) und Lagerkosten (Bestand) sind nicht untersucht; die Kappung wäre min(c, ρ_a + ρ_b) statt min(c, 2ρ). | |
| **Überhang R ≥ 1 auf einer Straße mit drei Lagern.** | Bestand größer als Nachfrage ist nach Vertauschen von Lager und Kunde dieselbe Rechnung, aber nicht gemessen; ein zweidimensionaler Ort ist nicht gemessen. | `ssp-demo` (Teiltransport-Kurve im Graphen) |
| **Das Skalierungsverfahren mit KL-Strafe ist hier selbst nachgebaut.** | Nicht gegen eine Bibliothek geprüft; nur Gewichte Dirichlet(1), 30 × 30, je 10 Instanzen; Konvergenz und Durchläufe sind nicht Thema dieser Demo. | Demo zum Matrix-Abgleichen (Furness, IPF) |
"""
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.**
- Straße der Länge 1: Lager bei ¼, ½, ¾ mit Beständen 40, 30, 30; die Kunden gleichverteilt, Nachfrage ganzzahlig 1 bis 8, auf R · 100 skaliert. Kosten = Weg |x − y|, höchstens 1. Experiment 3: standardnormale Punkte auf der Zahlengeraden, Masse 1 je Punkt, Wege |x − y|.
- Das Modell mit Strafe: ρ je nicht bewegte und je nicht gelieferte Einheit (eine bewegte Einheit spart 2ρ). Bewegungskosten stehen immer ohne Strafe, der Zielwert mit Strafe daneben.
- Zufall aus dem Hausgenerator SplitMix64; die Reihen benutzen feste Instanzen (Zufalls-Seed 2026) unabhängig vom Regler „Zufalls-Seed“, der nur die Straße oben und die Instanz in den Kernfragen bestimmt.

**Grenzen.**
- Alle Aussagen gelten in diesem Modell, bei diesen Parametern und mit der genannten Zähleinheit (Instanz, Kunde, Einheit Masse). Ein Mittel über 200 Instanzen ist kein Gesetz; Minimum und Maximum sind Extremwerte, die von Stichprobe zu Stichprobe springen.
- „Teilbedienung ist nie teurer als Normalisieren“ gilt bei Überhang R ≥ 1 für jedes ρ ≥ 0 als Satz: der normalisierte Plan ist für das Modell mit Schlupf zulässig, daraus folgt Bewegungskosten (Teilbedienung) ≤ Bewegungskosten (Normalisieren) − 2ρ · (Bestand − bewegte Menge); dazu die Zählung in den Reihen dieser Demo. Ein Vergleich auf gleicher Grundlage ist es erst ab ρ_voll.
- Der Anteil der Ausreißer an den Kosten ist bei Gleichstand keine Eigenschaft des Problems: er hängt an der Definition und am gewählten optimalen Plan (Experiment 3). Welche optimale Zuordnung das Verfahren der kürzesten Wege liefert, ist eine Eigenschaft des Verfahrens.
- Das Normalisieren ist nicht falsch: ab ρ_voll lassen beide Pläne dieselben Einheiten ungeliefert; die Normierung legt fest, wer wenig bekommt. Unter ρ_voll bewegt die Teilbedienung weniger, und der Kostenvergleich gilt dann für weniger bewegte Einheiten.
- Bestand größer als Nachfrage, zweidimensionale Orte, getrennte Strafen für beide Seiten und andere Gewichte der KL-Strafe als Dirichlet(1) sind nicht untersucht.

**Bekannte Probleme und Unsicherheiten.**
- Die Zahlen sind mit Zufalls-Seed 2026 gemessen; andere Zufalls-Seeds geben andere Zahlen im Band, nicht dieselben.
- Bei Gleichstand (zum Beispiel ρ genau gleich der halben Grenzkosten) gibt es mehrere optimale Pläne; gezeigt wird eine Ecke.
- Die Oberfläche ist nicht mit echten Mausklicks im Browser durchgeklickt (AppTest und Tests sehen Plotly nicht).
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* Bestände $a \in \mathbb{R}^n_{\ge 0}$, Nachfragen $b \in \mathbb{R}^m_{\ge 0}$ mit $\sum a \ne \sum b$, Wege $c_{ij} \ge 0$ und eine Strafe $\rho \ge 0$. *Gesucht:* ein Plan $P \ge 0$ mit
$P\mathbf 1 \le a$ und $P^{\top}\mathbf 1 \le b$, der Bewegungskosten plus Strafe für den nicht bewegten Bestand und die nicht gelieferte Nachfrage minimiert. Der Überhang ist $R = \sum b / \sum a$.

| Symbol | Bedeutung |
|---|---|
| $n$, $m$ | Zahl der Lager und der Kunden |
| $a$, $b$ | Bestände und Nachfragen |
| $c_{ij}$ | Weg vom Lager $i$ zum Kunden $j$ |
| $P$ | Plan, $P_{ij}$ = bewegte Menge von Lager $i$ zu Kunde $j$ |
| $R$ | Überhang $\sum b / \sum a$ |
| $\rho$ | Strafe je nicht bewegte und je nicht gelieferte Einheit |
| $s_i$, $t_j$ | nicht bewegter Bestand (vernichtet) und nicht gelieferte Nachfrage (erzeugt) |
| $g(m)$ | kleinste Bewegungskosten für genau $m$ bewegte Einheiten |
| $\rho_{\text{voll}}$ | kleinste Strafe, bei der der ganze (kleinere) Bestand bewegt wird |

**Normalisieren.** Mit $\tilde b = b \cdot \sum a / \sum b$ ist die Summe gleich, und
$$\min_{P \ge 0,\; P\mathbf 1 = a,\; P^{\top}\mathbf 1 = \tilde b} \langle C, P\rangle.$$
Jeder Kunde bekommt $1/R$ seiner Nachfrage (Randbedingung, exakt).

**Teilbedienung (lineares Programm mit Schlupf).**
$$\min_{P,\,s,\,t \,\ge\, 0} \langle C, P\rangle + \rho \Big(\sum_i s_i + \sum_j t_j\Big) \quad \text{u. d. N.} \quad P\mathbf 1 + s = a, \qquad P^{\top}\mathbf 1 + t = b.$$

**Hilfsknoten.** Mit einem Schein-Kunden $Z$ (Nachfrage $\sum a$, Weg $\rho$ von jedem Lager) und einem Schein-Lager $S$ (Bestand $\sum b$, Weg $\rho$ zu jedem Kunden, Weg 0 nach $Z$) ist die Aufgabe ein gewöhnliches Transportproblem mit den Summen $\sum a + \sum b$.

**Kappung.** Bei gleichen Gesamtmassen ist das Modell mit Schlupf das gewöhnliche Transportproblem mit den gekappten Wegen $\min(c_{ij}, 2\rho)$.

**Teiltransport-Kurve und Umschlag.** $g$ ist konvex und stückweise linear. Die Zielfunktion $g(m) + \rho(\sum a + \sum b - 2m)$ hat ihr Minimum dort, wo die Steigung von $g$ gleich $2\rho$ ist; der ganze Bestand wird bewegt, sobald
$$\rho \;\ge\; \rho_{\text{voll}} = \tfrac12\, g'(m_{\max}^-),$$
der halben Steigung der letzten Strecke ($m_{\max} = \min(\sum a, \sum b)$).

**Handbeispiel 1** (Lager bei 0 und 10 mit Bestand 2 und 2, Kunden bei 1, 5, 9 mit Nachfrage je 2, Wege $(1, 5, 9)$ und $(9, 5, 1)$). *Normalisieren:* die Nachfrage wird $\tfrac43$ je Kunde, der Plan ist $\begin{pmatrix} 4/3 & 2/3 & 0 \\ 0 & 2/3 & 4/3 \end{pmatrix}$,
die Bewegungskosten sind $\tfrac{28}{3}$, jeder Kunde bekommt $\tfrac23$ seiner Nachfrage. *Teilbedienung:* für $\rho = \tfrac14$ (und 0,49) wird nichts bewegt, der Zielwert ist $10\rho = \tfrac52$ (4,9); für $\rho = 0{,}51$ wird alles bewegt, $4 + 2\rho = 5{,}02$;
für $\rho = 1$ ist der Zielwert $6$, für $\rho = 3$ ist er $10 = 4 + 6$. Der Plan ist beide Male eindeutig: Lager 1 liefert 2 an Kunde 1, Lager 2 liefert 2 an Kunde 3, **Kunde 2 bleibt leer**, die Bewegungskosten sind 4. Der Umschlag liegt bei
$\rho = \tfrac12$ = (Grenzkosten der letzten Einheit, 1) / 2. Die Teilbedienung kostet $\tfrac47 = 57{,}1\,\%$ weniger Bewegungskosten als das Normalisieren (Basis Normalisieren); mit der Strafe $\rho = 1$ auf die in beiden Plänen gleichen 2 nicht gelieferten Einheiten stehen $\tfrac{34}{3}$ gegen $6$.

**Handbeispiel 2** (Quellen bei 0 und 5, Ziele bei 4 und 20): ausgeglichen kostet die geordnete Zuordnung $4 + 15 = 19$, die gekreuzte $20 + 1 = 21$; auf den Ausreißer entfallen $\tfrac{15}{19} = 79\,\%$ bei der Hälfte der Masse. Der Zielwert des Modells mit Strafe ist die Kappungsformel
$\min\big(\sum \min(c, 2\rho)\big)$ über beide Zuordnungen: bei $\rho = 8$ ist er 17 (Ausreißer nicht bedient, obwohl $2\rho = 16 > 15$), bei $\rho = 10$ ist er 19, bei $\rho = 9$ sind beide Pläne optimal (19). Der Umschlag $2\rho = 15 + 3$ liegt über $d_{\min}/2 = 7{,}5$.

**Handbeispiel 3** (KL-Strafe, 2 × 2, alle Kosten 0, $a = (\tfrac12, \tfrac12)$, $b = (0{,}65,\, 0{,}65)$, $\varepsilon = 1$): die Masse ist $4 \cdot (0{,}5 \cdot 0{,}65/4)^{\tau/(2\tau+\varepsilon)}$, das Skalierungsverfahren gibt sie auf $10^{-9}$ genau.
Für $\tau \to 0$ ist sie 4 (Summe der Kernwerte), für $\tau \to \infty$ gleich $\sqrt{1{,}3} = 1{,}1402$; über beiden Randsummen bleibt sie bis $\tau/\varepsilon \approx 4{,}3$ ($1{,}311$ bei 4, $1{,}278$ bei 5), bei $\tau/\varepsilon = 100$ ist sie $1{,}147$, über dem Bestand 1.
Das Modell mit Schlupf bewegt bei Kosten 0 den ganzen Bestand 1 und lässt 0,3 offen: Zielwert $0{,}3\,\rho$.

**Handbeispiel 4** (Quellen bei 0 und 1, Ziele bei 2 und 3): beide Zuordnungen kosten 4; der Anteil der Wege zum Ausreißer ist $\tfrac12$ bei der sortierten, $\tfrac34$ bei der gekreuzten Zuordnung, $\tfrac58$ bei ihrer Mischung zu gleichen Teilen; der Anteil am Integral
$\int |F - G|\,dx = 2$ oberhalb der Schnittstelle 2 ist $\tfrac14$.

**Zählung.** Ein Durchlauf des Skalierungsverfahrens sind zwei Matrix-Vektor-Produkte ($Kv$ und $K^{\top}u$). Abbruch bei relativer Änderung von $u$ und $v$ unter $10^{-9}$ (float64).

**Literatur** (nur so zitiert, wie gelesen): Rubner, Tomasi, Guibas (2000), *The Earth Mover's Distance as a metric for image retrieval*, Int. J. Computer Vision 40(2), 99–121 (Abschnitt 4: Teiltransport mit $\sum f_{ij} = \min(\sum w_p, \sum w_q)$,
normiert durch den Gesamtfluss; Anhang A: nur bei gleichem Gesamtgewicht eine Metrik); Chizat, Peyré, Schmitzer, Vialard (2018), *Scaling algorithms for unbalanced optimal transport problems*, Math. Comp. 87, arXiv 1607.05816 (nur der Abstract gelesen)
und *Unbalanced optimal transport: dynamic and Kantorovich formulations*, J. Funct. Anal. 274, arXiv 1508.05216 (Abstract und Einleitung gelesen); Nietert, Goldfeld, Cummings (2022), *Outlier-robust optimal transport: duality, structure, and statistical analysis*, AISTATS, arXiv 2111.01361 (nur der Abstract gelesen).
Das Hilfsknoten-Verfahren und das Verfahren der kürzesten erweiternden Wege sind Lehrbuchstoff des Operations Research.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)

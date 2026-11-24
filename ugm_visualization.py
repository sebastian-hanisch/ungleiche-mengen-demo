"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Beschriftungen sind Annotationen (kein Text-Trace, keine texttemplate); alle Annotationen eines Diagramms werden auf einmal gesetzt."""

import numpy as np
import plotly.graph_objects as go

import ugm_constants as C
import ugm_teiltransport as T


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def _titel(fig, titel, oben=45):
    if titel:
        fig.update_layout(title=dict(text=titel, font=dict(size=13)), margin=dict(l=10, r=10, t=oben, b=10))
    return fig


def build_strasse(strasse, geliefert, titel="", farbe=C.COLOR_TEIL, hoehe=330):
    """Die Straße: Lager als Dreiecke unter der Achse (Größe = Bestand, Beschriftung mit dem Bestand), Kunden als Balken (grau: Nachfrage, farbig: geliefert)."""
    x = np.asarray(strasse.x_kunden, float)
    nach = np.asarray(strasse.nachfrage, float)
    geliefert = np.asarray(geliefert, float)
    m = len(x)
    L = strasse.laenge
    breite = 0.6 * L / max(m, 12)
    ymax = float(nach.max()) if m else 1.0
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=nach, width=breite, name="Nachfrage", marker=dict(color=C.COLOR_GRAU), hovertemplate="Nachfrage %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=geliefert, width=breite, name="geliefert", marker=dict(color=farbe), hovertemplate="geliefert %{y:.2f}<extra></extra>"))
    bestand = np.asarray(strasse.bestand, float)
    gross = float(bestand.max()) if len(bestand) else 1.0
    fig.add_trace(go.Scatter(x=strasse.x_lager, y=[-0.12 * ymax] * len(bestand), mode="markers", name="Lager",
                             marker=dict(symbol="triangle-up", size=[10 + 14 * b / gross for b in bestand], color=C.COLOR_LAGER), hovertemplate="Lager, Bestand %{customdata:.2f}<extra></extra>",
                             customdata=bestand))
    beschriftungen = [dict(x=xl, y=-0.12 * ymax, text=f"L{i + 1}: {b:g}".replace(".", ","), showarrow=False, yshift=-22, font=dict(size=10))
                      for i, (xl, b) in enumerate(zip(strasse.x_lager, bestand))]
    fig.update_layout(barmode="overlay", annotations=beschriftungen)
    fig.update_xaxes(range=[-0.03 * L, 1.03 * L], title_text="Ort auf der Straße", tickfont=dict(size=9))
    schritt = max(1, int(np.ceil(ymax / 4.0)))
    fig.update_yaxes(range=[-0.32 * ymax, 1.12 * ymax], title_text="Einheiten", tickfont=dict(size=9), tickmode="array", tickvals=list(range(0, int(ymax) + 1, schritt)))
    return _titel(_base(fig, hoehe), titel)


def build_ersparnis(r: dict, hoehe=300):
    """Verteilung der Ersparnis (Teilbedienung gegen Normalisieren, Basis Normalisieren) über die Instanzen der Reihe, mit dem Mittel als Linie."""
    werte = [100.0 * w for w in r["ersparnis_liste"]]
    mittel = 100.0 * r["ersparnis"][0]
    fig = go.Figure(go.Histogram(x=werte, xbins=dict(start=0, end=100, size=2.5), marker=dict(color=C.COLOR_TEIL), hovertemplate="Ersparnis %{x} %: %{y} Instanzen<extra></extra>"))
    fig.add_vline(x=mittel, line=dict(color=C.COLOR_NORM, dash="dash"))
    fig.update_layout(annotations=[dict(x=mittel, y=1.0, yref="paper", text=f"Mittel {mittel:.1f} %".replace(".", ","), showarrow=False, xanchor="left", xshift=4, yanchor="top")])
    fig.update_xaxes(range=[-2, 102], title_text="Ersparnis der Bewegungskosten in % (Basis Normalisieren)", tickfont=dict(size=9))
    fig.update_yaxes(title_text=f"Instanzen (von {r['anzahl']})", tickfont=dict(size=9))
    return _base(fig, hoehe, legend=False)


def build_kurve(kurve, rho, laenge=1.0, hoehe=340):
    """Teiltransport-Kurve: die kleinsten Bewegungskosten für genau m bewegte Einheiten, mit der Geraden der Steigung 2ρ durch den Punkt, den die Strafe ρ wählt."""
    mengen, kosten = T.stuetzstellen(kurve)
    mengen_l, kosten_l = mengen, kosten
    m_opt = T.bewegt_bei(kurve, rho)
    g_opt = T.kosten_bei_menge(kurve, min(m_opt, kurve.bewegt_gesamt))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mengen_l, y=kosten_l, mode="lines", name="Kosten für genau m Einheiten", line=dict(color=C.COLOR_NORM, width=2.5),
                             hovertemplate="m = %{x:.2f}: Kosten %{y:.3f}<extra></extra>"))
    if mengen[-1] > 0:
        xs = np.array([0.0, mengen[-1]])
        ys = g_opt + 2.0 * rho * (xs - m_opt)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Gerade der Steigung 2ρ (ρ = {rho / laenge:.3g})".replace(".", ","), line=dict(color=C.COLOR_TEIL, dash="dash"),
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[m_opt], y=[g_opt], mode="markers", name="gewählt von ρ", marker=dict(color=C.COLOR_TEIL, size=12, symbol="diamond"),
                             hovertemplate="bewegt %{x:.2f}, Kosten %{y:.3f}<extra></extra>"))
    ymax = float(kosten[-1]) if kosten[-1] > 0 else 1.0
    fig.update_xaxes(range=[0, max(mengen[-1], 1e-9) * 1.03], title_text="bewegte Einheiten m", tickfont=dict(size=9))
    fig.update_yaxes(range=[-0.05 * ymax, 1.1 * ymax], title_text="Bewegungskosten", tickfont=dict(size=9))
    return _base(fig, hoehe)


def build_bewegt_rho(r: dict, rho_marke=None, hoehe=340):
    """Bewegter Anteil des Bestands gegen ρ (Mittel und Spanne über die Instanzen der Reihe), die Stufen der Tabelle als Punkte."""
    x = r["raster"]
    mittel = [100.0 * s[0] for s in r["raster_bewegt"]]
    lo = [100.0 * s[1] for s in r["raster_bewegt"]]
    hi = [100.0 * s[2] for s in r["raster_bewegt"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=hi + lo[::-1], fill="toself", fillcolor=C.COLOR_BAND, line=dict(width=0), name="Spanne der Instanzen", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mittel, mode="lines", name="Mittel", line=dict(color=C.COLOR_NORM, width=2.5), hovertemplate="ρ = %{x}: %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=r["rho"], y=[100.0 * s[0] for s in r["bewegt"]], mode="markers", name="Stufen der Tabelle", marker=dict(color=C.COLOR_TEIL, size=8), hoverinfo="skip"))
    anno = []
    if rho_marke is not None:
        fig.add_vline(x=rho_marke, line=dict(color=C.COLOR_LAGER, dash="dot"))
        anno.append(dict(x=rho_marke, y=0.04, yref="paper", text="ρ der Seitenleiste", showarrow=False, xanchor="left", xshift=4))
    rv = r["rho_voll"][0]
    fig.add_vline(x=rv, line=dict(color=C.COLOR_TEIL, dash="dash"))
    anno.append(dict(x=rv, y=0.5, yref="paper", text=f"Mittel ρ_voll = {rv:.3f}".replace(".", ","), showarrow=False, xanchor="left", xshift=4))
    fig.update_layout(annotations=anno)
    fig.update_xaxes(range=[0, x[-1]], title_text="Strafe ρ in Straßenlängen", tickfont=dict(size=9))
    fig.update_yaxes(range=[-2, 102], title_text="bewegter Anteil des Bestands in %", tickfont=dict(size=9))
    return _base(fig, hoehe)


def build_ausreisser(r: dict, rho=None, hoehe=340):
    """Anteil der bedienten Ausreißer gegen ρ (Mittel und Spanne über die Instanzen), mit den Marken d_min/2, ρ_erste und ρ_alle (Mittel)."""
    x = r["raster"]
    mittel = [100.0 * s[0] for s in r["raster_bedient"]]
    lo = [100.0 * s[1] for s in r["raster_bedient"]]
    hi = [100.0 * s[2] for s in r["raster_bedient"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=hi + lo[::-1], fill="toself", fillcolor=C.COLOR_BAND, line=dict(width=0), name="Spanne der Instanzen", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mittel, mode="lines", name="Mittel", line=dict(color=C.COLOR_NORM, width=2.5), hovertemplate="ρ = %{x}: %{y:.1f} %<extra></extra>"))
    anno = []
    for wert, text, farbe, hoch in ((r["d_min2"][0], "d_min/2", C.COLOR_GRAU, 0.92), (r["rho_erste"][0], "ρ_erste", C.COLOR_TEIL, 0.70), (r["rho_alle"][0], "ρ_alle", C.COLOR_KUNDE, 0.48)):
        fig.add_vline(x=wert, line=dict(color=farbe, dash="dash"))
        anno.append(dict(x=wert, y=hoch, yref="paper", text=f"{text} = {wert:.2f}".replace(".", ","), showarrow=False, xanchor="right", xshift=-4))
    if rho is not None:
        fig.add_vline(x=rho, line=dict(color=C.COLOR_LAGER, dash="dot"))
        anno.append(dict(x=rho, y=0.04, yref="paper", text="ρ der Seitenleiste", showarrow=False, xanchor="left", xshift=4))
    fig.update_layout(annotations=anno)
    fig.update_xaxes(range=[0, x[-1]], title_text="Strafe ρ in Standardabweichungen", tickfont=dict(size=9))
    fig.update_yaxes(range=[-2, 102], title_text="bediente Ausreißer in %", tickfont=dict(size=9))
    return _base(fig, hoehe)


def build_gesamtmasse(w: dict, hoehe=340):
    """Gesamtmasse der KL-Strafe gegen τ/ε: geschlossene Form des 2 × 2-Beispiels (Linie) und die 30 × 30-Instanzen (Mittel mit Spanne); Randsummen 1 und 1,3 als Linien."""
    from ugm_kl import masse_zwei_mal_zwei

    xs = np.logspace(-1.3, 2.3, 120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[masse_zwei_mal_zwei(t) for t in xs], mode="lines", name="2 × 2, Kosten 0 (geschlossene Form)", line=dict(color=C.COLOR_NORM, width=2.5),
                             hovertemplate="τ/ε = %{x:.3g}: Masse %{y:.3f}<extra></extra>"))
    mit = w["masse"]
    fig.add_trace(go.Scatter(x=w["tau_eps"], y=[s[0] for s in mit], mode="markers", name=f"{w['n']} × {w['n']}, Mittel mit Spanne ({w['anzahl']} Instanzen)",
                             marker=dict(color=C.COLOR_TEIL, size=10, symbol="diamond"),
                             error_y=dict(type="data", symmetric=False, array=[s[2] - s[0] for s in mit], arrayminus=[s[0] - s[1] for s in mit], color=C.COLOR_TEIL),
                             hovertemplate="τ/ε = %{x}: Mittel %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_GRAU, dash="dash"))
    fig.add_hline(y=C.UEBERHANG_WEICH, line=dict(color=C.COLOR_GRAU, dash="dash"))
    fig.update_layout(annotations=[dict(x=1, xref="paper", y=1.0, text="Bestand 1", showarrow=False, xanchor="right", yanchor="top"),
                                   dict(x=1, xref="paper", y=C.UEBERHANG_WEICH, text="Nachfrage 1,3", showarrow=False, xanchor="right", yanchor="bottom")])
    fig.update_xaxes(type="log", range=[-1.3, 2.3], title_text="τ/ε (Gewicht der Randsummen-Strafe zu ε)", tickfont=dict(size=9))
    ticks = [1.0, 1.3, 2.0, 3.0, 5.0, 10.0, 15.0]
    fig.update_yaxes(type="log", range=[np.log10(0.9), np.log10(20.0)], title_text="Gesamtmasse des Plans (logarithmisch)", tickfont=dict(size=9), tickmode="array", tickvals=ticks,
                     ticktext=[str(t).replace(".", ",").removesuffix(",0") for t in ticks])
    return _base(fig, hoehe)

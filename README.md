# Angebot ungleich Nachfrage: Normalisieren oder Teilbedienung – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-ungleiche-mengen-demo.streamlit.app/)** -->

Demo der Reihe **optimaler-transport** („Optimaler Transport: Verteilungen ineinander verschieben“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie setzt an der Annahme an, die die Wurzel der Reihe („Vom Zuordnen zum Transportieren“) stehen lässt: *Angebot und Nachfrage sind in der Summe gleich.* Diese Demo hebt sie auf: Bestand und Nachfrage dürfen verschieden sein, und die Rechnung muss sagen, was mit dem Rest geschieht (Teilbedienung, Strafe, Ausreißer). Vehikel ist eine Straße mit drei Lagern und 30 Kunden (Kosten = Weg).

## Warum dieses Problem

Bestand und Nachfrage haben in der Praxis selten dieselbe Summe, das gewöhnliche **Transportproblem** (englisch transportation problem) verlangt es aber. Eine stille Antwort ist das **Normalisieren** (englisch normalization): beide Seiten werden vor der Rechnung auf dieselbe Summe gebracht, und jeder Kunde bekommt denselben Anteil seiner Nachfrage. Die Alternative ist die **Teilbedienung** (englisch partial transport, auch unausgeglichener Transport, englisch unbalanced transport): Bestand und Nachfrage bleiben verschieden, eine **Strafe ρ** je nicht bewegte und je nicht gelieferte Einheit sagt, was mit dem Rest geschieht. Die Demo prüft eine Annahme: *Beide Seiten auf Summe 1 zu normieren ist eine Formalie.* Bei gleicher Summe stimmt das (Differenz 0 ab ρ_voll), bei ungleicher legt es fest, wer wenig bekommt. Die Gegenstimme steht ehrlich daneben: die Teilbedienung braucht einen Parameter ρ, den das Normalisieren nicht hat, und sein Wert steht in keinen Daten. Wo beide Verteilungen Wahrscheinlichkeiten sind (Summe 1 per Definition), ist Normalisieren richtig; die Frage nach der Normierung entsteht nur bei Mengen, und Masse zu verwerfen (Ausreißer, Experiment 3) kann auch bei Wahrscheinlichkeiten sinnvoll sein. Mathematisch ist nichts neu (Hilfsknoten, partieller und unausgeglichener Transport); die Demo zählt, was das Normalisieren mit den Kunden macht, und macht die stille Annahme sichtbar.

## Modell

Gegeben sind Bestände $a$ (3 Lager), Nachfragen $b$ ($m$ Kunden), Wege $c_{ij}$ und die Strafe $\rho$; gesucht ist der Plan $P$, der Bewegungskosten plus $\rho\,(\text{nicht bewegter Bestand} + \text{nicht gelieferte Nachfrage})$ minimiert. Die Formeln, die Handrechnungen in Brüchen und die Lesehilfe stehen im 📐-Expander der App; hier nur das Nötige:

- **Straße (Objekt Ort, eine Dimension):** Länge 1, Lager bei ¼, ½, ¾ mit Beständen 40, 30, 30 (Summe 100); die Kunden gleichverteilt auf der Straße, Nachfrage ganzzahlig 1 bis 8, auf $R \cdot 100$ skaliert ($R$ = Nachfrage / Bestand, Standard 1,3). Kosten = Weg $|x - y|$, höchstens 1. Experiment 3: standardnormale Punkte, Masse 1 je Punkt, 5 von 100 Zielen als Ausreißer.
- **Strafe ρ immer relativ:** in Straßenlängen (Experimente 1 und 2) beziehungsweise Standardabweichungen (Experiment 3) beziehungsweise ρ/OPT beim Vergleich mit der KL-Strafe; ρ ist die Strafe je vernichtete **und** je erzeugte Einheit, eine bewegte Einheit spart also 2ρ (Kappung bei 2ρ).
- **Zählweise:** Instanzen (Zähleinheit der Reihen), Kunden (voll / leer / teilweise; „leer“ heißt höchstens 1e-7 Einheiten geliefert), Einheiten Masse, Bewegungskosten in Einheiten mal Weg ohne Strafe, daneben der Zielwert mit Strafe; für die KL-Strafe Matrix-Vektor-Produkte (ein Durchlauf = zwei).
- **Zufall:** Hausgenerator SplitMix64 (`ugm_rng.py`, mit Ganzzahlen und `decimal` statt `libm` so gebaut, dass er auf Windows und Linux bitgleich ist; geprüft unter Windows); die Reihen benutzen feste Instanzen mit dem Zufalls-Seed 2026, unabhängig vom Regler „Zufalls-Seed“. Gewichte Dirichlet(1) sind unabhängige Exponentialzahlen, durch ihre Summe geteilt.

## Methodik

- **Normalisieren:** die Nachfrage mit $\sum a / \sum b$ multiplizieren, dann das gewöhnliche Transportproblem mit dem Transportsimplex (`ugm_simplex.py`, Kopie aus der Demo „Vom Zuordnen zum Transportieren“; ein Test vergleicht die Kopie mit der Quelle).
- **Teilbedienung:** über **Hilfsknoten** (englisch dummy nodes): ein Schein-Kunde mit der Nachfrage $\sum a$ (Weg ρ von jedem Lager), ein Schein-Lager mit dem Bestand $\sum b$ (Weg ρ zu jedem Kunden, Weg 0 zum Schein-Kunden); das ist ein ausgeglichenes Transportproblem, das derselbe Simplex löst (`ugm_ungleich.py`).
- **Teiltransport-Kurve** (`ugm_teiltransport.py`): kürzeste erweiternde Wege (englisch successive shortest paths) bewegen zuerst die billigste Einheit, dann die nächste unter Umlenken früherer Lieferungen. Die Kosten für genau $m$ bewegte Einheiten sind konvex und stückweise linear; aus **einer** Kurve je Instanz folgen die bewegte Menge und die Lieferungen für **jedes** ρ, und der Umschlag $\rho_{\text{voll}}$ ist die Hälfte der Steigung der letzten Strecke. Der Simplex der Teilbedienung und diese Kurve sind zwei Rechenwege im Haus; sie müssen im Zielwert übereinstimmen (Test).
- **KL-Strafe** (`ugm_kl.py`): das Skalierungsverfahren für weiche Randsummen nach Chizat, Peyré, Schmitzer, Vialard (2018), selbst nachgebaut (von der Arbeit nur der Abstract gelesen; eine Probe auf die Stationaritätsbedingung der Zielfunktion steht in den Tests) und nicht gegen eine Bibliothek geprüft: $P = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)$ mit dem nicht normierten Kern $K = e^{-C/\varepsilon}$, $u \leftarrow (a/(Kv))^\lambda$, $v \leftarrow (b/(K^{\top}u))^\lambda$, $\lambda = \tau/(\tau+\varepsilon)$; Abbruch bei relativer Änderung von $u$ und $v$ unter 1e-9 (float64).
- **Vorrechnung:** die Ausreißer-Reihe der Standardeinstellungen liegt in `tools/vorgerechnet.json` (erzeugt von `tools/vorrechnen.py`), damit die Seite sofort erscheint; jede andere Einstellung rechnet live. `tools/messreihen.py` rechnet die Reihen dieser README und schreibt `tools/messreihen.json`; `tests/test_claims.py` bildet die README-Zahlen aus dieser Datei und rechnet die Reihen neu.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus `tools/messreihen.json` (Zufalls-Seed 2026); sie gelten **in diesem Modell, bei diesen Parametern** (Straße mit drei Lagern, Bestand 100, 30 Kunden, wo nichts anderes steht, 200 Instanzen je Reihe, Strafe je vernichtete und je erzeugte Einheit) und sind keine Gesetze. Minimum und Maximum sind Extremwerte, die von Stichprobe zu Stichprobe springen; als Bandbreite taugen sie nicht, sie stehen nur zur Orientierung.

### Handbeispiel (von Hand gerechnet, exakt)

Lager bei 0 und 10 (Bestand 2 und 2), Kunden bei 1, 5, 9 (Nachfrage je 2). **Normalisieren:** jeder Kunde bekommt $\tfrac23$ seiner Nachfrage, die Bewegungskosten sind $\tfrac{28}{3}$. **Teilbedienung:** bei ρ = ¼ und ρ = 0,49 wird nichts bewegt (Zielwert $10\rho$); bei ρ = 0,51 wird alles bewegt (5,02); bei ρ = 1 ist der Zielwert 6 und der Plan eindeutig: Lager 1 liefert 2 an Kunde 1, Lager 2 liefert 2 an Kunde 3, Kunde 2 bleibt leer, die Bewegungskosten sind 4. Der Umschlag liegt bei ρ = ½, der halben Grenzkosten der letzten Einheit. Die Teilbedienung kostet $\tfrac47 = 57{,}1\,\%$ weniger Bewegungskosten als das Normalisieren (Basis Normalisieren). Ausreißer-Kleinbeispiel (Quellen bei 0 und 5, Ziele bei 4 und 20): der kürzeste Weg zum Ausreißer ist 15, die Untergrenze 7,5; der Ausreißer wird aber erst ab ρ = 9 bedient, weil wer ihn bedient die Quelle bei 5 vom Ziel bei 4 wegschickt (Zusatzweg 3, $2\rho = 15 + 3$). Quellen bei 0 und 1, Ziele bei 2 und 3 (Ausreißer bei 3): beide Zuordnungen kosten 4, der Ausreißer macht ½ (sortierte Zuordnung), ¾ (gekreuzte) oder, als Anteil des Integrals oberhalb der Schnittstelle, ¼ aus. Alle diese Zahlen stehen als Brüche in `tests/test_handbeispiele.py`.

### Experiment 1: Normalisieren oder Teilbedienung

In der Standardstraße der App (Zufalls-Seed 35, 30 Kunden, R = 1,3, ρ = 1 Straßenlängen, eine Instanz) kostet das Normalisieren 8,75 Bewegungskosten, die Teilbedienung 6,73 (23,1 % weniger, Basis Normalisieren); bei der Teilbedienung bekommen 21 Kunden alles, 6 nichts und 3 einen Teil, und beide Pläne lassen 30 Einheiten ungeliefert.

Reihe über 200 Zufallsstraßen (R wie angegeben, 30 Kunden, ρ = 1 Straßenlänge); Ersparnis = 1 − Bewegungskosten Teilbedienung / Bewegungskosten Normalisieren (Basis Normalisieren), Mittel [Minimum; Maximum]; Spalten: Ersparnis, Kunden leer (Mittel), Instanzen mit mindestens einem leeren Kunden, Bewegungskosten Normalisieren / Teilbedienung (Mittel).

| R | Ersparnis | Kunden leer (Mittel) | Instanzen mit leerem Kunden | Bewegungskosten Normalisieren / Teilbedienung (Mittel) |
|---|---|---|---|---|
| 1 | 0,0 % [0,0; 0,0] | 0,0 % | 0 von 200 | 11,60 / 11,60 |
| 1,15 | 22,2 % [9,0; 39,0] | 10,7 % | 200 von 200 | 11,60 / 9,00 |
| 1,3 | 33,1 % [15,3; 53,7] | 19,6 % | 200 von 200 | 11,60 / 7,71 |
| 1,6 | 45,0 % [24,7; 66,3] | 32,4 % | 200 von 200 | 11,60 / 6,30 |
| 2 | 54,2 % [31,3; 79,4] | 44,9 % | 200 von 200 | 11,60 / 5,24 |

R = 1,3, ρ = 1 Straßenlänge, 30 Kunden: Die Teilbedienung kostet im Mittel 33,1 % weniger Bewegungskosten als das Normalisieren (einzelne Instanzen 15,3 bis 53,7 %, Mittel über 200 Instanzen, Basis Normalisieren). Im Mittel bleiben 19,6 % der Kunden leer, in 200 von 200 Instanzen dieser Reihe mindestens einer. In allen 200 Instanzen bewegt die Teilbedienung den ganzen Bestand (ρ ≥ ρ_voll); dann lassen beide Pläne dieselben Einheiten ungeliefert (die Differenz von Nachfrage und Bestand): die Normierung ist nicht falsch, sie legt fest, wer wenig bekommt. Teurer als das Normalisieren war die Teilbedienung in 0 Instanzen.

R = 1 und ρ ≥ ρ_voll: beide Rechnungen sind dasselbe Problem; die Ersparnis ist 0 (Abweichung unter 1e-9, Rundung), in 0 von 200 Instanzen ist ein Kunde leer. Die Normierung ist nur bei ungleicher Summe eine Annahme.

Teurer als das Normalisieren war die Teilbedienung in 0 von 1000 Instanzen (fünf Reihen mit R = 1 bis 2, ρ = 1 Straßenlängen). Das ist kein Zufall der Stichprobe, sondern ein Satz (für ρ ab ρ_voll): der normalisierte Plan bewegt den ganzen Bestand, und die Teilbedienung wählt unter allen Plänen, die den ganzen Bestand bewegen, den billigsten (Eigenschaftstest). Für jedes ρ ≥ 0 gilt bei R ≥ 1 außerdem die Abschätzung Bewegungskosten (Teilbedienung) ≤ Bewegungskosten (Normalisieren) − 2ρ · (Bestand − bewegte Menge), weil der normalisierte Plan auch für das Modell mit Schlupf zulässig ist (Eigenschaftstest); ein Vergleich auf gleicher Grundlage ist es erst ab ρ_voll. Bei ρ = 0,1 Straßenlängen (R = 1,3) bewegt die Teilbedienung in 103 von 200 Instanzen nicht den ganzen Bestand (in diesen Instanzen im Mittel 92,4 %, über alle 200 Instanzen im Mittel 96,1 % des Bestands bewegt; bei 5 anderen Zufalls-Seeds der Reihe 100 bis 118 von 200); die Ersparnis von im Mittel 40,3 % gilt dann für weniger bewegte Einheiten und ist kein Vergleich auf gleicher Grundlage. Mit 60 statt 30 Kunden (R = 1,3, ρ = 1) ist die Ersparnis im Mittel 33,6 % [16,8; 55,7], in 200 von 200 Instanzen ist mindestens ein Kunde leer.

### Experiment 2: Die Strafe ρ

R = 1,3, 30 Kunden, 200 Instanzen; die Spalten sind ρ in Straßenlängen, der bewegte Anteil des Bestands (Mittel [Minimum; Maximum]) und der Anteil leerer Kunden (Mittel).

| ρ (Straßenlängen) | bewegter Anteil des Bestands | Kunden leer |
|---|---|---|
| 0,01 | 15,8 % [0,0; 41,2] | 87,8 % |
| 0,025 | 40,3 % [9,5; 69,1] | 69,0 % |
| 0,05 | 72,6 % [43,1; 96,8] | 42,8 % |
| 0,075 | 89,9 % [55,4; 100,0] | 28,3 % |
| 0,1 | 96,2 % [72,3; 100,0] | 23,1 % |
| 0,15 | 99,5 % [83,6; 100,0] | 20,5 % |
| 0,25 | 100,0 % [98,3; 100,0] | 20,1 % |

Mittel über 200 Instanzen, R = 1,3, 30 Kunden: ρ = 0,05: im Mittel 72,6 % des Bestands bewegt (42,8 % der Kunden leer); ρ = 0,1: im Mittel 96,2 % des Bestands bewegt (23,1 % der Kunden leer). Der Umschlag ρ_voll (der ganze Bestand wird bewegt) liegt im Mittel bei 0,109 Straßenlängen (einzelne Instanzen 0,051 bis 0,261) und ist gleich der halben Grenzkosten der letzten bewegten Einheit (Mittel der Grenzkosten 0,218). Die Strafe selbst steht in keinen Daten, sie muss gesetzt werden. Die 96,2 % bei ρ = 0,1 stehen neben den 96,1 % in Experiment 1: beide Reihen benutzen verschiedene Instanzen (getrennte Zufallsströme je Experiment), der Unterschied ist Stichprobenrauschen.

Der Umschlag ist **exakt** die halbe Steigung der letzten Strecke der Teiltransport-Kurve. Zwei Gegenproben stehen in den Tests: eine Bisektion über ρ mit dem Modell mit Strafe (25 Straßen, Abweichung höchstens 2e-8) und der Differenzenquotient der Kurve über 0,1 Einheiten, der nur bis 1e-2 übereinstimmt, weil er an dieser Auflösung hängt.

### Experiment 3: Ausreißer

100 Quellen und 100 Ziele, standardnormal, Masse 1 je Punkt; 5 Ziele bei 10 ± 0,1; 200 Instanzen; gleiche Gesamtmassen, Wege |x − y|, ρ in Standardabweichungen.

| Größe | Mittel [Minimum; Maximum] |
|---|---|
| erster Ausreißer wird bedient ab ρ_erste | 4,14 [3,01; 5,82] |
| alle Ausreißer werden bedient ab ρ_alle | 5,17 [4,12; 6,75] |
| Untergrenze d_min/2 (halber kürzester Weg zu einem Ausreißer) | 3,71 [3,01; 4,16] |
| Untergrenze d_max/2 für ρ_alle (halber größter der kürzesten Wege je Ausreißer) | 3,82 [3,10; 4,26] |

5 von 100 Zielen (5 % der Masse), 10 Standardabweichungen entfernt: Sie machen im Mittel 68,8 % der Wege der sortierten Zuordnung aus (einzelne Instanzen 48 bis 85 %). Eine andere optimale Zuordnung (die des Verfahrens der kürzesten Wege) gibt 74,7 %, der vom Plan unabhängige Anteil am Integral von |F − G| oberhalb der Schnittstelle 63,2 %: bei Gleichstand ist der Anteil keine Eigenschaft des Problems, sondern der Definition. Der erste Ausreißer wird im Mittel ab ρ = 4,14 bedient (einzelne Instanzen 3,01 bis 5,82), alle ab ρ = 5,17. Die Untergrenze d_min/2 (halber kürzester Weg) liegt im Mittel bei 3,71 und wird in 200 von 200 Instanzen nicht unterschritten; der Umschlag liegt im Mittel 0,43 Standardabweichungen darüber, was zum Kleinbeispiel passt: ein bedienter Ausreißer verdrängt eine Quelle von ihrem Nachbarn. Voraussetzung: gleiche Gesamtmassen, Wege |x − y|.

Die Zahl für „eine andere optimale Zuordnung“ ist eine Eigenschaft des Verfahrens der kürzesten Wege, das hier das Ende der Kurve liefert, keine des Problems; ein Test baut auf keiner dieser Zahlen, sondern auf der Definition (sortierte Zuordnung, Integral).

### Gesamtmasse bei der Strafe (feste Zahlen)

| τ/ε | 0,1 | 1 | 10 | 100 | τ/ε → ∞ |
|---|---|---|---|---|---|
| Masse im 2 × 2-Beispiel | 3,245 | 1,732 | 1,210 | 1,147 | 1,1402 |
| Masse, 30 × 30, Mittel [Minimum; Maximum] | 9,03 [5,09; 15,02] | 1,86 [1,27; 2,48] | 1,07 [0,99; 1,12] | 1,11 [1,11; 1,12] | – |
| Matrix-Vektor-Produkte (Median) | 12 | 34 | 203 | 1 938 | – |

| ρ/OPT (Modell mit Schlupf) | 0,1 | 0,3 | 1 | 3 |
|---|---|---|---|---|
| bewegte Masse, Mittel [Minimum; Maximum] | 0,37 [0,13; 0,54] | 0,64 [0,50; 0,74] | 0,93 [0,86; 1,00] | 1,00 [1,00; 1,00] |
| leere Lager (Mittel, größte Zahl) | 13,8 (21) | 5,7 (8) | 0,6 (2) | 0,0 (0) |

Σa = 1, Σb = 1,3: bei τ/ε = 1 hat der KL-Plan im Mittel 1,86 Masse (einzelne Instanzen 1,27 bis 2,48), in 9 von 10 Instanzen über beiden Randsummen; bei τ/ε = 10 im Mittel 1,07, 1 von 10 Instanzen unter dem Bestand 1; bei τ/ε = 100 im Mittel 1,11, in 10 von 10 Instanzen über dem Bestand (Gewichte Dirichlet(1), 30 × 30, je 10 Instanzen, Verfahren nach Chizat u. a. nachgebaut, nicht gegen eine Bibliothek geprüft). Das Modell mit Schlupf bewegt ab ρ = 3 OPT den ganzen Bestand (10 von 10) und lässt bei ρ = 0,1 OPT im Mittel 13,8 von 30 Lagern leer.

Die Lage der Masse des KL-Plans zu den beiden Randsummen hängt von τ/ε ab: bei schwacher Strafe liegt sie weit über beiden, bei starker Strafe im 2 × 2-Beispiel zwischen ihnen (ab τ/ε ≈ 4,3, Handbeispiel 3) und in den 30 × 30-Messungen bei τ/ε = 100 in 10 von 10 Instanzen über dem Bestand; gemessen ist das nur für Gewichte Dirichlet(1), 30 × 30, ε = 0,1 OPT und je 10 Instanzen, andere Gewichte sind nicht gemessen. Ursache der großen Masse bei schwacher Strafe ist der nicht normierte Kern $K = e^{-C/\varepsilon}$; für τ → 0 ist die Masse die Summe aller Kernwerte (Handbeispiel 3). Konvergenz und Zahl der Durchläufe des Skalierungsverfahrens sind nicht Thema dieser Demo; sie behandelt die Demo zum Matrix-Abgleichen (Furness, IPF) in derselben Reihe. ρ und τ sind verschiedene Parameter, eine Entsprechung wird nicht behauptet.

## Ehrliche Grenzen

- **Aussagen gelten in diesem Modell:** Straße (eine Dimension), Kosten = Weg, eine Strafe je vernichtete und je erzeugte Einheit, Zähleinheiten Kunden, Einheiten und Instanzen. Zwei Strafen (Fehlmenge und Lagerkosten getrennt), Bestand größer als Nachfrage und zweidimensionale Orte sind nicht untersucht.
- **Die Strafe ρ steht in keinen Daten.** Der Umschlag ρ_voll und die Untergrenze d/2 sind Eigenschaften der Instanz; welches ρ „richtig“ ist, ist eine Modellentscheidung. Die Teilbedienung hat darum einen Parameter mehr als das Normalisieren.
- **Normalisieren ist nicht falsch:** ab ρ_voll lassen beide Pläne dieselben Einheiten ungeliefert; die Normierung legt fest, wer wenig bekommt. Bei ρ unter ρ_voll bewegt die Teilbedienung weniger als den ganzen Bestand, und der Kostenvergleich mit dem Normalisieren ist dann kein Vergleich auf gleicher Grundlage.
- **Ausreißeranteil ohne Definition ist keine Zahl:** bei Gleichstand gibt es mehrere optimale Pläne, und der Anteil hängt an der Wahl (Handbeispiel: ½, ¾, ¼ bei gleichen Kosten). Die Demo nennt die Definition: Wege der sortierten Zuordnung.
- **Werkzeuge normieren still:** `scipy.stats.wasserstein_distance` gibt für Bestand (2, 2) bei 0 und 10 und Nachfrage (2, 2, 2) bei 1, 5, 9 den Wert 7/3 = (28/3)/4, auch bei den Gewichten (20, 20) gegen (1, 1, 1); es normiert die Gewichte stillschweigend (gemessen mit scipy 1.18.1; ein Test prüft das Verhalten). Die Dokumentation anderer Bibliotheken wurde nicht gelesen.
- **Die KL-Strafe ist selbst nachgebaut**, nur 10 Instanzen je Zelle, nur Gewichte Dirichlet(1), 30 × 30, nicht gegen eine Bibliothek geprüft.
- **Nicht mit echten Mausklicks im Browser durchgeklickt:** AppTest und Tests sehen die Plotly-Abbildungen nicht.
- **Literatur** (nur so zitiert, wie gelesen): Rubner, Tomasi, Guibas (2000): Teiltransport ist dort eingebaut (Abschnitt 4), die Earth Mover's Distance ist durch den Gesamtfluss normiert und nur bei gleichem Gesamtgewicht eine Metrik (Anhang A); Chizat, Peyré, Schmitzer, Vialard (2018), beide Arbeiten (*Scaling algorithms for unbalanced optimal transport problems*, Math. Comp. 87, arXiv 1607.05816: nur der Abstract; *Unbalanced optimal transport: dynamic and Kantorovich formulations*, J. Funct. Anal. 274, arXiv 1508.05216: Abstract und Einleitung); Nietert, Goldfeld, Cummings (2022), *Outlier-robust optimal transport: duality, structure, and statistical analysis*, AISTATS, arXiv 2111.01361: nur der Abstract. Das Hilfsknoten-Verfahren und die kürzesten erweiternden Wege sind Lehrbuchstoff des Operations Research. Einen Literaturtest mit Werten aus einer Quelle gibt es nicht.

**Abgrenzung zu den Demos im Portfolio.** `network-flow-demo` benutzt Schein-Knoten als Modellierungs-Kniff (Notbeschaffung) in einem Mindestkostenfluss; hier ist ρ ein Regler im Maßstab der Wege. `ssp-demo` zeigt die steigenden Grenzkosten im Graphen; hier ist die Kurve die Brücke zum Umschlag. `auction-algorithm-demo` hat eine Verzicht-Option mit so großer Strafe (M = min(n, m) · größte Fahrzeit + 1), dass erst die Zahl der Paare zählt und dann die Kosten; hier entscheidet das Verhältnis von Strafe und Weg. `lp-dualitaet-demo` zeigt Schattenpreise und Zukauf; ρ als Zukaufspreis ist verwandt, aber ein anderes Experiment.

## Verifikation

- **Kern:** Handbeispiele 1 bis 4 in Brüchen und in Zahlen (`tests/test_handbeispiele.py`; die Aufzählung aller Ecken über aufspannende Bäume in `tests/ecken_brueche.py` ist unabhängig von Simplex und HiGHS), Randfälle (leere Seiten, ein Lager und ein Kunde, ρ = 0, ρ sehr groß, Massen von 0,01 bis 100, gleich weite Kunden, ungültige Eingaben).
- **Orakel** (`tests/test_oracle_ungleiche_mengen.py`, ein anderer Rechenweg): das Modell mit Strafe gegen **`linprog` mit Schlupfvariablen** (40 Straßen, 5 Werte von ρ, relativ 1e-9), gegen die **Aufzählung aller Ecken in Brüchen** (30 kleine Aufgaben) und bei gleichen Gesamtmassen gegen die **Kappungsformel mit `linear_sum_assignment`** (20 Instanzen, 5 Werte von ρ); die Teiltransport-Kurve gegen das **Linearprogramm mit „bewegte Menge = m“** (25 Straßen, 6 Werte von m); ρ_voll gegen **Bisektion** und gegen den **Differenzenquotienten** (25 Straßen); ρ_erste und ρ_alle gegen Bisektion mit der Kappungsformel (12 Instanzen); die **Stationaritätsbedingung** der KL-Strafe in jeder Zelle (6 Instanzen, zwei Werte von τ/ε); `scipy.stats.wasserstein_distance` gegen das Normalisieren (10 Straßen); die Kundenmengen sind über alle optimalen Pläne bestimmt (6 Straßen, Spannweite unter 1e-5, an der Toleranz des Lösers).
- **Eigenschaften** (`tests/test_eigenschaften.py`, 15 bis 50 Instanzen): bei gleicher Summe ist die Teilbedienung gleich dem Normalisieren; ab ρ_voll nie höhere Bewegungskosten als das Normalisieren, und für jedes ρ die Abschätzung Bewegungskosten (Teilbedienung) ≤ Bewegungskosten (Normalisieren) − 2ρ · (Bestand − bewegte Menge); der Zielwert ist bei jedem ρ nicht größer; mehr ρ bewegt nie weniger; der Zielwert ist konkav in ρ und gleich Kurve plus Strafe; Kosten und ρ mit demselben Faktor; Umnummerieren; ein Kunde ohne Nachfrage; ab ρ_voll wird immer das kleinere von Bestand und Nachfrage bewegt; jeder Kunde bekommt beim Normalisieren den Anteil 1/R; ρ_erste ≥ d_min/2.
- **Verdikte und Zahlen:** Verdikte sind reine Funktionen (`ugm_evaluation.py`) und an handgesetzten Zeilen geprüft (`tests/test_evaluation.py`); jede Zahl dieser README steht in `tests/test_claims.py`.
- **Oberfläche:** AppTest (jeder Button, jeder Regler an Minimum und Maximum, die Optionen der Auswahlregler, jedes Preset mit seiner erwarteten Aussage, Permalink, Worst-Case am Regler-Maximum), Layout- und Footer-Test.
- **Kopien:** `ugm_simplex.py` ist eine Kopie der Wurzel-Demo, `ugm_rng.py` der Vorlage des Portfolios (Kopier-Tests).
- **Testzahl:** 196 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur für die Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, plotly |
| `tools/messreihen.py` | rechnet die Messreihen der README neu und schreibt `tools/messreihen.json` |
| `tools/vorrechnen.py` | rechnet die Ausreißer-Reihe der Standardeinstellungen vor und schreibt `tools/vorgerechnet.json` |
| `ugm_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets und Hilfetexte |
| `ugm_evaluation.py` | Kennzahlen, Reihen der drei Experimente, Absatz zur Gesamtmasse, Verdikte |
| `ugm_kl.py` | Skalierungsverfahren mit KL-Strafe (weiche Randsummen) |
| `ugm_layout.py` | Fester Kopf (Stylesheet) |
| `ugm_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `ugm_rng.py` | Hausgenerator SplitMix64 (plattformgleich) |
| `ugm_scenario.py` | Instanzen: Straße, Handbeispiel, Ausreißer, 30 × 30 |
| `ugm_simplex.py` | Transportsimplex mit Basisbaum (Kopie der Wurzel-Demo) |
| `ugm_teiltransport.py` | Teiltransport-Kurve durch kürzeste erweiternde Wege |
| `ugm_ungleich.py` | Normalisieren und Teilbedienung über Hilfsknoten |
| `ugm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `ugm_vorgerechnet.py` | liest die vorgerechnete Reihe |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Szenario und Kopien, Verdikte, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- Getrennte Strafen für Fehlmenge und Lagerkosten, Bestand größer als Nachfrage, zweidimensionale Orte: nicht gemessen, daher nicht gezeigt.
- Kein Geschwindigkeitsvergleich zwischen Simplex und Skalierungsverfahren (keine Aussage über „schneller“).
- Keine Bibliothek zur Laufzeit außer numpy und plotly (`scipy` nur in den Tests).
- Kein PDF-Export.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).

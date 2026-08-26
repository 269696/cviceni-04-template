# Cvičení 4: Pokročilé shlukování — DBSCAN a spektrální shlukování

Cílem čtvrtého cvičení je implementovat dva algoritmy shlukování, které — na rozdíl od Cvičení 02 (hierarchické shlukování) a Cvičení 03 (k-means, fuzzy c-means) — **nepředpokládají, že shluk je kulatý útvar kolem středu**: **DBSCAN** (hustotně založené shlukování) a **spektrální shlukování** (shlukování přes graf podobnosti a jeho spektrum). Oba algoritmy se aplikují na syntetická 2D data (soustředné kružnice, půlměsíce, bloby), na kterých je limit „kulových" metod z předchozích cvičení dobře vidět.

Cvičení staví na `Distance` třídách z Cvičení 01 (znovu zkopírovaných do `src/distance.py`) a prohlubuje téma z Cvičení 03: zatímco k-means je **induktivní** (natrénuje obecný model — těžiště — použitelný i na nová data), DBSCAN i spektrální shlukování jsou **transduktivní** — vytvářejí rozdělení pouze pro data, na kterých proběhlo `fit()`, a nemají smysluplný `predict()` pro nová data. Tento rozdíl je v kódu realizován záměrně a je součástí učiva.

---

## Obsah

1. [Cíle cvičení](#cíle-cvičení)
2. [Struktura repozitáře](#struktura-repozitáře)
3. [Instalace a spuštění](#instalace-a-spuštění)
4. [Teoretický základ](#teoretický-základ)
5. [Konfigurace projektu](#konfigurace-projektu)
6. [Pokyny k vypracování](#pokyny-k-vypracování)
7. [Lokální testování](#lokální-testování)
8. [Doplňkové (papírové) příklady](#doplňkové-papírové-příklady)
9. [Odevzdání](#odevzdání)

---

## Cíle cvičení

Po dokončení tohoto cvičení student:

1. **Implementuje DBSCAN od základů** — rozumí pojmům jádrový/hraniční/šumový bod (core/border/noise), hustotní dosažitelnosti a růstu regionu (region growing), a chápe, proč zde na rozdíl od k-means **není** potřeba zadávat počet shluků $k$ — ten se vynoří sám z hustoty dat.
2. **Implementuje spektrální shlukování** — rozumí převodu dat na graf podobnosti (afinitní matice), sestavení Laplaciánu grafu a spektrálnímu embeddingu: proč se vybírají vlastní vektory odpovídající **nejmenším** vlastním číslům (a v čem je to opačné než u PCA).
3. **Rozumí rozdílu mezi induktivním a transduktivním shlukováním** — k-means z Cvičení 03 má `predict()` pro nová data; DBSCAN a spektrální shlukování ho záměrně nemají, a proto, kde to v kódu vidět a jak je to odlišeno od nedokončeného úkolu.
4. **Pracuje s typovanou konfigurací** — čte parametry z `config.yaml` přes dataclassy (`cfg.dbscan.eps` místo `cfg["dbscan"]["eps"]`), stejný vzor jako v Cvičení 03.
5. **Aplikuje oba algoritmy na nekonvexní syntetická data** (soustředné kružnice, půlměsíce) a na vlastní kůži vidí, proč zde metody založené na těžišti nebo lineárním spojování (k-means, hierarchické shlukování) selhávají.
6. **(Bonus) Implementuje Adjusted Rand Index** — externí validační metriku, která na rozdíl od interní siluety z Cvičení 03 vyžaduje znalost skutečných (ground-truth) popisků.

---

## Struktura repozitáře

```
cviceni-04-template/
├── cviceni_04.py                # Hlavní pipeline — spusťte pro průběžné ověření
├── config.yaml                  # Konfigurace experimentu (YAML)
├── priklady_04.md               # Papírové (teoretické) příklady k DBSCAN a spektrálnímu shlukování
├── requirements.txt             # Python závislosti
├── src/
│   ├── __init__.py              # Re-exporty balíčku (neupravujte)
│   ├── distance.py              # Distance (ABC), EuclideanDistance, Manhattan, Cosine — z Cvičení 01
│   ├── base.py                  # Clusterer (ABC) — tenké sdílené rozhraní fit/predict (neupravujte)
│   ├── dbscan.py                # DBSCAN — hustotně založené shlukování
│   ├── spectral.py              # SpectralClustering — shlukování přes graf podobnosti a jeho spektrum
│   └── metrics.py               # adjusted_rand_index — BONUS, externí validace
├── dataio/
│   ├── __init__.py              # Re-exporty balíčku (neupravujte)
│   ├── config_manager.py        # Dataclassy + load_config, validate_config (předimplementováno)
│   ├── datasets.py               # make_dataset() — generátory kružnic/půlměsíců/blobů (předimplementováno)
│   └── plotting.py              # Vizualizace shluků, k-distance grafu a vlastních čísel (předimplementováno)
├── data/
│   └── .gitkeep                 # Připraveno pro budoucí rozšíření — data se v tomto cvičení generují synteticky
├── graphs/                      # Výstupní složka pro grafy (generuje se automaticky)
├── test_cviceni_04.py           # Automatické testy (pytest)
└── .gitignore
```

> **Poznámka k souborům `__init__.py`:** Každá složka s Python kódem (`src/`, `dataio/`) obsahuje `__init__.py`, který ji označuje jako balíček a definuje veřejné API. Díky tomu lze psát `from src import DBSCAN` místo `from src.dbscan import DBSCAN`. **Tyto soubory neupravujte.**

> **Na rozdíl od Cvičení 03** je celý balíček `dataio/` (včetně konfigurace) v tomto cvičení **předimplementován** — žádný `NotImplementedError` v `dataio/` nenajdete. Studentská práce je soustředěna výhradně do `src/`.

---

## Instalace a spuštění

### 1. Vytvoření virtuálního prostředí

```bash
python -m venv .venv
```

Aktivace (Windows):
```bash
.venv\Scripts\activate
```

Aktivace (Linux / macOS):
```bash
source .venv/bin/activate
```

### 2. Instalace závislostí

```bash
pip install -r requirements.txt
```

### 3. Spuštění

```bash
python cviceni_04.py
```

Pipeline načte konfiguraci, vygeneruje syntetický dataset, spustí DBSCAN a spektrální shlukování a zobrazí výsledky. Každá fáze (DBSCAN, spektrální shlukování) je obalena samostatným blokem `try/except NotImplementedError`, takže selhání jedné fáze (např. DBSCAN ještě není hotový) nezabrání spuštění a vyhodnocení ostatních — dostanete co nejvíce zpětné vazby o tom, co ještě zbývá doplnit. U každé nedokončené fáze skript vypíše zprávu začínající `Úkol:` a explicitně ji označí jako *úkol pro studenta*.

> **Pozor na jeden typ hlášky, který úkolem není:** metoda `predict()` u obou algoritmů vždy vyvolá `NotImplementedError`, jehož zpráva **nezačíná** slovem `Úkol:` — jde o záměrné, trvalé architektonické rozhodnutí (DBSCAN a spektrální shlukování jsou transduktivní, viz níže), ne o nedokončenou implementaci. `cviceni_04.py` tuto metodu ani nevolá; pokud si ji vyzkoušíte sami v konzoli, nesnažte se ji „opravit".

Jednotlivé algoritmy lze mezitím ověřovat přes `pytest`, viz [Lokální testování](#lokální-testování).

---

## Teoretický základ

### 1. Proč DBSCAN a spektrální shlukování?

K-means (Cvičení 03) přiřazuje body k nejbližšímu těžišti a hierarchické shlukování (Cvičení 02) spojuje body na základě vzdálenosti — obě metody tak implicitně předpokládají, že shluk je přibližně **kulový/kompaktní útvar**. Na dvou soustředných kružnicích nebo dvou propletených půlměsících tento předpoklad neplatí: těžiště obou kružnic leží ve stejném bodě (středu), takže k-means je nedokáže rozlišit vůbec.

DBSCAN a spektrální shlukování řeší tento problém dvěma zcela odlišnými cestami:

- **DBSCAN** nedefinuje shluk pomocí středu, ale pomocí **hustoty** — shluk je souvislá oblast s dostatečně vysokou hustotou bodů, oddělená od ostatních shluků oblastmi s nízkou hustotou.
- **Spektrální shlukování** data nejprve **přemapuje do jiného prostoru** (odvozeného z grafu podobnosti), ve kterém teprve spustí obyčejné k-means. Klíčová myšlenka: *když data nejsou oddělitelná v původním prostoru, najdi prostor, ve kterém oddělitelná jsou* — stejný motiv se objeví znovu u PCA (Cvičení 05) a u kernel trick u SVM.

---

### 2. DBSCAN — hustotně založené shlukování

DBSCAN (*Density-Based Spatial Clustering of Applications with Noise*) pracuje se dvěma parametry:

- $\varepsilon$ (`eps`) — poloměr okolí bodu,
- `min_samples` — minimální počet bodů (včetně bodu samotného) v tomto okolí, aby byl bod považován za „hustý".

Na základě těchto parametrů se každý bod klasifikuje do jedné ze tří kategorií:

| Kategorie | Definice |
|:---|:---|
| **Jádrový bod (core point)** | Má ve svém $\varepsilon$-okolí alespoň `min_samples` bodů (včetně sebe). |
| **Hraniční bod (border point)** | Sám není jádrový, ale leží v $\varepsilon$-okolí nějakého jádrového bodu. |
| **Šum (noise)** | Ani jádrový, ani hraniční. Ve výsledných popiscích má hodnotu `-1`. |

**Algoritmus (region growing):**

```
Pro každý dosud nenavštívený bod i:
    sousedé = eps-okolí bodu i (_region_query)
    pokud len(sousedé) < min_samples:
        i je (zatím) šum — může být později "přebrán" jako hraniční bod
    jinak:
        i je jádrový bod → založ nový shluk a expanduj ho:
            postupně procházej frontu sousedů; kdykoli narazíš na
            další jádrový bod, přidej JEHO sousedy do fronty dál
            (klasická BFS/DFS expanze regionu)
```

> **BFS/DFS:** *Breadth-First Search* (prohledávání do šířky) a *Depth-First Search* (prohledávání do hloubky) — dva klasické způsoby procházení grafu. BFS bere body z fronty v pořadí, v jakém byly přidány (nejdřív všichni bezprostřední sousedé, pak sousedé sousedů), DFS naopak zajde nejdřív co nejdál jedním směrem a pak se vrací. Pro výsledek DBSCAN je to jedno — obě strategie nakonec navštíví stejnou množinu bodů (celý hustotně propojený region) a přiřadí je do stejného shluku, liší se jen pořadí, ve kterém se k nim dostanou.

Důležité důsledky tohoto postupu:

- **Počet shluků $k$ není parametrem** — na rozdíl od k-means se nezadává, ale vyplyne za běhu z hustoty dat vzhledem k $\varepsilon$ a `min_samples`.
- **Hustotní dosažitelnost ≠ přímá blízkost.** Dva body mohou patřit do stejného shluku, i když jejich vzájemná vzdálenost výrazně přesahuje $\varepsilon$ — stačí, že jsou propojeny řetězcem jádrových bodů.
- DBSCAN je **jednoprůchodový** algoritmus (žádná iterativní smyčka konvergence jako u k-means/FCM) — proto v tomto cvičení nedědí ze žádné sdílené iterační bázové třídy, pouze z tenkého rozhraní `Clusterer`.

**Volba $\varepsilon$:** typicky se odhaduje z tzv. *k-distance grafu* — pro každý bod se spočítá vzdálenost k jeho $k$-tému nejbližšímu sousedovi (kde $k \approx$ `min_samples`), tyto vzdálenosti se seřadí vzestupně a vykreslí. „Loket" (elbow) v grafu napovídá vhodnou hodnotu $\varepsilon$. V pipeline tento graf zajišťuje `dataio.plotting.plot_k_distance`.

---

### 3. Spektrální shlukování — shlukování přes graf

**Co je to graf.** V matematice je graf dvojice $G = (V, E)$: množina **vrcholů** (uzlů) $V$ a množina **hran** $E$, které vrcholy spojují. V tomto cvičení je graf:

- **neorientovaný** — hrana mezi body $i$ a $j$ nemá směr, jen jednu společnou váhu $W_{ij} = W_{ji}$ (podobnost je oboustranná: jak moc je $i$ podobné $j$, tolik je $j$ podobné $i$),
- **ohodnocený (vážený)** — hrana nenese jen informaci „spojeno/nespojeno", ale reálné číslo vyjadřující míru podobnosti,
- **úplný** — v této verzi existuje hrana mezi úplně každou dvojicí bodů, jen s různě velkou vahou (u nepodobných bodů blízkou nule).

Vrcholy grafu jsou přímo vzorky dat ($n$ bodů → $n$ vrcholů) a hrany s jejich vahami jsou přesně afinitní matice $W$ z kroku 1. Od kroku 2 dál se pak pracuje čistě s touto maticí — původní souřadnice bodů se dál nepoužívají, jen síť vzájemných podobností mezi nimi.

Spektrální shlukování má tři kroky, které v `src/spectral.py` odpovídají třem metodám:

**Krok 1 — afinitní matice (`_affinity_matrix`):** data se poprvé „stávají grafem" — každý vzorek je uzel a hrana mezi uzly $i$, $j$ má váhu podle Gaussova (RBF) jádra:

$$W_{ij} = \exp\!\left(-\frac{d(i,j)^2}{2\sigma^2}\right)$$

kde $d(i,j) = \texttt{self.distance.calculate(x[i], x[j])}$. Parametr $\sigma$ řídí „dosah" podobnosti: malé $\sigma$ dělá graf řídký (podobné jsou jen velmi blízké body), velké $\sigma$ ho zahušťuje.

**Krok 2 — Laplacián grafu (`_laplacian`):** nenormalizovaný Laplacián

$$L = D - W$$

kde $D$ je diagonální matice stupňů, $D_{ii} = \sum_j W_{ij}$ — kolik celkové podobnosti má bod $i$ ke všem ostatním bodům dohromady.

*Co si pod $L$ představit:* zkuste každému bodu přiřadit nějaké číslo $f_i$ (jeho budoucí pozici v embeddingu) a spočítat „cenu" za každou dvojici bodů jako $W_{ij}\cdot(f_i-f_j)^2$ — podobnost krát čtverec rozdílu přiřazených čísel. Když dva hodně podobné body ($W_{ij}$ vysoké) dostanou hodně odlišná čísla, cena je vysoká; u nepodobných bodů ($W_{ij}\approx 0$) na rozdílu skoro nezáleží. Součet této ceny přes všechny dvojice bodů je přesně (až na konstantu) $f^T L f$:

$$f^T L f = \frac12\sum_{i,j} W_{ij}(f_i - f_j)^2$$

Hledání vlastních vektorů $L$ s **nejmenšími** vlastními čísly = hledání takového přiřazení čísel, které tuto celkovou „cenu za nesouhlas" minimalizuje — tedy dá podobným bodům podobná čísla a nepodobným bodům volnost se rozejít. Jinými slovy: nejmenší vlastní čísla odpovídají nejlevnějším způsobům, jak graf „přestřihnout" na kusy (přestřihnete jen slabé, nepodobné hrany) — a to je přesně hranice mezi shluky, kterou hledáme.

**Co je to vlastní vektor a vlastní číslo matice.** Pro čtvercovou matici $A$ (zde Laplacián $L$) se nenulový vektor $v$ nazývá **vlastní vektor**, existuje-li číslo $\lambda$ (**vlastní číslo**), pro které platí:

$$A v = \lambda v$$

Násobení matice takovým vektorem tedy nezmění jeho směr — pouze ho **roztáhne nebo zmenší** (podle velikosti $\lambda$), případně otočí o 180°, je-li $\lambda$ záporné. U naprosté většiny vektorů násobení maticí změní jak směr, tak délku; vlastní vektory jsou výjimečné směry, které matice pouze škáluje.

Laplacián $L$ je symetrická matice (plyne ze symetrie $W$), a pro symetrické matice platí důležitá věta lineární algebry: mají $n$ navzájem kolmých (ortogonálních) vlastních vektorů a všechna jejich vlastní čísla jsou reálná. Právě proto `_spectral_embedding` používá `numpy.linalg.eigh` (specializovanou variantu pro symetrické matice) místo obecné `numpy.linalg.eig` — je rychlejší, numericky stabilnější a vrací vlastní čísla rovnou seřazená vzestupně.

Spojení s předchozí intuicí: podle matematického faktu známého jako Rayleighův podíl je vlastní vektor $L$ s **nejmenším** vlastním číslem přesně ten vektor $f$, který minimalizuje „cenu za nesouhlas" $f^TLf$ popsanou výše. Výběr $k$ vlastních vektorů s nejmenšími vlastními čísly v kroku 3 tedy znamená: vyberte $k$ nejlevnějších (nejpřirozenějších) způsobů, jak body podle podobnosti rozdělit.

**Krok 3 — spektrální embedding (`_spectral_embedding`):** spočítejte vlastní čísla a vektory $L$ (pomocí `numpy.linalg.eigh`) a vyberte $k$ vlastních vektorů odpovídajících $k$ **nejmenším** vlastním číslům. Poskládáním těchto vektorů jako sloupců vznikne embedding tvaru $(n, k)$.

> **Pozor na záměnu s PCA:** PCA pro redukci dimenze vybírá vlastní vektory odpovídající **největším** vlastním číslům (směry s největším rozptylem dat). Zde je tomu záměrně naopak — vybírají se **nejmenší** vlastní čísla Laplaciánu (směry „nejslabších" řezů grafu, které nejpřirozeněji oddělují shluky). Prohození pořadí celou metodu tiše rozbije — nevyvolá se žádná chyba, embedding ale bude nesmyslný.

**Poslední krok (předimplementován ve `fit()`):** samotný embedding ještě žádné shluky nevytváří — je to jen transformace prostoru, ve které jsou shluky (na rozdíl od původního prostoru) snadno oddělitelné. Skutečné rozdělení do shluků provede až obyčejné k-means spuštěné na tomto embeddingu.

**Volba počtu shluků — eigengap:** Laplacián $L$ je čtvercová matice $(n,n)$, má tedy přesně $n$ vlastních čísel — tolik, kolik je datových bodů. Počet (skoro) nulových vlastních čísel odpovídá počtu (skoro) souvislých komponent grafu. Výrazná mezera (*eigengap*) mezi $k$-tým a $(k+1)$-ním nejmenším vlastním číslem je klasická heuristika pro volbu $k$.

> **Jak mezeru správně hledat:** vlastní čísla s rostoucím indexem obecně rostou, takže absolutně největší mezera mezi sousedními hodnotami bývá typicky až někde uprostřed nebo na konci seřazené řady — tam jsou čísla větší, takže i drobné relativní rozdíly vypadají v absolutní hodnotě velké. Pro volbu $k$ to ale nic neznamená. Relevantní je mezera **mezi nejmenšími vlastními čísly**, a to spíš **relativně** (kolikrát je další hodnota větší než ta předchozí) než v absolutním rozdílu. Prakticky: podívejte se jen na prvních cca 10–20 nejmenších vlastních čísel a hledejte mezi nimi výrazný poměrový skok. Na lineární ose s celým rozsahem $n$ hodnot bývá tato oblast u větších datasetů stlačená do pár pixelů u nuly a prakticky neviditelná — vyplatí se ji buď vykreslit zvlášť (jen prvních pár desítek hodnot), nebo pro tuto část použít logaritmické měřítko osy $y$.

V pipeline tuto vizualizaci zajišťuje `dataio.plotting.plot_eigenvalues` nad atributem `model.eigenvalues_`.

---

### 4. Induktivní vs. transduktivní shlukování

| | K-means (Cvičení 03) | DBSCAN / spektrální shlukování (toto cvičení) |
|:---|:---|:---|
| Co se „naučí" | Obecný model — těžiště v prostoru příznaků | Pouze rozdělení dat, na kterých proběhl `fit()` |
| `predict()` na nová data | Ano — spočítá vzdálenost k těžištím | **Ne** — nedefinováno |
| Proč | Těžiště existují nezávisle na konkrétních bodech | Neexistuje obecný model prostoru (hustota/graf jsou vlastností celé trénovací množiny) |

Tento rozdíl je v kódu realizován záměrně:

- Obě třídy dědí z tenkého společného rozhraní `Clusterer` (`src/base.py`), který definuje `fit(x)` a `predict()`, ale **žádnou** sdílenou logiku navíc.
- `predict()` u obou tříd vždy vyvolá `NotImplementedError` s vysvětlením, **proč** predikce nedává smysl — tato zpráva **nezačíná** slovem `Úkol:`, protože nejde o nedokončený úkol, ale o trvalé, záměrné omezení algoritmu.
- V celém repozitáři tak existují **dva zcela odlišné druhy** `NotImplementedError`:
  1. **Studentský úkol** — zpráva začíná `Úkol:` (případně `Úkol: (BONUS)` u bonusové metriky). Toto je vždy potřeba doplnit.
  2. **Záměrné/trvalé chování** — zpráva nezačíná `Úkol:`. Toto **neopravujte** — je to hotové a správné chování.

---

### 5. Metriky vzdálenosti (z Cvičení 01)

Stejné třídy jako v Cvičení 01 a Cvičení 03, znovu zkopírované do `src/distance.py`:

| Třída | Vzorec | Je metrikou? |
|:---|:---|:---:|
| `EuclideanDistance` | $\sqrt{\sum_j (a_j - b_j)^2}$ | Ano |
| `ManhattanDistance` | $\sum_j \|a_j - b_j\|$ | Ano |
| `CosineCoeficient` | $1 - \dfrac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$ | Ne* |

*Kosinová vzdálenost porušuje trojúhelníkovou nerovnost.

Metoda `create_distance_matrix(x)` je **předimplementována** v abstraktní třídě `Distance`. V tomto cvičení se ale přímo nevyužívá — DBSCAN i spektrální shlukování volají `self.distance.calculate(...)` po jednotlivých dvojicích bodů (zůstává zde jen kvůli návaznosti na předchozí cvičení).

---

### 6. Adjusted Rand Index — externí validace (BONUS)

Cvičení 03 zavedlo **interní** silhouetové skóre, které kvalitu shlukování posuzuje pouze z dat samotných, bez znalosti „správného" řešení. Adjusted Rand Index (ARI) je naproti tomu **externí** validační metrika — vyžaduje ground-truth popisky.

To je zde smysluplné, protože u syntetických dat (kružnice, půlměsíce) skutečné popisky známe (byly použity k vygenerování dat), zatímco silueta na protáhlých nebo prstencových shlucích často selhává nebo zavádí (implicitně předpokládá přibližně kulaté, kompaktní shluky).

ARI navíc řeší problém, že číslování shluků je arbitrární (zda algoritmus pojmenuje shluk „0" nebo „1", nemá sémantický význam) — místo porovnávání popisků napřímo počítá shodu na úrovni **dvojic vzorků** (pair-counting):

$$\text{ARI} = \frac{\text{index} - \text{očekávaný index}}{\text{maximální index} - \text{očekávaný index}}$$

kde `index` $= \sum_{ij} \binom{n_{ij}}{2}$ je počet dvojic bodů, které se shodnou v obou popiscích (kontingenční tabulka $n_{ij}$), a zbylé dva členy normalizují výsledek tak, aby náhodné přiřazení dávalo hodnotu blízkou 0 a dokonalá shoda hodnotu 1.

---

## Konfigurace projektu

### Soubor `config.yaml`

```yaml
data:
  dataset: circles        # circles | moons | blobs
  n_samples: 300
  noise: 0.05
  factor: 0.5              # pouze pro circles — poměr vnitřního a vnějšího poloměru
  random_state: 42

dbscan:
  eps: 0.2
  min_samples: 5

spectral:
  n_clusters: 2
  sigma: 0.1                # šířka Gaussova jádra afinitní matice
```

Přepnutí na jiný syntetický dataset nebo jiné hyperparametry je pouze úprava konfigurace, ne kódu.

### Typovaná konfigurace (dataclassy)

Konfigurace je načtena funkcí `load_config()` a vrácena jako instance typované dataclassy `ExperimentConfig` — přístup k parametrům je přes atributy, ne slovníkové klíče:

```
# Místo:   cfg["dbscan"]["eps"]     ← runtime chyba při překlepu
# Správně: cfg.dbscan.eps           ← editor odhalí překlep okamžitě
```

Struktura dataclassů:

```
ExperimentConfig
    ├── data: DataConfig
    │       ├── dataset: str
    │       ├── n_samples: int
    │       ├── noise: float
    │       ├── factor: float
    │       └── random_state: int
    ├── dbscan: DBSCANConfig
    │       ├── eps: float
    │       └── min_samples: int
    └── spectral: SpectralConfig
            ├── n_clusters: int
            └── sigma: float
```

`load_config()` volá `validate_config()`, která ověří rozsahy hodnot (např. `eps > 0`, `n_clusters >= 2`) a při neplatné konfiguraci vyvolá srozumitelný `ValueError` — tato část je **předimplementována**, není potřeba do ní zasahovat.

---

## Pokyny k vypracování

Otevřete soubory popsané níže a nahraďte všechny výskyty `raise NotImplementedError(...)`, jejichž zpráva začíná `Úkol:`, funkčním kódem. **Zprávy, které `Úkol:` nezačínají** (metody `predict()` u obou algoritmů), neopravujte — jde o záměrné trvalé chování, viz [Induktivní vs. transduktivní shlukování](#4-induktivní-vs-transduktivní-shlukování).

Komentáře ve tvaru `# assert  Ověřte, že ...` jsou **nápovědy pro validaci vstupů**. Napište odpovídající příkazy `assert` na daná místa.

---

### Předpoklad: třídy vzdálenosti z Cvičení 01 — `src/distance.py`

DBSCAN i afinitní matice spektrálního shlukování volají `self.distance.calculate(x, y)` — bez funkční implementace se cvičení nespustí. **Zkopírujte** svoji implementaci z Cvičení 01. Kostra tříd je připravena — stačí doplnit těla metod `calculate` v `EuclideanDistance`, `ManhattanDistance` a `CosineCoeficient`.

---

### Blok I: DBSCAN — `src/dbscan.py`

#### `_region_query(x, point_idx, eps)`

```
# Pro každý bod v x vypočítejte vzdálenost k bodu x[point_idx]
# pomocí self.distance.calculate(x[point_idx], x[i])
# Vraťte seznam indexů bodů, jejichž vzdálenost je <= eps
# (point_idx je v seznamu vždy zahrnut — vzdálenost od sebe sama je 0)
```

#### `fit(x)`

```
# 1. Připravte pole štítků (start: vše "nenavštíveno" / šum -1)
#    a zvlášť evidenci navštívených bodů — navštívení bodu a jeho
#    konečné zařazení do shluku jsou dvě různé věci
# 2. Pro každý dosud nenavštívený bod i:
#    a. označte i jako navštívený
#    b. sousedé = self._region_query(x, i, self.eps)
#    c. pokud len(sousedé) < self.min_samples: i je (zatím) šum
#       (může být později "přebrán" jako hraniční bod)
#    d. jinak: i je jádrový bod — založte nový shluk a expandujte ho
#       (BFS/DFS: kdykoli narazíte na dalšího jádrového souseda,
#       přidejte JEHO sousedy do fronty k prozkoumání dál)
# 3. Uložte výsledek do self.labels_, vraťte self
```

> `predict()` je v tomto souboru už hotový — vždy vyvolá vysvětlující výjimku (transduktivní algoritmus). Neupravujte ho.

---

### Blok II: Spektrální shlukování — `src/spectral.py`

#### `_affinity_matrix(x)`

```
# Pro každou dvojici bodů i, j:
#   d = self.distance.calculate(x[i], x[j])
#   w[i, j] = exp(-d**2 / (2 * self.sigma**2))
# Vraťte symetrickou matici (n, n) s jedničkami na diagonále
```

#### `_laplacian(w)`

```
# d_diag[i] = součet řádku w[i, :]   (diagonální matice stupňů D)
# Vraťte L = diag(d_diag) - w
```

#### `_spectral_embedding(l, k)`

```
# eigvals, eigvecs = numpy.linalg.eigh(l)
# Vyberte k sloupců eigvecs odpovídajících k NEJMENŠÍM eigvals
#   (pozor — opak PCA, viz teoretický základ výše)
# Vraťte (embedding, eigvals), embedding má tvar (n, k)
```

> `fit()` (spojuje všechny tři kroky a spouští k-means na výsledném embeddingu) i `predict()` jsou v tomto souboru už hotové. Neupravujte je.

---

### Blok III (BONUS): Adjusted Rand Index — `src/metrics.py`

#### `adjusted_rand_index(labels_true, labels_pred)`

```
# 1. Sestavte kontingenční tabulku n_ij (labels_true vs. labels_pred)
# 2. Spočtěte řádkové součty a_i, sloupcové součty b_j a celkem n
# 3. C(k, 2) = k * (k - 1) / 2   (pro k < 2 je C(k, 2) = 0)
# 4. index          = sum_ij C(n_ij, 2)
#    expected_index = [sum_i C(a_i, 2)] * [sum_j C(b_j, 2)] / C(n, 2)
#    max_index      = 0.5 * ([sum_i C(a_i, 2)] + [sum_j C(b_j, 2)])
#    ARI = (index - expected_index) / (max_index - expected_index)
```

Tento blok je nepovinný — pipeline (`cviceni_04.py`) ho volá jen v závěrečném porovnání a jeho absence žádnou jinou část nezablokuje.

---

## Lokální testování

Spusťte automatické testy příkazem:

```bash
python -m pytest test_cviceni_04.py -v
```

| Test | Co ověřuje |
|:---|:---|
| `test_dbscan_matches_sklearn_on_blobs` | Shoda vlastní implementace DBSCAN se `sklearn.cluster.DBSCAN` na dobře oddělených blobs (přes `adjusted_rand_score`) a shodu v počtu šumových bodů |
| `test_spectral_clustering_separates_circles` | Spektrální shlukování dobře odděluje dvě soustředné kružnice (`make_circles`) |
| `test_affinity_matrix_is_symmetric_with_unit_diagonal` | `_affinity_matrix` vrací symetrickou matici s jedničkovou diagonálou |
| `test_laplacian_rows_sum_to_zero` | Řádky `_laplacian` sčítají (přibližně) na 0 |

Testy záměrně neporovnávají syrové popisky shluků přímo (číslování shluků je arbitrární), ale přes `adjusted_rand_score`. Používají vlastní pomocnou třídu `DummyDistance` místo `src.distance.Distance`, takže testy DBSCAN a spektrálního shlukování běží nezávisle na tom, zda už máte hotový Předpoklad výše.

Dokud nejsou příslušné metody implementovány, testy se hlásí jako `xfail` (očekávané selhání) a celá sada testů proto skončí s návratovým kódem 0 — to je záměrné chování, ne chyba. Jakmile implementaci doplníte, stejné testy začnou procházet.

Průběžně ověřujte i celou pipeline:

```bash
python cviceni_04.py
```

---

## Doplňkové (papírové) příklady

Soubor `priklady_04.md` obsahuje šest číselných příkladů na papír (tři na spektrální shlukování — Gaussovo jádro, Laplacián, koncepční počet vlastních čísel; tři na DBSCAN — klasifikace core/border/noise, region growing, kvalitativní vliv parametrů), včetně kompletně vypracovaného řešení. Slouží k procvičení výpočtů z teoretického základu bez nutnosti psát kód.

---

## Odevzdání

Úloha se odevzdává prostřednictvím systému **GitHub Classroom**. Po dokončení implementace proveďte:

```bash
git add src/distance.py src/dbscan.py src/spectral.py src/metrics.py
git commit -m "Implementace cvičení 4"
git push
```

Po přijetí příkazu `push` se automaticky spustí testovací skripty, které ověří správnost výpočtů. Výsledek bude zobrazen přímo v rozhraní GitHub u vašeho repozitáře formou zelené fajfky (úspěch) nebo červeného křížku (neúspěch).

> **Soubory, které se neodevzdávají:** `dataio/__init__.py`, `dataio/config_manager.py`, `dataio/datasets.py`, `dataio/plotting.py`, `src/__init__.py`, `src/base.py`, `cviceni_04.py` a `test_cviceni_04.py` jsou předimplementovány a nemají se měnit. Systém tyto soubory ignoruje a hodnotí pouze výše uvedené.

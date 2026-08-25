# Příklady 04 — Pokročilé shlukování (DBSCAN, spektrální shlukování)

> Poznámka ke konvenci: v příkladech na DBSCAN používáme konvenci, že do
> velikosti sousedství bodu `p` (pro rozhodnutí, zda je `p` jádrový bod)
> **počítáme i samotný bod `p`**. Bod je tedy jádrový (*core point*), pokud
> počet bodů (včetně sebe sama) v okolí o poloměru `eps` je alespoň
> `min_samples`. Tato konvence odpovídá např. implementaci
> `sklearn.cluster.DBSCAN`.

## Spektrální shlukování

### Příklad 1 — Afinita přes Gaussovo jádro

Máme čtyři body v rovině:

- A = (0, 0)
- B = (1, 0)
- C = (0, 1)
- D = (1, 1)

Afinitu (podobnost) mezi dvojicí bodů počítáme pomocí Gaussova jádra:

```
W[i,j] = exp( -d(i,j)^2 / (2*sigma^2) )
```

kde `d(i,j)` je euklidovská vzdálenost bodů `i` a `j`, a `sigma = 1`.

**Úkol:** Spočítejte vzdálenosti `d(i,j)` pro všech 6 dvojic bodů a z nich
afinity `W[i,j]`. Sestavte výslednou symetrickou matici afinity `W` (4×4,
s nulovou diagonálou).

---

### Příklad 2 — Laplacián

Je dána matice afinity (podobnosti) pro čtyři body:

```
        1     2     3     4
   1 [ 0.0   0.8   0.1   0.0 ]
   2 [ 0.8   0.0   0.0   0.1 ]
   3 [ 0.1   0.0   0.0   0.7 ]
   4 [ 0.0   0.1   0.7   0.0 ]
```

**Úkol:**
a) Spočítejte stupňovou matici `D` (diagonální matice, na diagonále součty
   řádků `W`).
b) Spočítejte nenormalizovaný Laplacián grafu `L = D - W`.
c) Ověřte, že součet prvků v každém řádku `L` je roven 0 (tato vlastnost
   platí pro Laplacián vždy — proč?).

---

### Příklad 3 — Koncepční počet vlastních čísel

Máme čtyři body rozdělené do dvou dvojic. Matice afinity je téměř blokově
diagonální:

```
        1      2      3      4
   1 [ 0.00   0.90   0.01   0.01 ]
   2 [ 0.90   0.00   0.01   0.01 ]
   3 [ 0.01   0.01   0.00   0.90 ]
   4 [ 0.01   0.01   0.90   0.00 ]
```

Body 1 a 2 mají mezi sebou vysokou afinitu (0.90) a jen velmi slabou vazbu
na body 3 a 4 (0.01). Podobně body 3 a 4 mají vysokou afinitu mezi sebou a
slabou vazbu na 1 a 2.

**Úkol (bez ručního počítání vlastních čísel!):** Na základě struktury grafu
(počtu a síly propojení, tzv. "skoro-komponent") odhadněte:

a) Kolik vlastních čísel Laplaciánu `L` bude (skoro) nulových?
b) Co to znamená pro doporučený počet shluků `k` při spektrálním
   shlukování?
c) Co byste čekali za rozdíl v odpovědi, kdyby afinity 0.01 byly přesně 0
   (body 1,2 a 3,4 by tvořily zcela izolované komponenty)?

*(Nápověda: počet přesně nulových vlastních čísel Laplaciánu odpovídá počtu
souvislých komponent grafu. Slabé, ale nenulové propojení mezi dvěma
"skoro-komponentami" se projeví jako jedno vlastní číslo blízké nule, ale
ne přesně nule — vznikne výrazná mezera, tzv. eigengap, mezi tímto malým
číslem a dalším, výrazně větším vlastním číslem.)*

---

### Příklad 4 — Stupňová matice a „hub" bod

Je dána afinitní matice `W` pro pět bodů:

```
        1      2      3      4      5
   1 [ 0.00   0.40   0.70   0.30   0.05 ]
   2 [ 0.40   0.00   0.75   0.35   0.05 ]
   3 [ 0.70   0.75   0.00   0.65   0.15 ]
   4 [ 0.30   0.35   0.65   0.00   0.10 ]
   5 [ 0.05   0.05   0.15   0.10   0.00 ]
```

**Úkol:**

a) Spočítejte stupňovou matici `D` (tj. `D_ii = sum_j W[i,j]` pro každý bod).
b) Který bod má nejvyšší stupeň — je tedy nejvíc "propojený" (podobný)
   se zbytkem grafu? Který bod má naopak nejnižší stupeň — je od
   zbytku grafu nejvíc izolovaný?
c) Krátce (jednou větou) odhadněte, jak by se tyto dva krajní body chovaly
   ve spektrálním embeddingu — kam by je algoritmus pravděpodobně umístil
   vzhledem k ostatním bodům?

---

## DBSCAN

### Příklad 1 — Klasifikace na core/border/noise

Máme 8 bodů v rovině:

| bod | souřadnice |
|-----|------------|
| P1  | (0, 0)     |
| P2  | (1, 0)     |
| P3  | (2, 0)     |
| P4  | (2, 1)     |
| P5  | (0, 3)     |
| P6  | (0, 4)     |
| P7  | (5, 5)     |
| P8  | (3.4, 1)   |

Parametry DBSCAN: `eps = 1.5`, `min_samples = 3` (počítáno včetně bodu
samotného — viz konvence v úvodu).

**Úkol:** Pro každý bod spočítejte, kolik bodů (včetně něj samého) leží v
jeho `eps`-okolí, a na základě toho jej klasifikujte jako **jádrový**
(*core*), **hraniční** (*border*), nebo **šumový** (*noise*) bod.

---

### Příklad 2 — Region growing

Použijte stejnou množinu bodů a stejné parametry (`eps = 1.5`,
`min_samples = 3`) jako v příkladu 1.

**Úkol:** Proveďte ručně hustotní růst shluků (*region growing*) a napište
výsledný vektor popisků shluků pro body P1–P8 (šum označte jako `-1`,
shluky číslujte od 0).

> Pozor: hustotní dosažitelnost (*density-reachability*) není totéž co
> přímá blízkost dvou bodů! Bod může patřit do stejného shluku jako jiný
> bod, i když spolu **přímo** nesousedí (jejich vzdálenost je větší než
> `eps`) — stačí, když jsou propojeny řetězcem jádrových bodů, z nichž
> každý sousedí s dalším. Při řešení se zaměřte na to, přes které jádrové
> body se hraniční body do shluku "napojují", ne na přímou vzdálenost mezi
> všemi dvojicemi bodů.

---

### Příklad 3 — Kvalitativní vliv parametrů

Uvažujte znovu situaci z příkladu 1 (stejné body, `eps = 1.5`,
`min_samples = 3`).

**Úkol:** Jednou až dvěma větami s odůvodněním odpovězte:

a) Co se stane s počtem shluků a množstvím šumu, pokud **zvýšíme** `eps`
   (parametry `min_samples` a body necháme beze změny)?
b) Co se stane s počtem shluků a množstvím šumu, pokud **snížíme**
   `min_samples` (parametr `eps` a body necháme beze změny)?

(Neočekává se konkrétní číslo — jde o obecné, ale zdůvodněné pozorování
směru změny.)

---

## Adjusted Rand Index (BONUS)

### Příklad 1 — Výpočet ARI přes pair-counting

Máme 6 bodů. Skutečné (ground-truth) shluky a shluky vrácené nějakým
algoritmem jsou:

```
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 1, 1]
```

(Algoritmus tedy rozdělení skoro trefil — jen třetí bod, který má
`labels_true = 0`, zařadil do shluku `1`.)

**Úkol:**

a) Sestavte kontingenční tabulku `n_ij` (počet bodů se skutečným
   štítkem `i` a predikovaným štítkem `j`; `i, j ∈ {0, 1}`).
b) Spočítejte řádkové součty `a_i`, sloupcové součty `b_j` a celkový
   počet bodů `n`.
c) Pomocí `C(k,2) = k(k-1)/2` spočítejte `index`, `expected_index` a
   `max_index` podle vzorce z `src/metrics.py`, a z nich výsledné ARI.
d) Slovně interpretujte výsledek: je hodnota blíž 1 (skoro dokonalá
   shoda), blíž 0 (náhodné přiřazení), nebo záporná (horší než náhodné)?

*(Nápověda: `C(0,2) = C(1,2) = 0`, `C(2,2) = 1`, `C(3,2) = 3`,
`C(4,2) = 6`, `C(6,2) = 15`.)*
# -*- coding: utf-8 -*-

"""
Created on 25. 08. 2026 at 20:59:30

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Externí validační metriky pro shlukování (BONUS).

    Na rozdíl od interní silueta (silhouette score) z Cvičení 03, která
    posuzuje kvalitu shluků pouze na základě samotných dat (kompaktnost
    a oddělenost shluků, bez znalosti "správného" řešení), metriky v tomto
    modulu vyžadují znalost skutečných (ground-truth) štítků. Proto se
    označují jako externí validace.

    To je užitečné zejména na syntetických datech s neznámým/nekonvexním
    tvarem shluků (soustředné kružnice, půlměsíce apod.), jaká se používají
    v tomto cvičení pro DBSCAN a spektrální shlukování - u nich totiž
    skutečné štítky známe (byly použity k vygenerování dat), zatímco
    silueta na takových protáhlých nebo prstencových shlucích často selhává
    či zavádí, protože implicitně předpokládá přibližně konvexní a kompaktní
    shluky.
"""

import numpy as np


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Vypočítá Adjusted Rand Index (ARI) mezi skutečným (ground-truth)
    přiřazením štítků labels_true a predikovaným přiřazením labels_pred.

    ARI je externí validační metrika shlukování - na rozdíl od interní
    siluety (Cvičení 03) potřebuje znalost skutečných štítků. Zásadní
    vlastností ARI je invariance vůči permutaci štítků: číslování shluků
    (např. zda daný shluk algoritmus pojmenuje "0" nebo "1") je
    arbitrární a nemá žádný sémantický význam, takže naivní přesnost
    (accuracy) by byla nepoužitelná. ARI místo porovnávání štítků
    přímo počítá shodu na úrovni dvojic vzorků (pair-counting) - zda
    dva vzorky patří do stejného/různého shluku podle labels_true a
    podle labels_pred -, a je tedy vůči přeznačení shluků zcela
    necitlivá.

    Postup výpočtu (pair-counting):
    1. Sestavte kontingenční tabulku (matici) n_ij - počet vzorků,
       které mají skutečný štítek i (podle labels_true) a zároveň
       predikovaný štítek j (podle labels_pred).
    2. Spočtěte řádkové součty a_i = sum_j n_ij a sloupcové součty
       b_j = sum_i n_ij, a celkový počet vzorků n.
    3. Pro libovolné celé číslo k definujte C(k, 2) = k * (k - 1) / 2
       (počet neuspořádaných dvojic z k prvků; pro k < 2 je C(k, 2) = 0).
    4. Dosaďte do vzorce:

           index = sum_ij C(n_ij, 2)
           expected_index = [sum_i C(a_i, 2)] * [sum_j C(b_j, 2)] / C(n, 2)
           max_index = 0.5 * ([sum_i C(a_i, 2)] + [sum_j C(b_j, 2)])

           ARI = (index - expected_index) / (max_index - expected_index)

    Výsledná hodnota je 1.0 pro dokonalou shodu, blízká 0.0 pro
    náhodné přiřazení štítků a může být i záporná pro přiřazení horší
    než náhodné.

    Parametry
    ---------
    labels_true : np.ndarray
        1D pole skutečných (ground-truth) štítků shluků, délky n_vzorků.
    labels_pred : np.ndarray
        1D pole predikovaných štítků shluků (např. výstup DBSCAN nebo
        spektrálního shlukování), délky n_vzorků.

    Návratová hodnota
    ------------------
    float
        Hodnota Adjusted Rand Index z intervalu přibližně [-0.5, 1.0].
    """
    # assert  Ověřte, že labels_true i labels_pred jsou typu np.ndarray
    # assert  Ověřte, že labels_true a labels_pred jsou 1D pole stejné délky
    raise NotImplementedError(
        "Úkol: (BONUS) Implementujte výpočet Adjusted Rand Index (ARI) mezi "
        "labels_true a labels_pred pomocí pair-countingu: sestavte "
        "kontingenční tabulku n_ij (skutečný štítek i vs. predikovaný "
        "štítek j), spočtěte řádkové/sloupcové součty a_i, b_j a celkový "
        "počet vzorků n, a dosaďte do vzorce ARI = (sum_ij C(n_ij,2) - "
        "[sum_i C(a_i,2) * sum_j C(b_j,2)] / C(n,2)) / (0.5 * [sum_i "
        "C(a_i,2) + sum_j C(b_j,2)] - [sum_i C(a_i,2) * sum_j C(b_j,2)] / "
        "C(n,2)), kde C(k,2) = k*(k-1)/2."
    )

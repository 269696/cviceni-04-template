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
    Společné abstraktní rozhraní pro shlukovací algoritmy tohoto cvičení.
"""

from abc import ABC, abstractmethod

import numpy as np


class Clusterer(ABC):
    """
    Tenké společné rozhraní použité jak třídou DBSCAN, tak třídou
    SpectralClustering v tomto cvičení.

    Na rozdíl od k-means z Cvičení 03 jsou oba tyto algoritmy
    transduktivní - vytvářejí rozdělení (shluky) pouze pro data, na
    kterých byly natrénovány metodou fit(), a nemají žádný smysluplný
    způsob, jak zařadit zcela nová data do shluků. Proto zde metoda
    predict() záměrně nepřijímá žádná nová data jako argument - buď
    pouze vrátí výsledné labels_ získané ve fit(), nebo (podle konkrétní
    implementace) explicitně vyvolá výjimku vysvětlující, proč je
    predikce na nových datech pro daný algoritmus nedefinovaná. Tuto
    volbu ponechávají konkrétní podtřídy na sobě.
    """

    @abstractmethod
    def fit(self, x: np.ndarray) -> "Clusterer":
        """
        Natrénuje (nafituje) shlukovací algoritmus na datech x.

        Parametry
        ---------
        x : np.ndarray
            Matice dat o rozměru (n_vzorků, n_příznaků).

        Návratová hodnota
        ------------------
        Clusterer
            Vrací self, aby bylo možné volání zřetězit (fit(x).predict()).
        """
        raise NotImplementedError(
            "Úkol: \
            Implementujte metodu fit() pro shlukovací algoritmus podle popisu v docstringu."
        )

    @abstractmethod
    def predict(self) -> np.ndarray:
        """
        Vrátí výsledné přiřazení vzorků do shluků získané metodou fit().

        Návratová hodnota
        ------------------
        np.ndarray
            Pole štítků shluků o délce n_vzorků.
        """
        raise NotImplementedError(
            "Úkol: \
            Implementujte metodu predict() pro shlukovací algoritmus podle popisu v docstringu."
        )

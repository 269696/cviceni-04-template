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
    Zkopírujte sem své řešení z Cvičení 01.

    Jak DBSCAN (vyhledávání sousedů v epsilon-okolí), tak spektrální shlukování
    (sestavení afinitní matice) v tomto cvičení volají metodu ``calculate()``
    této hierarchie tříd - bez jejího doplnění se cvičení nespustí.
"""

from abc import ABC, abstractmethod

import numpy as np


class Distance(ABC):
    """
    Abstraktní základní třída pro výpočet vzdálenosti (nepodobnosti)
    mezi dvěma vektory příznaků.

    Konkrétní podtřídy definují třídní atribut ``is_metric``, který říká,
    zda daná míra splňuje axiomy metriky (zejména trojúhelníkovou
    nerovnost). Např. eukleidovská a manhattanská vzdálenost jsou skutečné
    metriky, zatímco kosinová vzdálenost (1 - kosinová podobnost) trojúhelníkovou
    nerovnost obecně nesplňuje, a tedy metrikou není.
    """

    is_metric: bool

    @abstractmethod
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá vzdálenost (nepodobnost) mezi dvěma 1D vektory příznaků
        x a y.

        Parametry
        ---------
        x : np.ndarray
            První vektor příznaků (1D pole).
        y : np.ndarray
            Druhý vektor příznaků (1D pole), stejné délky jako x.

        Návratová hodnota
        ------------------
        float
            Nezáporné číslo vyjadřující vzdálenost/nepodobnost mezi x a y.
        """
        raise NotImplementedError(
            "Úkol: Implementujte výpočet vzdálenosti mezi dvěma vektory x a y."
            )

    def create_distance_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Sestaví čtvercovou matici vzdáleností pro všechny dvojice vzorků
        v X.

        Pro každou dvojici řádků (vzorků) X[i] a X[j] se zavolá
        self.calculate(X[i], X[j]). Výsledná matice je symetrická
        (calculate je symetrická funkce svých argumentů) a na diagonále
        má nuly (vzdálenost vzorku sama od sebe).

        Poznámka: v tomto cvičení (DBSCAN, spektrální shlukování) se tato
        metoda přímo nevyužívá - oba algoritmy volají self.calculate()
        po jednotlivých dvojicích bodů. Metoda je zde ponechána kvůli
        návaznosti na předchozí cvičení, která ji používají.

        Parametry
        ---------
        x : np.ndarray
            Matice dat o rozměru (n_vzorků, n_příznaků).

        Návratová hodnota
        ------------------
        np.ndarray
            Symetrická matice vzdáleností o rozměru (n_vzorků, n_vzorků)
            s nulovou diagonálou.
        """
        n_samples = x.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples), dtype=float)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance = self.calculate(x[i], x[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix


class EuclideanDistance(Distance):
    """Eukleidovská (L2) vzdálenost mezi dvěma vektory."""

    is_metric = True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá eukleidovskou vzdálenost mezi dvěma 1D vektory příznaků
        x a y.

        Parametry
        ---------
        x : np.ndarray
            První vektor příznaků (1D pole).
        y : np.ndarray
            Druhý vektor příznaků (1D pole), stejné délky jako x.

        Návratová hodnota
        ------------------
        float
            Eukleidovská vzdálenost mezi x a y.
        """
        # assert  Ověřte, že x i y jsou typu np.ndarray
        # assert  Ověřte, že x a y mají stejný tvar (stejný počet příznaků)
        raise NotImplementedError(
            "Úkol: Implementujte výpočet eukleidovskou vzdálenost mezi dvěma vektory x a y."
            )


class ManhattanDistance(Distance):
    """Manhattanská (L1, taxicab) vzdálenost mezi dvěma vektory."""

    is_metric = True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá manhattanskou vzdálenost mezi dvěma 1D vektory příznaků
        x a y.

        Parametry
        ---------
        x : np.ndarray
            První vektor příznaků (1D pole).
        y : np.ndarray
            Druhý vektor příznaků (1D pole), stejné délky jako x.

        Návratová hodnota
        ------------------
        float
            Manhattanská vzdálenost mezi x a y.
        """
        # assert  Ověřte, že x i y jsou typu np.ndarray
        # assert  Ověřte, že x a y mají stejný tvar (stejný počet příznaků)
        raise NotImplementedError(
            "Úkol: Implementujte výpočet manhattanskou vzdálenost mezi dvěma vektory x a y."
            )


class CosineCoeficient(Distance):
    """
    Kosinová vzdálenost (1 - kosinová podobnost) mezi dvěma vektory.

    Pozor: nejedná se o skutečnou metriku (neplatí trojúhelníková
    nerovnost), viz is_metric = False.
    """

    is_metric = False

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá kosinovou vzdálenost (1 - kosinový koeficient podobnosti)
        mezi dvěma 1D vektory příznaků x a y.

        Parametry
        ---------
        x : np.ndarray
            První vektor příznaků (1D pole).
        y : np.ndarray
            Druhý vektor příznaků (1D pole), stejné délky jako x.

        Návratová hodnota
        ------------------
        float
            Kosinová vzdálenost mezi x a y (1 - kosinová podobnost).
        """
        # assert  Ověřte, že x i y jsou typu np.ndarray
        # assert  Ověřte, že x a y mají stejný tvar (stejný počet příznaků)
        # assert  Ověřte, že x ani y nejsou nulové vektory (dělení nulou při normalizaci)
        raise NotImplementedError(
            "Úkol: Implementujte výpočet kosinovou vzdálenost/koeficient mezi dvěma vektory x a y."
        )

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
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Na rozdíl od k-means (Cvičení 03) DBSCAN nevyžaduje předem zadaný počet
    shluků k - ten se přirozeně "vynoří" z hustoty dat na základě dvou
    parametrů: poloměru okolí eps a minimálního počtu sousedů min_samples.
    Algoritmus navíc umí explicitně označit body, které do žádného shluku
    nepatří, jako šum (noise), s konvencí štítku -1 (stejně jako
    sklearn.cluster.DBSCAN).
"""

from typing import Optional

import numpy as np

from src.base import Clusterer
from src.distance import Distance


class DBSCAN(Clusterer):
    """
    Shlukovací algoritmus DBSCAN.

    Algoritmus rozděluje body datové sady do tří kategorií:

    - **jádrové body (core points)** - body, které mají ve svém
      eps-okolí alespoň min_samples sousedů (včetně sebe sama);
    - **hraniční body (border points)** - body, které samy nejsou
      jádrové (mají méně než min_samples sousedů), ale leží v
      eps-okolí nějakého jádrového bodu, a jsou tedy z jádrového bodu
      dosažitelné;
    - **šum (noise)** - body, které nejsou ani jádrové, ani hraniční;
      těmto bodům je přiřazen štítek -1.

    Počet výsledných shluků k **není** parametrem algoritmu - vzniká
    až za běhu jako vedlejší produkt hustoty dat a zvolených hodnot
    eps a min_samples.

    Atributy
    --------
    eps : float
        Poloměr okolí, ve kterém se vyhledávají sousedé bodu.
    min_samples : int
        Minimální počet bodů (včetně bodu samotného) v eps-okolí
        potřebný k tomu, aby byl bod považován za jádrový.
    distance : Distance
        Injektovaný objekt pro výpočet vzdálenosti/nepodobnosti mezi
        dvěma vzorky (viz src.distance.Distance). Instance třídy
        DBSCAN si žádnou konkrétní implementaci vzdálenosti sama
        nevytváří - je jí vždy dodána zvenčí (dependency injection),
        stejně jako v předchozích cvičeních.
    labels_ : Optional[np.ndarray]
        Výsledné štítky shluků získané metodou fit(). Hodnota -1
        označuje šum. Dokud fit() neproběhne, je None.
    """

    def __init__(self, eps: float, min_samples: int, distance: "Distance") -> None:
        """
        Vytvoří instanci DBSCAN s danými hyperparametry a vzdálenostní
        mírou.

        Parametry
        ---------
        eps : float
            Poloměr eps-okolí použitý při vyhledávání sousedů bodu.
        min_samples : int
            Minimální počet bodů v eps-okolí (včetně bodu samotného),
            aby byl bod považován za jádrový.
        distance : Distance
            Objekt implementující výpočet vzdálenosti mezi dvěma
            vzorky (dependency injection - instance se vytváří mimo
            třídu DBSCAN a předává se sem hotová).

        Návratová hodnota
        ------------------
        None
        """
        self.eps: float = eps
        self.min_samples: int = min_samples
        self.distance: "Distance" = distance
        self.labels_: Optional[np.ndarray] = None

    def _region_query(self, x: np.ndarray, point_idx: int, eps: float) -> list[int]:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2
        assert 0 <= point_idx < x.shape[0]

        neighbors = []

        for i in range(x.shape[0]):
            distance = self.distance.calculate(x[point_idx], x[i])

            if distance <= eps:
                neighbors.append(i)

        return neighbors

    def fit(self, x: np.ndarray) -> "DBSCAN":
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2

        n_samples = x.shape[0]

        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0

        for i in range(n_samples):

            if visited[i]:
                continue

            visited[i] = True

            neighbors = self._region_query(x, i, self.eps)

            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue

            labels[i] = cluster_id

            queue = list(neighbors)

            while queue:
                current = queue.pop(0)

                if not visited[current]:
                    visited[current] = True

                    current_neighbors = self._region_query(
                        x, current, self.eps
                    )

                    if len(current_neighbors) >= self.min_samples:
                        for neighbor in current_neighbors:
                            if neighbor not in queue:
                                queue.append(neighbor)

                if labels[current] == -1:
                    labels[current] = cluster_id

            cluster_id += 1

        self.labels_ = labels

        return self

    def predict(self) -> np.ndarray:
        """
        DBSCAN nemá metodu predict() pro nová data.

        DBSCAN je transduktivní algoritmus - nevytváří žádné centroidy
        ani jiný obecný model prostoru příznaků, pouze rozdělí do
        shluků (a šumu) přesně ta data, na kterých proběhlo fit().
        Nemá tedy smysluplný způsob, jak zařadit zcela nový, dosud
        neviděný bod do některého z nalezených shluků - takové
        zařazení by u husotně založeného algoritmu vyžadovalo znovu
        prohledat okolí nového bodu vůči celé původní datové sadě.
        Z tohoto důvodu tato metoda pro nová data nic nepredikuje a
        místo toho vždy vyvolává výjimku.

        Návratová hodnota
        ------------------
        np.ndarray
            Tato metoda nikdy nevrací hodnotu - vždy vyvolá výjimku
            NotImplementedError (viz výše).

        Vyvolává
        --------
        NotImplementedError
            Vždy - DBSCAN nemá jak predikovat na nových datech.
        """
        raise NotImplementedError(
            "DBSCAN je transduktivní algoritmus - nemá centroidy ani jiný model, který by "
            "šlo použít na nová data, a proto nepredikuje. Jediný smysluplný výstup je "
            "self.labels_ získané metodou fit() na trénovacích datech."
        )

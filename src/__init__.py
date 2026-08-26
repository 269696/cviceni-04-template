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
    Balíček src - algoritmy pokročilého shlukování pro Cvičení 04.

    Veřejné API:
        Distance            - abstraktní základ pro metriky vzdálenosti (stub - z Cvičení 01)
        EuclideanDistance   - euklidovská metrika (stub - k implementaci)
        ManhattanDistance   - manhattanská metrika (stub - k implementaci)
        CosineCoeficient    - kosinová podobnost (stub - k implementaci)
        Clusterer           - tenké abstraktní rozhraní fit/predict pro DBSCAN a SpectralClustering
        DBSCAN              - hustotně založené shlukování (stub - k implementaci)
        SpectralClustering  - shlukování přes graf podobnosti a jeho spektrum
                              (stub - k implementaci)
        adjusted_rand_index - Adjusted Rand Index, externí validační metrika (BONUS - stub)
"""

from __future__ import annotations

from src.base import Clusterer
from src.dbscan import DBSCAN
from src.distance import CosineCoeficient, Distance, EuclideanDistance, ManhattanDistance
from src.metrics import adjusted_rand_index
from src.spectral import SpectralClustering

__all__ = [
    "Clusterer",
    "DBSCAN",
    "CosineCoeficient",
    "Distance",
    "EuclideanDistance",
    "ManhattanDistance",
    "adjusted_rand_index",
    "SpectralClustering",
]

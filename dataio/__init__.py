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
    Balíček dataio - vstup/výstup a vizualizace pro Cvičení 04.

    Veřejné API:
        ExperimentConfig    - typovaná konfigurace cvičení
        DataConfig          - nastavení generování syntetického datasetu
        DBSCANConfig        - hyperparametry DBSCAN (eps, min_samples)
        SpectralConfig      - hyperparametry spektrálního shlukování (n_clusters, sigma)
        load_config         - načte konfiguraci z YAML souboru → ExperimentConfig
        make_dataset        - vygeneruje syntetický 2D dataset (circles/moons/blobs)
        plot_clusters       - vykreslí 2D shluky (odliší šum -1)
        plot_k_distance     - k-distance graf pro odhad parametru eps (DBSCAN)
        plot_eigenvalues    - graf vlastních čísel Laplaciánu (eigengap, spektrální shlukování)
"""

from __future__ import annotations

from dataio.config_manager import (
    DataConfig,
    DBSCANConfig,
    ExperimentConfig,
    SpectralConfig,
    load_config,
)
from dataio.datasets import make_dataset
from dataio.plotting import plot_clusters, plot_eigenvalues, plot_k_distance

__all__ = [
    "DataConfig",
    "DBSCANConfig",
    "ExperimentConfig",
    "SpectralConfig",
    "load_config",
    "make_dataset",
    "plot_clusters",
    "plot_eigenvalues",
    "plot_k_distance",
]

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
    Generatory syntetickych 2D datasetu pro demonstraci shlukovacich algoritmu.

    Modul je tenka obalka nad generatory ze sklearn.datasets. Vraci vzdy
    dvojici (X, y_true), kde X jsou 2D souradnice bodu a y_true je vektor
    referencnich (ground-truth) tridnich labelu.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

from dataio.config_manager import DataConfig


def make_dataset(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Vygeneruje syntetický 2D dataset podle konfigurace.

    Podle cfg.dataset vybere prislusny generator ze sklearn:
      - "circles" -> sklearn.datasets.make_circles
      - "moons"   -> sklearn.datasets.make_moons
      - "blobs"   -> sklearn.datasets.make_blobs

    Parametry generatoru (pocet vzorku, sum, nahodny seed apod.) se berou
    primo z predane typovane konfigurace cfg, ktera odpovida sekci "data:"
    z hlavniho config.yaml souboru (viz dataio.config_manager.DataConfig).

    Dulezite: navratova hodnota y_true obsahuje referencni (ground-truth)
    labely pouze pro ucely vyhodnoceni kvality shlukovani a pro vykresleni
    "spravneho" reseni v grafech. y_true se NESMI predavat samotnym
    shlukovacim algoritmum (DBSCAN, spektralni shlukovani apod.) jako
    vstup - ty pracuji pouze s X, jelikoz shlukovani je metoda uceni bez
    ucitele.

    Args:
        cfg: Typovana konfigurace datasetu (DataConfig) s atributy
            dataset, n_samples, noise, factor a random_state.

    Returns:
        Dvojice (X, y_true):
            - X: pole tvaru (n_samples, 2) se souradnicemi bodu
            - y_true: pole tvaru (n_samples,) s referencnimi celociselnymi
              labely (pouze pro vyhodnoceni/vizualizaci, nikdy ne jako
              vstup shlukovacich algoritmu)

    Raises:
        ValueError: pokud cfg.dataset neni jedno z podporovanych jmen.
    """
    if cfg.dataset == "circles":
        x, y_true = make_circles(
            n_samples=cfg.n_samples,
            noise=cfg.noise,
            factor=cfg.factor,
            random_state=cfg.random_state,
        )
    elif cfg.dataset == "moons":
        x, y_true = make_moons(
            n_samples=cfg.n_samples,
            noise=cfg.noise,
            random_state=cfg.random_state,
        )
    elif cfg.dataset == "blobs":
        x, y_true = make_blobs(
            n_samples=cfg.n_samples,
            random_state=cfg.random_state,
        )
    else:
        raise ValueError(
            f"Neznamy nazev datasetu: {cfg.dataset!r}. "
            "Podporovane hodnoty jsou 'circles', 'moons' a 'blobs'."
        )

    return x, y_true

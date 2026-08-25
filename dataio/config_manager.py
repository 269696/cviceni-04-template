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
    Sprava konfigurace cviceni ze souboru YAML.

    Konfigurace je nactena do typovanych dataclass instanci - preklep v nazvu
    atributu odhali editor okamzite, ne az za behu (cfg.dbscan.eps misto
    cfg["dbscan"]["eps"]).
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class DataConfig:
    """Nastaveni generovani syntetickeho 2D datasetu."""

    dataset: str
    n_samples: int
    noise: float
    factor: float
    random_state: int


@dataclass
class DBSCANConfig:
    """Hyperparametry algoritmu DBSCAN."""

    eps: float
    min_samples: int


@dataclass
class SpectralConfig:
    """Hyperparametry spektralniho shlukovani."""

    n_clusters: int
    sigma: float


@dataclass
class ExperimentConfig:
    """Kompletni konfigurace cviceni nactena z YAML souboru."""

    data: DataConfig
    dbscan: DBSCANConfig
    spectral: SpectralConfig


def load_config(filepath: str = "config.yaml") -> ExperimentConfig:
    """Nacte konfiguraci cviceni ze souboru YAML.

    Vraci typovanou instanci ``ExperimentConfig`` - pristup pres atributy
    (``cfg.dbscan.eps``) misto slovnikovych klicu (``cfg["dbscan"]["eps"]``).

    Pred vracenim je konfigurace overena funkci ``validate_config`` - pri
    neplatnych hodnotach je vyvolana vyjimka, takze volajici vzdy dostane
    bud platnou konfiguraci, nebo srozumitelnou chybu.

    Parameters
    ----------
    filepath:
        Cesta ke konfiguracnimu souboru YAML (vychozi: ``config.yaml``
        v pracovnim adresari).

    Returns
    -------
    ExperimentConfig
        Typovana a overena konfigurace cviceni.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    cfg = ExperimentConfig(
        data=DataConfig(**raw["data"]),
        dbscan=DBSCANConfig(**raw["dbscan"]),
        spectral=SpectralConfig(**raw["spectral"]),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    """Overi platnost konfigurace a vyvola vyjimku pri chybne hodnote.

    Overovane podminky:
        - ``cfg.data.dataset`` je jedno z podporovanych jmen
          (``circles``, ``moons``, ``blobs``)
        - ``cfg.data.n_samples >= 1``
        - ``cfg.dbscan.eps > 0`` (zaporny nebo nulovy polomer okoli nedava smysl)
        - ``cfg.dbscan.min_samples >= 1``
        - ``cfg.spectral.n_clusters >= 2`` (shlukovani s min. nez dvema shluky
          nedava smysl)
        - ``cfg.spectral.sigma > 0`` (sirka Gaussova jadra musi byt kladna)

    Tato funkce je volana z ``load_config`` - kazde nacteni konfigurace tak
    projde overenim a pipeline se nikdy nespusti s neplatnym vstupem. Jde
    o bezny obranny vzor: chranite i uzivatele, kteri nejsou pri upravach
    config.yaml dusledni.

    Parameters
    ----------
    cfg:
        Typovana konfigurace sestavena funkci ``load_config``.

    Raises
    ------
    ValueError
        Pokud kterakoli hodnota v konfiguraci nesplnuje uvedene podminky.
    """
    allowed_datasets: set[str] = {"circles", "moons", "blobs"}

    if cfg.data.dataset not in allowed_datasets:
        raise ValueError(
            f"Neznamy data.dataset: {cfg.data.dataset!r}. "
            f"Povolene hodnoty: {', '.join(allowed_datasets)}"
        )
    if cfg.data.n_samples < 1:
        raise ValueError(f"data.n_samples musi byt >= 1, dostali jsme {cfg.data.n_samples}")
    if cfg.dbscan.eps <= 0:
        raise ValueError(f"dbscan.eps musi byt > 0, dostali jsme {cfg.dbscan.eps}")
    if cfg.dbscan.min_samples < 1:
        raise ValueError(
            f"dbscan.min_samples musi byt >= 1, dostali jsme {cfg.dbscan.min_samples}"
        )
    if cfg.spectral.n_clusters < 2:
        raise ValueError(
            f"spectral.n_clusters musi byt >= 2, dostali jsme {cfg.spectral.n_clusters}"
        )
    if cfg.spectral.sigma <= 0:
        raise ValueError(f"spectral.sigma musi byt > 0, dostali jsme {cfg.spectral.sigma}")

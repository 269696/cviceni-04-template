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
    Pomocne vykreslovaci funkce pro vizualizaci dat a vysledku shlukovani.

    Kazda funkce ulozi vysledny graf jako PNG do adresare "graphs/" (pokud
    neni explicitne zadana jina cesta) a soucasne graf zobrazi pomoci
    plt.show(), aby jej bylo mozne prohlizet interaktivne pri spusteni
    hlavniho skriptu.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _ensure_graphs_dir(save_path: str) -> None:
    """Zajisti, ze adresar pro ulozeni grafu existuje."""
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def plot_clusters(
    x: np.ndarray,
    labels: np.ndarray,
    title: str = "Shluky",
    save_path: str | None = None,
) -> None:
    """
    Vykresli 2D bodovy graf dat obarveny podle prislusnosti do shluku.

    Body oznacene labelem -1 (konvence sklearn/DBSCAN pro sumove body,
    ktere nepatri do zadneho shluku) jsou vykresleny zvlast, malymi
    sedymi znacky, aby byly vizualne jasne odlisitelne od skutecnych
    shluku. Ostatni body jsou vykresleny ve druhem pruchodu a obarveny
    podle cisla shluku pomoci barevne mapy.

    Args:
        x: Pole tvaru (n_samples, 2) se souradnicemi bodu.
        labels: Pole tvaru (n_samples,) s cisly shluku pro kazdy bod;
            hodnota -1 oznacuje sumovy bod (bez shluku).
        title: Nadpis grafu.
        save_path: Cesta pro ulozeni PNG souboru. Pokud je None, pouzije
            se vychozi cesta "graphs/clusters.png".
    """
    if save_path is None:
        save_path = "graphs/clusters.png"
    _ensure_graphs_dir(save_path)

    noise_mask = labels == -1
    cluster_mask = ~noise_mask

    fig, ax = plt.subplots(figsize=(6, 6))

    # Sumove body - male sede znacky, vykresleny jako prvni (pod ostatnimi)
    if np.any(noise_mask):
        ax.scatter(
            x[noise_mask, 0],
            x[noise_mask, 1],
            s=15,
            c="lightgray",
            marker="x",
            label="sum (-1)",
        )

    # Skutecne shluky - obarvene podle cisla shluku
    if np.any(cluster_mask):
        _ = ax.scatter(
            x[cluster_mask, 0],
            x[cluster_mask, 1],
            s=25,
            c=labels[cluster_mask],
            cmap="tab10",
        )

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if np.any(noise_mask):
        ax.legend(loc="best")

    fig.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_k_distance(x: np.ndarray, k: int, save_path: str | None = None) -> None:
    """
    Vykresli tzv. k-distance graf pouzivany k odhadu parametru eps pro DBSCAN.

    Pro kazdy bod se spocita vzdalenost k jeho k-temu nejblizsimu sousedovi
    (pomoci sklearn.neighbors.NearestNeighbors), tyto vzdalenosti se
    seradi vzestupne a vykresli jako carovy graf. "Loket" (elbow) v grafu
    napovida vhodnou hodnotu parametru eps pro DBSCAN.

    Args:
        x: Pole tvaru (n_samples, n_features) se souradnicemi bodu.
        k: Pocet nejblizsich sousedu (typicky odpovida min_samples pro DBSCAN).
        save_path: Cesta pro ulozeni PNG souboru. Pokud je None, pouzije
            se vychozi cesta "graphs/k_distance.png".
    """
    if save_path is None:
        save_path = "graphs/k_distance.png"
    _ensure_graphs_dir(save_path)

    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(x)
    distances, _ = neighbors.kneighbors(x)

    # Vzdalenost k k-temu (poslednimu) sousedovi pro kazdy bod, serazena vzestupne
    k_distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_distances)
    ax.set_title(f"K-distance graf (k={k})")
    ax.set_xlabel("Body serazene podle k-te vzdalenosti")
    ax.set_ylabel(f"Vzdalenost k {k}. nejblizsimu sousedovi")

    fig.savefig(save_path)
    plt.show()
    plt.close(fig)


def plot_eigenvalues(eigenvalues: np.ndarray, save_path: str | None = None) -> None:
    """
    Vykresli serazena vlastni cisla Laplacianu grafu pro odhad poctu shluku.

    Pouziva se pri spektralnim shlukovani k vizualizaci tzv. eigengap -
    vyrazneho skoku mezi po sobe jdoucimi vlastnimi cisly, jehoz poloha
    napovida vhodny pocet shluku.

    Args:
        eigenvalues: Pole vlastnich cisel Laplacianu grafu.
        save_path: Cesta pro ulozeni PNG souboru. Pokud je None, pouzije
            se vychozi cesta "graphs/eigenvalues.png".
    """
    if save_path is None:
        save_path = "graphs/eigenvalues.png"
    _ensure_graphs_dir(save_path)

    sorted_eigenvalues = np.sort(eigenvalues)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sorted_eigenvalues, marker="o")
    ax.set_title("Vlastni cisla Laplacianu grafu")
    ax.set_xlabel("Index (serazeno vzestupne)")
    ax.set_ylabel("Vlastni cislo")

    fig.savefig(save_path)
    plt.show()
    plt.close(fig)

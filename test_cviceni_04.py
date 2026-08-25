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
    Kouřové (smoke) testy pro Cvičení 04 - DBSCAN a spektrální shlukování.

    Testy cíleně ověřují vlastní implementace tříd src.dbscan.DBSCAN a
    src.spectral.SpectralClustering, nikoli src.distance.py - k tomu slouží
    níže definovaná pomocná třída DummyDistance, která poskytuje funkční
    eukleidovskou vzdálenost bez ohledu na to, zda student už dokončil
    src/distance.py. Testy tedy izolovaně cvičí právě metody fit,
    _region_query, _affinity_matrix, _laplacian a _spectral_embedding.

    Protože jsou tyto metody v aktuálním stavu repozitáře nedokončené
    studentské úkoly (vyvolávají NotImplementedError), jsou příslušné testy
    označeny dekorátorem @pytest.mark.xfail(raises=NotImplementedError,
    strict=False) - dokud úkoly nejsou hotové, testy se hlásí jako "xfail"
    (očekávané selhání) a celá sada testů proto skončí s návratovým kódem
    0. Jakmile student implementaci doplní, testy začnou procházet - díky
    strict=False se to projeví jako "xpass" (neočekávaný úspěch), a NEBUDE
    to považováno za selhání sady testů - to je zde žádoucí chování.

    Testy záměrně NEporovnávají syrové (raw) štítky shluků s referenčními
    štítky ze sklearn pomocí == - číslování shluků je arbitrární
    (permutace číslování shluků nemá žádný sémantický význam), a proto se
    místo toho používá sklearn.metrics.adjusted_rand_score, která je vůči
    přeznačení shluků invariantní. Pozor: nejde o studentskou (bonusovou)
    implementaci src.metrics.adjusted_rand_index - ta je zde záměrně
    nepoužita, aby testy nezávisely na dalším nepovinném úkolu.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import adjusted_rand_score

from src.dbscan import DBSCAN
from src.spectral import SpectralClustering


class DummyDistance:
    """
    Jednoduchá, vždy funkční eukleidovská vzdálenost pro účely testů.

    Nededí od src.distance.Distance (kachní typování stačí - DBSCAN i
    SpectralClustering volají pouze self.distance.calculate(x, y)).
    Účelem je oddělit testy DBSCAN/SpectralClustering od toho, zda
    student už dokončil samostatný úkol v src/distance.py.
    """

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Vrátí eukleidovskou (L2) vzdálenost mezi vektory x a y."""
        return float(np.linalg.norm(x - y))


# Parametry níže byly nalezeny laděním pomocné (nekomitované) referenční
# implementace DBSCAN a spektrálního shlukování vůči těmto konkrétním
# datovým sadám - viz zadání práce; hodnoty dávají čistě oddělené shluky
# (ARI blízké/rovné 1.0 vůči sklearn/ground-truth).
BLOBS_N_SAMPLES = 150
BLOBS_CENTERS = 3
BLOBS_CLUSTER_STD = 1.0
BLOBS_RANDOM_STATE = 42
DBSCAN_EPS = 0.8
DBSCAN_MIN_SAMPLES = 5

CIRCLES_N_SAMPLES = 200
CIRCLES_NOISE = 0.05
CIRCLES_FACTOR = 0.5
CIRCLES_RANDOM_STATE = 42
SPECTRAL_N_CLUSTERS = 2
SPECTRAL_SIGMA = 0.1


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason="DBSCAN.fit/_region_query je zatím nedokončený studentský úkol",
    strict=False,
)
def test_dbscan_matches_sklearn_on_blobs() -> None:
    """DBSCAN na dobře oddělených blobs se má shodovat se sklearn.cluster.DBSCAN."""
    x, _ = make_blobs(
        n_samples=BLOBS_N_SAMPLES,
        centers=BLOBS_CENTERS,
        cluster_std=BLOBS_CLUSTER_STD,
        random_state=BLOBS_RANDOM_STATE,
    )

    model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, distance=DummyDistance())
    model.fit(x)
    my_labels = model.labels_

    sklearn_model = SklearnDBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    sklearn_labels = sklearn_model.fit_predict(x)

    ari = adjusted_rand_score(sklearn_labels, my_labels)
    assert ari >= 0.99, f"Očekávána shoda s sklearn (ARI >= 0.99), vyšlo {ari:.4f}"

    # Smysluplná kontrola šumu: obě implementace by měly označit za šum
    # přibližně stejný počet bodů (ne nutně identické indexy, ale
    # srovnatelný rozsah) - u tak dobře oddělených blobs by měl počet
    # šumových bodů být malý (řádově jednotky procent) a shodný mezi
    # oběma implementacemi.
    my_noise_count = int(np.sum(my_labels == -1))
    sklearn_noise_count = int(np.sum(sklearn_labels == -1))
    assert my_noise_count == sklearn_noise_count, (
        f"Počet šumových bodů se liší: vlastní implementace={my_noise_count}, "
        f"sklearn={sklearn_noise_count}"
    )


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason=(
        "SpectralClustering.fit/_affinity_matrix/_laplacian/_spectral_embedding "
        "je zatím nedokončený studentský úkol"
    ),
    strict=False,
)
def test_spectral_clustering_separates_circles() -> None:
    """Spektrální shlukování má dobře oddělit dvě soustředné kružnice."""
    x, y_true = make_circles(
        n_samples=CIRCLES_N_SAMPLES,
        noise=CIRCLES_NOISE,
        factor=CIRCLES_FACTOR,
        random_state=CIRCLES_RANDOM_STATE,
    )

    model = SpectralClustering(
        n_clusters=SPECTRAL_N_CLUSTERS,
        sigma=SPECTRAL_SIGMA,
        distance=DummyDistance(),
    )
    model.fit(x)

    ari = adjusted_rand_score(y_true, model.labels_)
    assert ari >= 0.9, f"Očekávána dobrá shoda s ground-truth (ARI >= 0.9), vyšlo {ari:.4f}"


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason="SpectralClustering._affinity_matrix je zatím nedokončený studentský úkol",
    strict=False,
)
def test_affinity_matrix_is_symmetric_with_unit_diagonal() -> None:
    """Afinitní matice musí být symetrická a mít diagonálu (přibližně) rovnou 1."""
    x, _ = make_circles(
        n_samples=CIRCLES_N_SAMPLES,
        noise=CIRCLES_NOISE,
        factor=CIRCLES_FACTOR,
        random_state=CIRCLES_RANDOM_STATE,
    )

    model = SpectralClustering(
        n_clusters=SPECTRAL_N_CLUSTERS,
        sigma=SPECTRAL_SIGMA,
        distance=DummyDistance(),
    )
    w = model._affinity_matrix(x)

    assert np.allclose(w, w.T), "Afinitní matice W musí být symetrická"
    assert np.allclose(np.diag(w), 1.0), "Diagonála afinitní matice W musí být přibližně 1.0"


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason=(
        "SpectralClustering._affinity_matrix/_laplacian je zatím nedokončený "
        "studentský úkol"
    ),
    strict=False,
)
def test_laplacian_rows_sum_to_zero() -> None:
    """Řádky nenormalizovaného Laplaciánu L = D - W musí sčítat (přibližně) na 0."""
    x, _ = make_circles(
        n_samples=CIRCLES_N_SAMPLES,
        noise=CIRCLES_NOISE,
        factor=CIRCLES_FACTOR,
        random_state=CIRCLES_RANDOM_STATE,
    )

    model = SpectralClustering(
        n_clusters=SPECTRAL_N_CLUSTERS,
        sigma=SPECTRAL_SIGMA,
        distance=DummyDistance(),
    )
    w = model._affinity_matrix(x)
    l = model._laplacian(w)

    row_sums = l.sum(axis=1)
    assert np.allclose(row_sums, 0.0, atol=1e-8), (
        f"Řádky Laplaciánu musí sčítat na 0, maximální odchylka: "
        f"{np.max(np.abs(row_sums)):.6g}"
    )

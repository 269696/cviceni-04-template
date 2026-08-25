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
    Cviceni 04 - Pokrocile shlukovani: DBSCAN a spektralni shlukovani.

    Tento skript je hlavni vstupni bod cviceni. Postupne:

    1. Nacte konfiguraci ze souboru config.yaml a z ni vygeneruje syntetickou
       2D datovou sadu (kruznice / pulmesice / bloby).
    2. Vykresli referencni ("spravne") rozdeleni na zaklade skutecnych
       (ground-truth) labelu.
    3. Vykresli k-distance graf, ktery pomaha odhadnout vhodnou hodnotu
       parametru eps pro DBSCAN.
    4. Spusti DBSCAN a vykresli jeho vysledne shluky.
    5. Spusti spektralni shlukovani a vykresli jeho vysledne shluky spolu
       s grafem vlastnich cisel Laplacianu (pro odhad poctu shluku).
    6. Pokud oba algoritmy uspesne dobehly, provede jejich strucne
       porovnani vcetne (bonusove) metriky Adjusted Rand Index (ARI).

    Jednotlive faze jsou zamerne obaleny do samostatnych bloku
    try/except NotImplementedError - tridy src/dbscan.py, src/spectral.py,
    src/distance.py a src/metrics.py jsou v aktualnim stavu repozitare
    studentske ukoly (jejich metody vyvolavaji NotImplementedError). Diky
    tomuto obaleni selhani jedne faze (napr. DBSCAN jeste neni hotovy)
    nezabrani spusteni a vyhodnoceni ostatnich faz - student tak dostane
    co nejvice zpetne vazby o tom, ktere casti je jeste potreba doplnit.

    Pozor: v tomto skriptu se NIKDY nevola metoda predict() - DBSCAN i
    spektralni shlukovani jsou transduktivni algoritmy a jejich predict()
    je zamerne, natrvalo nedokoncena (vyvola NotImplementedError, ktery
    ale NEZACINA retezcem "Ukol:" - jde o zamerne architektonicke
    rozhodnuti, nikoli o studentsky ukol). V pipeline se pracuje pouze s
    vysledkem fit() ulozenym v atributu .labels_.
"""

from __future__ import annotations

from dataio.config_manager import load_config
from dataio.datasets import make_dataset
from dataio.plotting import plot_clusters, plot_eigenvalues, plot_k_distance
from src.dbscan import DBSCAN
from src.distance import EuclideanDistance
from src.metrics import adjusted_rand_index
from src.spectral import SpectralClustering


def _print_stub_error(stage_name: str, error: NotImplementedError) -> None:
    """
    Vypise srozumitelnou hlasku o tom, ze dana faze pipeline selhala na
    NotImplementedError, a rozlisi, o jaky druh vyjimky jde.

    V tomto cvicebnim repozitari existuji dva zcela odlisne druhy
    NotImplementedError:

    1. Studentske ukoly - zprava zacina retezcem "Ukol:". Jde o
       nedokoncenou implementaci, kterou ma student doplnit.
    2. Zamerne/trvale chovani (transduktivni predict()) - zprava
       "Ukol:" NEzacina. Toto NENI ukol k doplneni, ale zamerne
       architektonicke rozhodnuti, ktere se vysvetluje studentovi.

    V teto pipeline se predict() nikdy nevola, takze prakticky vzdy
    pujde o prvni pripad (nedokoncena faze fit/_region_query/
    _affinity_matrix/_laplacian/_spectral_embedding/calculate). Funkce
    nicmene rozlisuje oba pripady obecne, pro srozumitelnost a
    odolnost vuci budoucim zmenam.

    Args:
        stage_name: Nazev faze pipeline, ve ktere vyjimka nastala
            (pro vypis).
        error: Zachycena vyjimka NotImplementedError.
    """
    message = str(error)
    if message.startswith("Úkol:") or message.startswith("Ukol:"):
        kind = "toto je ukol pro studenta (Uloha jeste neni implementovana)"
    else:
        kind = "toto je zamerne/trvale chovani, ne ukol"
    print(f"[{stage_name}] Faze se nezdarila: {message}")
    print(f"[{stage_name}] Vysvetleni: {kind}.")


def main() -> None:
    """
    Spusti celou pipeline cviceni 04 od nacteni konfigurace az po
    zaverecne porovnani DBSCAN a spektralniho shlukovani.

    Returns:
        None
    """
    # 1. Nacteni konfigurace a vygenerovani datove sady (bez ukolu).
    cfg = load_config()
    x, y_true = make_dataset(cfg.data)

    # 2. Referencni graf skutecneho rozdeleni.
    plot_clusters(x, y_true, "Skutečné rozdělení")

    # 3. K-distance graf pro odhad parametru eps.
    plot_k_distance(x, cfg.dbscan.min_samples)

    # 4. DBSCAN.
    dbscan_model: DBSCAN | None = None
    try:
        dbscan_model = DBSCAN(
            eps=cfg.dbscan.eps,
            min_samples=cfg.dbscan.min_samples,
            distance=EuclideanDistance(),
        )
        dbscan_model.fit(x)
        plot_clusters(x, dbscan_model.labels_, "DBSCAN")
    except NotImplementedError as e:
        _print_stub_error("DBSCAN", e)
        dbscan_model = None

    # 5. Spektralni shlukovani.
    spectral_model: SpectralClustering | None = None
    try:
        spectral_model = SpectralClustering(
            n_clusters=cfg.spectral.n_clusters,
            sigma=cfg.spectral.sigma,
            distance=EuclideanDistance(),
        )
        spectral_model.fit(x)
        plot_clusters(x, spectral_model.labels_, "Spektrální shlukování")
        plot_eigenvalues(spectral_model.eigenvalues_)
    except NotImplementedError as e:
        _print_stub_error("Spektrální shlukování", e)
        spectral_model = None

    # 6. Zaverecne porovnani.
    if dbscan_model is not None and spectral_model is not None:
        print(
            "Porovnání: DBSCAN nalezl shluky na základě hustoty bodů "
            "(a umí označit šum), spektrální shlukování využívá strukturu "
            "grafu podobnosti přes vlastní čísla Laplaciánu a k-means "
            "v prostoru embeddingu."
        )
        try:
            ari_dbscan = adjusted_rand_index(y_true, dbscan_model.labels_)
            ari_spectral = adjusted_rand_index(y_true, spectral_model.labels_)
            print(f"Adjusted Rand Index (DBSCAN):               {ari_dbscan:.4f}")
            print(f"Adjusted Rand Index (spektrální shlukování): {ari_spectral:.4f}")
        except NotImplementedError as e:
            if str(e).startswith("Úkol:") or str(e).startswith("Ukol:"):
                print(
                    "(bonus metrika ARI zatím není implementována - "
                    "viz src/metrics.py, funkce adjusted_rand_index)"
                )
            else:
                _print_stub_error("Adjusted Rand Index", e)
    else:
        print(
            "Porovnání přeskočeno, DBSCAN/spektrální shlukování zatím "
            "není dokončeno."
        )


if __name__ == "__main__":
    main()

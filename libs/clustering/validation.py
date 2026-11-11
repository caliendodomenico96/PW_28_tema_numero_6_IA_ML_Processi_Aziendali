# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Validazione separata, su modelli addestrati e salvati (nessun refit)."""


import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .configuration import KMeansConfig
from .preprocessing import build_feature_frame, load_customer_data
from .reporting import plot_silhouette_detail, score_partition
from .storage import json_read, json_write, resolve_training_run, sha256_file, unique_utc_dir


def validate_kmeans(project_root: Path, cfg: KMeansConfig) -> Path:
    training_dir = resolve_training_run(cfg, project_root)
    report = json_read(training_dir / "training_report.json")
    source = Path(report["dataset_csv"])
    if not source.is_absolute():
        source = Path(project_root) / source
    if not source.is_file():
        raise FileNotFoundError(f"Il dataset usato dal training non è più disponibile: {source}")
    if sha256_file(source) != report["dataset_sha256"]:
        raise ValueError("CSV modificato dopo il training: validazione interrotta")
    if sha256_file(source.with_suffix(".metadati.json")) != report["metadati_sha256"]:
        raise ValueError("Metadati modificati dopo il training: validazione interrotta")

    df, _, reference = load_customer_data(source)
    features, _ = build_feature_frame(df, reference, tuple(report["features"]), report["log1p_spesa"])
    preprocessor = joblib.load(training_dir / "models" / "preprocessor.joblib")
    x = preprocessor.transform(features)
    if not np.isfinite(x).all():
        raise ValueError("Dati trasformati non finiti")
    compare_k = list(report["k_confronto"])
    if len(compare_k) < 2:
        raise ValueError("Sono richieste almeno due configurazioni di confronto")
    validations = []
    labels_by_k = {}
    for k in compare_k:
        model = joblib.load(training_dir / "models" / f"kmeans_k{k}.joblib")
        labels = model.predict(x)
        predictions = pd.read_csv(training_dir / f"cluster_k{k}.csv", dtype={"id_cliente": "string"})
        if list(predictions["id_cliente"]) != list(df["id_cliente"]):
            raise ValueError(f"Ordine o ID dei clienti diversi dall'addestramento: K={k}")
        if not np.array_equal(labels, predictions["cluster"].to_numpy()):
            raise ValueError(f"Assegnazioni non riproducibili col modello salvato: K={k}")
        metric = score_partition(x, labels, inertia=float(model.inertia_))
        labels_by_k[k] = labels
        validations.append(metric)
        logging.info("Validazione K=%d: silhouette=%.4f; Davies-Bouldin=%.4f",
                    k, metric["silhouette"], metric["davies_bouldin"])

    validation_dir = unique_utc_dir(training_dir / "validations")
    for k in compare_k:
        plot_silhouette_detail(x, labels_by_k[k], validation_dir / f"silhouette_k{k}.png")

    pd.DataFrame(validations).drop(columns=["dimensioni_cluster"]).to_csv(
        validation_dir / "confronto_k.csv", index=False, encoding="utf-8")
    better = max(validations, key=lambda metric: (metric["silhouette"], -metric["k"]))
    cross = None
    if len(compare_k) == 2:
        left, right = compare_k
        cross = {
            "k_a": left, "k_b": right,
            "adjusted_rand_index": round(float(adjusted_rand_score(labels_by_k[left], labels_by_k[right])), 6),
            "nota": "Accordo tra due partizioni, non accuratezza rispetto a etichette reali.",
        }
    # Percorso portabile per i training nel progetto; i test possono usare
    # directory temporanee esterne alla root.
    try:
        training_reference = str(training_dir.resolve().relative_to(Path(project_root).resolve()))
    except ValueError:
        training_reference = str(training_dir.resolve())

    json_write(validation_dir / "validation_report.json", {
        "training_dir": training_reference,
        "dataset_sha256_verificato": True,
        "preprocessing_riprodotto": True,
        "assegnazioni_modelli_verificate": True,
        "n_clienti": len(df),
        "configurazioni_confrontate": compare_k,
        "metriche": validations,
        "k_con_silhouette_maggiore_fra_quelli_confrontati": better["k"],
        "scelta_k_training": report["scelta_k"],
        "accordo_fra_partizioni": cross,
        "grafici_validation": [f"silhouette_k{k}.png" for k in compare_k],
        "grafici_descrittivi": "I grafici gomito, silhouette media, heatmap, PCA, boxplot e profili dei cluster sono conservati nella cartella del training.",
        "note": [
            "Silhouette e indici interni sono valutati sugli stessi dati di training: non è una validazione esterna.",
            "Silhouette: maggiore è meglio, intervallo da -1 a 1.",
            "Davies-Bouldin: minore è meglio; Calinski-Harabasz: maggiore è meglio.",
            "Non assumere che K=4 equivalga ai quattro comportamenti esplicitati dalla traccia.",
            "La caratteristica cliente recente può sovrapporsi alle fasce di attività.",
            "Le percentuali descrittive dei cluster non sono le etichette di training.",
        ],
    })
    logging.info("Validazione salvata in %s", validation_dir)
    return validation_dir
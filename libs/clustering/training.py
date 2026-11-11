# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Orchestrazione del training K-Means, senza logica nel main."""


import logging
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from .configuration import KMeansConfig
from .preprocessing import build_feature_frame, create_preprocessor, load_customer_data
from .reporting import (
    cluster_profiles,
    knee_by_distance,
    plot_centroid_heatmap,
    plot_cluster_boxplots,
    plot_k_metrics,
    plot_pca_clusters,
    plot_silhouette_comparison,
    recommended_k,
    score_partition,
)
from .storage import json_write, resolve_dataset, sha256_file, unique_utc_dir


def train_kmeans(project_root: Path, cfg: KMeansConfig) -> Path:
    csv_path = resolve_dataset(project_root, cfg.dataset_csv)
    df, metadata, reference = load_customer_data(csv_path)
    raw_features, quality_report = build_feature_frame(df, reference, cfg.features, cfg.log1p_spesa)
    preprocessor = create_preprocessor()
    x = preprocessor.fit_transform(raw_features)
    if not np.isfinite(x).all():
        raise ValueError("Preprocessing fallito: esistono valori non finiti")
    unique_points = np.unique(x, axis=0).shape[0]
    if cfg.k_max >= min(len(df), unique_points):
        raise ValueError(f"k_max={cfg.k_max} troppo grande: clienti={len(df)}, punti distinti={unique_points}")

    runs = {}
    metrics = []
    for k in range(cfg.k_min, cfg.k_max + 1):
        model = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=cfg.n_init)
        labels = model.fit_predict(x)
        metric = score_partition(x, labels, inertia=float(model.inertia_))
        metrics.append(metric)
        runs[k] = (model, labels)
        logging.info("K=%d | silhouette=%.4f | inertia=%.2f | min_cluster=%d",
                    k, metric["silhouette"], metric["inertia"], metric["cluster_minimo"])

    elbow = knee_by_distance(metrics)
    suggested, motivation = recommended_k(metrics, elbow)
    saved_ks = sorted(set(cfg.k_confronto) | {suggested, elbow})
    output = unique_utc_dir(cfg.output_dir)
    (output / "models").mkdir(exist_ok=False)
    quality_report.update({
        "data_riferimento": reference.strftime("%d/%m/%Y"),
        "mediane_usate_per_imputazione": {
            feature: round(float(preprocessor.named_steps["imputer"].statistics_[i]), 6)
            for i, feature in enumerate(cfg.features)
        },
        "metodo_scaling": "StandardScaler (z-score)",
        "colonne_escluse_dal_clustering": [c for c in df.columns if c not in cfg.features],
        "regola_valori_mancanti": "Errori/negativi -> NaN; mediana per ciascuna feature; oltre 25% -> errore",
    })
    json_write(output / "preprocessing_report.json", quality_report)
    joblib.dump(preprocessor, output / "models" / "preprocessor.joblib")
    pd.DataFrame(x, columns=cfg.features).assign(id_cliente=df["id_cliente"].values).to_csv(
        output / "features_scalate.csv", index=False, encoding="utf-8")
    pd.DataFrame(metrics).drop(columns=["dimensioni_cluster"]).to_csv(
        output / "metriche_k.csv", index=False)
    plot_k_metrics(metrics, elbow, output / "grafico_gomito.png")
    plot_silhouette_comparison(metrics, suggested, output / "grafico_silhouette.png")

    for k in saved_ks:
        model, labels = runs[k]
        joblib.dump(model, output / "models" / f"kmeans_k{k}.joblib")
        predictions = df.copy()
        predictions["cluster"] = labels.astype(int)
        predictions.to_csv(output / f"cluster_k{k}.csv", index=False, encoding="utf-8")
        json_write(output / f"profili_cluster_k{k}.json", cluster_profiles(df, labels, reference, metadata))
        plot_centroid_heatmap(model.cluster_centers_, list(cfg.features), output / f"heatmap_centroidi_k{k}.png",
                              title=f"Heatmap dei centroidi - K = {k}")
        plot_pca_clusters(x, labels, model.cluster_centers_, output / f"scatter_pca_k{k}.png",
                          title=f"PCA 2D dei cluster - K = {k}")
        plot_cluster_boxplots(df, labels, reference, list(cfg.features), output / f"boxplot_feature_k{k}.png",
                              title_prefix=f"K = {k}")

    try:
        dataset_reference = str(csv_path.relative_to(Path(project_root).resolve()))
    except ValueError:
        dataset_reference = str(csv_path)
    report = {
        "versione_pipeline": "1.0.0",
        "dataset_csv": dataset_reference,
        "dataset_sha256": sha256_file(csv_path),
        "metadati_sha256": sha256_file(csv_path.with_suffix(".metadati.json")),
        "n_clienti": len(df),
        "data_riferimento": reference.strftime("%d/%m/%Y"),
        "features": list(cfg.features),
        "log1p_spesa": cfg.log1p_spesa,
        "preprocessing": "SimpleImputer(median) + StandardScaler",
        "k_analizzati": list(range(cfg.k_min, cfg.k_max + 1)),
        "k_confronto": list(cfg.k_confronto),
        "k_modelli_salvati": saved_ks,
        "random_state": cfg.random_state,
        "n_init": cfg.n_init,
        "scelta_k": motivation,
        "metriche": metrics,
        "versioni": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                      "pandas": pd.__version__, "numpy": np.__version__},
        "artefatti_aggiuntivi": {
            "heatmap_centroidi": [f"heatmap_centroidi_k{k}.png" for k in saved_ks],
            "scatter_pca": [f"scatter_pca_k{k}.png" for k in saved_ks],
            "boxplot_feature": [f"boxplot_feature_k{k}.png" for k in saved_ks],
        },
        "note": [
            "Training non supervisionato: nessuna etichetta latente è usata.",
            "Cluster 0/1/... sono identificatori arbitrari, non etichette comportamentali.",
            "Il metodo del gomito e la silhouette sono euristiche; non determinano il K vero.",
            "Le metriche interne sono calcolate sui dati usati per il fitting, non su un test indipendente.",
            "La data d'iscrizione è trasformata in anzianità rispetto alla data di riferimento del dataset.",
        ],
    }
    # Scrivere LAST: identifica un training completato per lo script di validation.
    json_write(output / "training_report.json", report)
    logging.info("Training completato: %s", output)
    logging.info("Gomito K=%d | migliore silhouette K=%d | raccomandato K=%d",
                elbow, motivation["k_massimo_silhouette"], suggested)
    return output
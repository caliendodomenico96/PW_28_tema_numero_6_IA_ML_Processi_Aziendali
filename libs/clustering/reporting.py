# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Statistiche di clustering e visualizzazioni degli esperimenti."""


from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score


def score_partition(x: np.ndarray, labels: np.ndarray, inertia: float | None = None) -> dict:
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    n_clusters = len(cluster_sizes)
    if not 2 <= n_clusters < len(labels):
        raise ValueError("Silhouette richiede da 2 a n_clienti-1 cluster non vuoti")
    result = {
        "k": int(n_clusters),
        "silhouette": round(float(silhouette_score(x, labels)), 6),
        "davies_bouldin": round(float(davies_bouldin_score(x, labels)), 6),
        "calinski_harabasz": round(float(calinski_harabasz_score(x, labels)), 6),
        "dimensioni_cluster": {str(int(k)): int(v) for k, v in cluster_sizes.items()},
        "cluster_minimo": int(cluster_sizes.min()),
        "cluster_massimo": int(cluster_sizes.max()),
    }
    if inertia is not None:
        result["inertia"] = round(float(inertia), 6)
    return result


def knee_by_distance(metrics: list[dict]) -> int:
    """Gomito: K con distanza massima dalla retta che unisce gli estremi."""
    ordered = sorted(metrics, key=lambda x: x["k"])
    coords = np.asarray([(m["k"], m["inertia"]) for m in ordered], dtype=float)
    mins = coords.min(axis=0)
    spans = np.ptp(coords, axis=0)
    spans[spans == 0] = 1
    coords = (coords - mins) / spans
    v = coords[-1] - coords[0]
    norm = np.linalg.norm(v)
    if norm == 0:
        return int(ordered[0]["k"])
    distances = np.abs(v[0] * (coords[:, 1] - coords[0, 1]) - v[1] * (coords[:, 0] - coords[0, 0])) / norm
    distances[0] = distances[-1] = -1
    return int(ordered[int(np.argmax(distances))]["k"])


def recommended_k(metrics: list[dict], elbow: int) -> tuple[int, dict]:
    """Massimizza silhouette; se il gomito è distante al massimo 0.02, preferisce il gomito."""
    scores = {m["k"]: m["silhouette"] for m in metrics}
    best_k = max(scores, key=lambda k: (scores[k], -k))
    epsilon = 0.02
    if scores[elbow] >= scores[best_k] - epsilon:
        chosen = elbow
        rule = "gomito entro 0,02 punti di silhouette dal miglior valore"
    else:
        chosen = best_k
        rule = "massimo silhouette; il gomito è inferiore di oltre 0,02 punti"
    return chosen, {
        "k_massimo_silhouette": best_k,
        "k_gomito": elbow,
        "k_raccomandato": chosen,
        "regola": rule,
        "soglia_tolleranza_silhouette": epsilon,
        "nota": "Indicazione euristica: il valore finale richiede anche interpretazione dei gruppi.",
    }


def plot_k_metrics(metrics: list[dict], selected: int, path: Path) -> None:
    ordered = sorted(metrics, key=lambda x: x["k"])
    ks = [m["k"] for m in ordered]
    values = [m["inertia"] for m in ordered]
    plt.figure(figsize=(8, 4.5))
    plt.plot(ks, values, marker="o")
    plt.axvline(selected, linestyle="--", label=f"K suggerito = {selected}")
    plt.title("Metodo del gomito - K-Means")
    plt.xlabel("Numero di cluster (K)")
    plt.ylabel("Inertia (somma delle distanze quadratiche)")
    plt.xticks(ks)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_silhouette_comparison(metrics: list[dict], selected: int, path: Path) -> None:
    ordered = sorted(metrics, key=lambda x: x["k"])
    ks = [m["k"] for m in ordered]
    plt.figure(figsize=(8, 4.5))
    plt.plot(ks, [m["silhouette"] for m in ordered], marker="o")
    plt.axvline(selected, linestyle="--", label=f"K suggerito = {selected}")
    plt.title("Silhouette media per K")
    plt.xlabel("Numero di cluster (K)")
    plt.ylabel("Silhouette media (più alto è meglio)")
    plt.xticks(ks)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_silhouette_detail(x: np.ndarray, labels: np.ndarray, path: Path) -> None:
    vals = silhouette_samples(x, labels)
    k = len(set(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ylower = 10
    cmap = plt.get_cmap("tab10")
    for cluster in sorted(set(labels)):
        sample_values = np.sort(vals[labels == cluster])
        yupper = ylower + len(sample_values)
        ax.fill_betweenx(np.arange(ylower, yupper), 0, sample_values,
                         facecolor=cmap(int(cluster) % 10), alpha=0.7)
        ax.text(-0.1, ylower + len(sample_values) / 2, str(cluster), fontsize=9)
        ylower = yupper + 10
    ax.axvline(float(vals.mean()), linestyle="--", label=f"Media = {vals.mean():.3f}")
    ax.set_title(f"Silhouette per cluster - K = {k}")
    ax.set_xlabel("Coefficiente silhouette")
    ax.set_ylabel("Clienti raggruppati per cluster")
    ax.set_xlim(min(-0.2, float(vals.min()) - 0.05), 1)
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_centroid_heatmap(centers: np.ndarray, feature_names: list[str] | tuple[str, ...], path: Path, title: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(feature_names) * 1.5), max(3.2, len(centers) * 0.8 + 1.5)))
    vmax = float(np.max(np.abs(centers))) if centers.size else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(centers, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(feature_names)), labels=feature_names, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(centers)), labels=[f"Cluster {i}" for i in range(len(centers))])
    for row in range(centers.shape[0]):
        for col in range(centers.shape[1]):
            value = float(centers[row, col])
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9, color="black")
    ax.set_title(title or "Heatmap dei centroidi (feature standardizzate)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("z-score del centroide")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pca_clusters(x: np.ndarray, labels: np.ndarray, centers: np.ndarray, path: Path, title: str | None = None) -> None:
    if x.shape[1] < 2:
        raise ValueError("La proiezione PCA richiede almeno 2 feature")
    pca = PCA(n_components=2, random_state=0)
    x_2d = pca.fit_transform(x)
    centers_2d = pca.transform(centers)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for cluster in sorted(set(labels)):
        mask = labels == cluster
        ax.scatter(x_2d[mask, 0], x_2d[mask, 1], s=24, alpha=0.7, color=cmap(int(cluster) % 10), label=f"Cluster {cluster}")
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], marker="X", s=220, color="black", edgecolor="white", linewidth=1.0, label="Centroidi")
    ratio = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ratio[0]:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1f}% var.)")
    ax.set_title(title or "Proiezione PCA 2D dei clienti e dei centroidi")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _boxplot_frame(df: pd.DataFrame, labels: np.ndarray, reference: pd.Timestamp, feature_names: tuple[str, ...] | list[str]) -> pd.DataFrame:
    tmp = pd.DataFrame(index=df.index)
    tmp["cluster"] = labels.astype(int)
    for feature in feature_names:
        if feature == "giorni_da_iscrizione":
            dates = pd.to_datetime(df["data_iscrizione"], format="%d/%m/%Y", errors="coerce")
            tmp[feature] = (reference - dates).dt.days.astype("float64")
        else:
            tmp[feature] = pd.to_numeric(df[feature].astype("string").str.replace(",", ".", regex=False), errors="coerce")
    return tmp


def plot_cluster_boxplots(df: pd.DataFrame, labels: np.ndarray, reference: pd.Timestamp, feature_names: tuple[str, ...] | list[str], path: Path, title_prefix: str | None = None) -> None:
    data = _boxplot_frame(df, labels, reference, feature_names)
    features = list(feature_names)
    n = len(features)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for idx, feature in enumerate(features):
        ax = axes[idx // cols, idx % cols]
        grouped = [data.loc[data["cluster"] == cluster, feature].dropna().to_numpy() for cluster in sorted(data["cluster"].unique())]
        labels_x = [f"C{cluster}" for cluster in sorted(data["cluster"].unique())]
        ax.boxplot(grouped, tick_labels=labels_x)
        ax.set_title(f"{title_prefix + ' - ' if title_prefix else ''}{feature}")
        ax.set_xlabel("Cluster")
        ax.set_ylabel(feature)
        ax.grid(axis="y", alpha=0.2)
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray, reference: pd.Timestamp, metadata: dict) -> dict:
    """Analisi descrittiva post-hoc: non modifica gli assegnamenti."""
    tmp = df.copy()
    tmp["cluster"] = labels
    recent_days = int(metadata.get("configurazione", {}).get("giorni_cliente_nuovo", 30))
    very_active_min = int(metadata.get("configurazione", {}).get("richieste_min_molto_attivi", 6))
    inscription = pd.to_datetime(tmp["data_iscrizione"], format="%d/%m/%Y", errors="coerce")
    tmp["giorni_da_iscrizione"] = (reference - inscription).dt.days
    for feature in ("numero_richieste", "importo_totale_speso", "importo_medio_speso", "giorni_ultima_richiesta"):
        tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
    result = {}
    for cluster, subset in tmp.groupby("cluster", sort=True):
        frequency = subset["numero_richieste"]
        mode_city = subset["zona"].mode() if "zona" in subset else pd.Series(dtype=str)
        mode_cat = subset["categoria_preferita"].mode() if "categoria_preferita" in subset else pd.Series(dtype=str)
        result[str(int(cluster))] = {
            "clienti": len(subset),
            "percentuale_clienti": round(len(subset) / len(tmp) * 100, 2),
            "media_richieste": round(float(frequency.mean()), 2),
            "media_spesa_euro": round(float(subset["importo_totale_speso"].mean()), 2),
            "mediana_spesa_euro": round(float(subset["importo_totale_speso"].median()), 2),
            "media_recency_giorni": round(float(subset["giorni_ultima_richiesta"].mean()), 2),
            "media_anzianita_giorni": round(float(subset["giorni_da_iscrizione"].mean()), 2),
            "percentuale_inattivi": round(float((frequency == 0).mean() * 100), 2),
            "percentuale_molto_attivi": round(float((frequency >= very_active_min).mean() * 100), 2),
            "percentuale_recenti": round(float((subset["giorni_da_iscrizione"] < recent_days).mean() * 100), 2),
            "zona_prevalente": str(mode_city.iloc[0]) if len(mode_city) else None,
            "categoria_preferita_prevalente": str(mode_cat.iloc[0]) if len(mode_cat) else None,
        }
    return result
# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Grafici di confronto dei nuovi clienti, senza alterare dataset e cluster storici."""


from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from sklearn.decomposition import PCA

from libs.clustering.preprocessing import build_feature_frame
from libs.dashboard.inference import PredictionContext


COLORS = qualitative.Plotly  # Rende coerenti i colori dei cluster nei due grafici.


def _pca_coordinates(
    historical: pd.DataFrame,
    new: pd.DataFrame,
    context: PredictionContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """PCA sulla sola matrice del training, proiettando poi il nuovo cliente."""
    scaled_file = context.run_dir / "features_scalate.csv"
    stored = pd.read_csv(scaled_file, dtype={"id_cliente": "string"})
    feature_names = list(context.report["features"])
    if not set(feature_names).issubset(stored.columns):
        raise ValueError("features_scalate.csv non include tutte le feature del modello")
    if stored["id_cliente"].duplicated().any() or set(stored["id_cliente"]) != set(historical["id_cliente"]):
        raise ValueError("ID incompatibili tra training e matrice di preprocessing salvata")
    original_scaled = stored.set_index("id_cliente").loc[historical["id_cliente"], feature_names].to_numpy(dtype=float)
    if original_scaled.shape[1] < 2:
        raise ValueError("La PCA 2D richiede almeno 2 feature")
    pca = PCA(n_components=2)
    history_2d = pca.fit_transform(original_scaled)

    raw_features, _ = build_feature_frame(
        new, context.reference,
        tuple(feature_names), bool(context.report["log1p_spesa"]),
    )
    new_scaled = context.preprocessor.transform(raw_features)
    new_2d = pca.transform(new_scaled)
    centroids_2d = pca.transform(context.model.cluster_centers_)
    explained = float(sum(pca.explained_variance_ratio_) * 100)
    return history_2d, new_2d, centroids_2d, explained


def plot_customer_in_clusters(
    historical: pd.DataFrame,
    predictions: pd.DataFrame,
    context: PredictionContext,
    chosen_id: str,
    projection: str = "Richieste e spesa",
    show_all: bool = False,
) -> go.Figure:
    """Punti storici invariati; predizioni temporanee con una stella in primo piano.

    Se show_all=True, aggiunge anche gli altri clienti simulati come stelle più
    piccole. La PCA è sempre stimata sui soli clienti di training.
    """
    selected = predictions.loc[predictions["id_cliente"] == chosen_id]
    if len(selected) != 1:
        raise ValueError("Selezionare un solo NEW_CUSTOMER per il grafico")
    shown = predictions.reset_index(drop=True) if show_all else selected.reset_index(drop=True)
    chosen_pos = shown.index[shown["id_cliente"] == chosen_id][0]
    if projection == "PCA 2D":
        history_xy, new_xy, centers_xy, explained = _pca_coordinates(historical, shown, context)
        x_label = "Componente principale 1"
        y_label = "Componente principale 2"
        note = f"PCA sui soli clienti storici · varianza spiegata PC1+PC2: {explained:.1f}%"
    elif projection == "Richieste e spesa":
        history_xy = historical[["numero_richieste", "importo_totale_speso"]].to_numpy(dtype=float)
        new_xy = shown[["numero_richieste", "importo_totale_speso"]].to_numpy(dtype=float)
        centers_xy = None
        x_label = "Richieste negli ultimi 90 giorni"
        y_label = "Spesa totale nel periodo (€)"
        note = "Valori in unità originali: lo scatter non rappresenta le distanze del modello"
    else:
        raise ValueError(f"Proiezione non riconosciuta: {projection}")

    fig = go.Figure()
    labels = historical["cluster"].to_numpy(dtype=int)
    for cluster_id in sorted(np.unique(labels)):
        subset = historical.loc[labels == cluster_id]
        xy = history_xy[labels == cluster_id]
        fig.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1], mode="markers",
            name=f"Cluster {cluster_id} · {len(subset)} clienti",
            marker=dict(size=7, color=COLORS[int(cluster_id) % len(COLORS)], opacity=0.52),
            customdata=np.array(subset["id_cliente"].astype(str)).reshape(-1, 1),
            hovertemplate="Cliente %{customdata[0]}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        ))
    if centers_xy is not None:
        fig.add_trace(go.Scatter(
            x=centers_xy[:, 0], y=centers_xy[:, 1], mode="markers+text",
            text=[f"C{i}" for i in range(len(centers_xy))], textposition="top center",
            name="Centroidi K-Means", marker=dict(symbol="x", color="#222222", size=14, line=dict(width=2)),
            hovertemplate="Centroide K-Means<br>PC1 %{x:.2f}<br>PC2 %{y:.2f}<extra></extra>",
        ))
    if show_all and len(shown) > 1:
        others = shown.drop(index=chosen_pos)
        others_xy = np.delete(new_xy, chosen_pos, axis=0)
        meta = np.column_stack((others["id_cliente"].astype(str), others["cluster"].astype(int)))
        fig.add_trace(go.Scatter(
            x=others_xy[:, 0], y=others_xy[:, 1], mode="markers",
            name=f"Altri simulati ({len(others)})",
            marker=dict(symbol="star", size=13, color="#F6AD36", opacity=0.75,
                        line=dict(color="#855A15", width=1)),
            customdata=meta,
            hovertemplate="<b>%{customdata[0]}</b><br>Cluster %{customdata[1]}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        ))
    prediction = shown.iloc[chosen_pos]
    safe_id = html.escape(chosen_id)
    fig.add_trace(go.Scatter(
        x=[float(new_xy[chosen_pos, 0])], y=[float(new_xy[chosen_pos, 1])],
        mode="markers+text", name=f"{chosen_id} · Cluster {int(prediction['cluster'])}",
        text=[chosen_id], textposition="top center", textfont=dict(size=12, color="#9D5E00"),
        marker=dict(symbol="star", size=24, color="#FFC107", line=dict(color="#1E293B", width=2.5)),
        hovertemplate=("<b>" + safe_id + "</b><br>Cluster " + str(int(prediction["cluster"]))
                       + "<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"),
    ))
    fig.update_layout(
        title=dict(text="Posizione dei clienti simulati nei cluster", font=dict(size=20)),
        xaxis_title=x_label, yaxis_title=y_label,
        height=550, legend=dict(orientation="h", y=-0.23),
        margin=dict(l=30, r=20, t=65, b=100), hovermode="closest", template="plotly_white",
        annotations=[dict(text=note, xref="paper", yref="paper", x=0, y=1.11,
                          xanchor="left", showarrow=False, font=dict(size=11, color="#64748B"))],
    )
    return fig

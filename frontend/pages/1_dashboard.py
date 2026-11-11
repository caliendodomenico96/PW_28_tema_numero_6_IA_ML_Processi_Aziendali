# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Prima pagina della dashboard: risultati, filtri e scatter plot."""


from pathlib import Path

import streamlit as st
import pandas as pd

from libs.dashboard.validation_reports import latest_validation_for_k, validation_metrics_rows

from libs.dashboard.data import (
    read_training_report, load_cluster_results,
    filter_clients, cluster_summary,
)

ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def load_cached(run_directory: str, k: int, csv_mtime: float):
    # mtime è parte della cache key: un file rigenerato invalida la cache.
    return load_cluster_results(Path(run_directory), k)


st.title("Segmentazione della clientela")
st.caption("Project Work L-31 · analisi descrittiva dei segmenti K-Means")

selected_run = st.session_state.get("training_run_select")
selected_k = st.session_state.get("k_select")
if not selected_run or selected_k is None:
    st.info("Nessun training completo. Esegui src/2_kmeans_training.py: la dashboard lo rileverà automaticamente.")
    st.stop()

run_dir = ROOT / "results" / "kmeans" / selected_run
report = read_training_report(run_dir)

try:
    dataset, profiles = load_cached(str(run_dir), selected_k,
                                    (run_dir / f"cluster_k{selected_k}.csv").stat().st_mtime)
except (ValueError, FileNotFoundError, KeyError) as exc:
    st.error(f"Impossibile leggere l'esperimento: {exc}")
    st.stop()

with st.sidebar:
    st.divider()
    st.header("Filtri")
    zones = sorted(dataset["zona"].dropna().astype(str).unique().tolist())
    selected_zones = st.multiselect("Zona", zones, default=zones)
    categories = sorted(dataset["categoria_preferita"].dropna().astype(str).unique().tolist())
    selected_categories = st.multiselect("Categoria preferita", categories, default=categories)
    date_choice = st.selectbox("Periodo riferito a", ["data_iscrizione", "data_ultima_richiesta"],
                               format_func=lambda v: "Data iscrizione" if v == "data_iscrizione" else "Ultima richiesta")
    dates = dataset[date_choice].dropna()
    if not dates.empty:
        start_limit, end_limit = dates.min().date(), dates.max().date()
        selected_range = st.date_input("Intervallo date", value=(start_limit, end_limit),
                                       min_value=start_limit, max_value=end_limit)
    else:
        selected_range = ()

start_date, end_date = (selected_range[0], selected_range[1]) if len(selected_range) == 2 else (None, None)
filtered = filter_clients(dataset, selected_zones, selected_categories, date_choice, start_date, end_date)
if len(selected_range) == 1:
    st.info("Completa la selezione dell'intervallo temporale per applicare il filtro date.")

metrics = next((m for m in report.get("metriche", []) if int(m["k"]) == selected_k), {})
st.caption(f"Training {selected_run} · K={selected_k} · riferimento {report['data_riferimento']} · "
           "i filtri non modificano le assegnazioni K-Means")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clienti visualizzati", f"{len(filtered):,}".replace(",", "."))
c2.metric("Cluster nella selezione", filtered["cluster"].nunique())
c3.metric("Spesa nella selezione", f"€ {filtered['importo_totale_speso'].sum():,.2f}".replace(",", "_").replace(".", ",").replace("_", "."))
c4.metric("Silhouette del training", f"{float(metrics.get('silhouette', float('nan'))):.3f}")

st.subheader("Distribuzione dei clienti per cluster")
if filtered.empty:
    st.warning("Nessun cliente corrisponde ai filtri selezionati. "
               "I risultati globali di training e validation restano disponibili qui sotto.")
else:
    axis_names = [c for c in ("numero_richieste", "importo_totale_speso", "giorni_ultima_richiesta",
                              "giorni_da_iscrizione", "importo_medio_speso") if c in filtered]
    left, right = st.columns(2)
    with left:
        x_name = st.selectbox("Asse X", axis_names, index=0)
    with right:
        y_name = st.selectbox("Asse Y", axis_names, index=1)

    st.scatter_chart(filtered, x=x_name, y=y_name, color="segmento", size=40, height=430)
    st.caption("Ogni punto rappresenta un cliente. Colore = cluster calcolato sul dataset completo. "
               "Sono possibili sovrapposizioni tra clienti con valori uguali.")

    st.subheader("Profilo dei segmenti")
    summary = cluster_summary(filtered)
    st.dataframe(
        summary.style.format({
            "richieste_medie": "{:.2f}",
            "spesa_media_euro": "{:.2f}",
            "spesa_totale_euro": "{:.2f}",
            "recency_media_giorni": "{:.1f}",
            "anzianita_media_giorni": "{:.1f}",
            "percentuale_clienti": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Le medie nella tabella sono calcolate sulla selezione filtrata; "
               "per i profili completi usa i JSON originali del training.")

with st.expander("Visualizza grafici del training originale"):
    for fname, label in (
        (f"heatmap_centroidi_k{selected_k}.png", "Heatmap centroidi"),
        (f"scatter_pca_k{selected_k}.png", "PCA 2D e centroidi"),
        (f"boxplot_feature_k{selected_k}.png", "Boxplot delle feature"),
    ):
        image = run_dir / fname
        if image.is_file():
            st.image(str(image), caption=f"{label} · training completo · K={selected_k}")

st.subheader("Validazione dei cluster")
validation_info = latest_validation_for_k(run_dir, int(selected_k))
if validation_info is None:
    st.info(
        f"Nessun grafico di silhouette dettagliata disponibile per K={selected_k} "
        "nel training selezionato. Esegui `python src/3_kmeans_validation.py` "
        "impostando `[VALIDATION] training_run` sul training corrente "
        "e includendo K in `k_confronto` durante il training."
    )
else:
    validation_dir, validation_report = validation_info
    st.caption(f"Validazione {validation_dir.name} · training {selected_run} · "
               "misure riferite al dataset storico completo, indipendenti dai filtri della dashboard")
    plot_path = validation_dir / f"silhouette_k{selected_k}.png"
    st.image(str(plot_path), caption=f"Silhouette dettagliata per K={selected_k}", use_container_width=True)
    st.caption("Ogni banda rappresenta un cluster; la larghezza indica il coefficiente di silhouette "
               "dei singoli clienti. La linea tratteggiata rappresenta la media complessiva. "
               "Valori negativi indicano possibili assegnazioni ambigue.")
    with st.expander("Confronto metriche della validation"):
        metric_rows = validation_metrics_rows(validation_report)
        if metric_rows:
            st.dataframe(
                pd.DataFrame(metric_rows).style.format({
                    "Silhouette": "{:.4f}",
                    "Davies-Bouldin": "{:.4f}",
                    "Calinski-Harabasz": "{:.2f}",
                }),
                hide_index=True,
                use_container_width=True,
            )
        agreement = validation_report.get("accordo_fra_partizioni")
        if isinstance(agreement, dict) and agreement.get("adjusted_rand_index") is not None:
            st.caption("ARI tra K={k_a} e K={k_b}: {ari:.3f} (accordo tra partizioni, non accuratezza).".format(
                k_a=agreement.get("k_a"), k_b=agreement.get("k_b"),
                ari=float(agreement["adjusted_rand_index"])))
        st.caption("Validazione interna: le metriche sono calcolate sui dati storici di training; "
                   "non costituiscono una valutazione su dati indipendenti.")

with st.expander("Clienti della selezione"):
    visible = [c for c in ("id_cliente", "zona", "data_iscrizione", "categoria_preferita",
                           "numero_richieste", "importo_totale_speso", "giorni_ultima_richiesta", "cluster")
               if c in filtered]
    st.dataframe(filtered[visible], hide_index=True, use_container_width=True)

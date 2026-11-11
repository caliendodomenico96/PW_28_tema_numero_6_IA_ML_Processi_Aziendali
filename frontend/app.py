# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Unico entrypoint Streamlit, catalogo auto-aggiornato e selezione condivisa."""


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from libs.dashboard.data import (
    available_dataset_folders,
    completed_training_runs,
    dashboard_catalog_snapshot,
    read_training_report,
    training_run_status,
    training_source_path,
    usable_training_runs,
)

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")


@st.fragment(run_every="10s")
def watch_catalog():
    """Controlla la disponibilità senza ricaricare la dashboard a intervalli fissi.

    Un full rerun avviene solo se cambia il catalogo: anche cancellando cartelle.
    Il monitor resta nell'entrypoint, ed è attivo in entrambe le pagine.
    """
    current = dashboard_catalog_snapshot(PROJECT_ROOT)
    previous = st.session_state.get("_catalog_signature")
    if previous is None:
        st.session_state["_catalog_signature"] = current
    elif current != previous:
        old_runs = {name for name, usable, _ in previous[0] if usable}
        new_runs = [name for name, usable, _ in current[0] if usable and name not in old_runs]
        st.session_state["_catalog_signature"] = current
        if new_runs:
            st.session_state["_pending_training_run"] = new_runs[0]
        # Trigger di un rerun completo se vengono aggiunte o eliminate cartelle.
        st.rerun()


watch_catalog()

runs = usable_training_runs(PROJECT_ROOT)
run_by_name = {path.name: path for path in runs}
all_runs = completed_training_runs(PROJECT_ROOT)
datasets = available_dataset_folders(PROJECT_ROOT)

with st.sidebar:
    st.header("Esperimento K-Means")
    if runs:
        pending = st.session_state.pop("_pending_training_run", None)
        if pending in run_by_name:
            st.session_state["training_run_select"] = pending
        elif st.session_state.get("training_run_select") not in run_by_name:
            st.session_state["training_run_select"] = runs[0].name

        selected_run = st.selectbox(
            "Training", options=list(run_by_name), key="training_run_select",
            help="I training nuovi compaiono automaticamente, quelli rimossi non sono più selezionabili.",
        )
        report = read_training_report(run_by_name[selected_run])
        choices = sorted(int(k) for k in report["k_modelli_salvati"])
        suggested = int(report.get("scelta_k", {}).get("k_raccomandato", choices[-1]))
        if st.session_state.get("_last_selected_run") != selected_run:
            st.session_state["_last_selected_run"] = selected_run
            st.session_state["k_select"] = suggested if suggested in choices else choices[0]
        elif st.session_state.get("k_select") not in choices:
            st.session_state["k_select"] = suggested if suggested in choices else choices[0]
        st.selectbox("Numero di cluster (K)", options=choices, key="k_select")

        latest_source = training_source_path(PROJECT_ROOT, report)
        newest_dataset = datasets[0] if datasets else None
        if newest_dataset is not None and newest_dataset != latest_source.parent:
            st.info("È disponibile un dataset diverso da quello del training selezionato. "
                    "Per usarlo, esegui `python src/2_kmeans_training.py`.")
    else:
        st.session_state.pop("training_run_select", None)
        st.session_state.pop("k_select", None)
        st.info("Nessun training utilizzabile. Genera un dataset e avvia `src/2_kmeans_training.py`.")
    invalid = [run for run in all_runs if not training_run_status(PROJECT_ROOT, run)[0]]
    if invalid:
        st.warning(f"{len(invalid)} training nascosti perché il dataset originale o altri file "
                   "necessari non sono più disponibili.")
    st.caption(f"Dataset disponibili: {len(datasets)} · Training utilizzabili: {len(runs)}")
    st.caption("Aggiornamento automatico ogni 10 secondi, anche dopo cancellazioni")

pages = [
    st.Page("pages/1_dashboard.py", title="Analisi dei cluster", icon="📊"),
    st.Page("pages/2_nuovo_cliente.py", title="Nuovi clienti", icon="👤"),
]
page = st.navigation(pages)
page.run()

# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Simulazione multi-cliente con tabella selezionabile e visualizzazione dinamica.

I nuovi clienti sono conservati SOLO in st.session_state, separati per training.
La selezione di K cambia le predizioni ma non riaddestra né modifica i dati storici.
"""


from pathlib import Path

import pandas as pd
import sklearn
import streamlit as st

from frontend.components.customer_form import render_customer_form
from libs.dashboard.data import load_cluster_results, read_training_report
from libs.dashboard.inference import MAX_BATCH_ROWS, load_prediction_context, read_uploaded_csv
from libs.dashboard.simulation import (
    add_customers, cluster_scenario_summary, delete_customer, edit_customer,
    empty_simulation, predict_simulation,
)
from libs.dashboard.visualization import plot_customer_in_clusters

ROOT = Path(__file__).resolve().parents[2]


@st.cache_resource(show_spinner="Caricamento modello K-Means...")
def load_model_cached(run_dir: str, k: int, model_mtime_ns: int, scaler_mtime_ns: int):
    return load_prediction_context(Path(run_dir), k)


@st.cache_data(show_spinner=False)
def load_historical_cached(run_dir: str, k: int, assignments_mtime_ns: int):
    return load_cluster_results(Path(run_dir), k)[0]


st.title("Nuovi clienti · Simulazione multi-cliente")
st.caption("Aggiungi, modifica, seleziona e confronta clienti senza cambiare il training originale")
run_name = st.session_state.get("training_run_select")
k = st.session_state.get("k_select")
if not run_name or k is None:
    st.info("Nessun training utilizzabile. Genera il dataset ed esegui il training K-Means.")
    st.stop()

run_dir = ROOT / "results" / "kmeans" / run_name
try:
    training_report = read_training_report(run_dir)
    model_path = run_dir / "models" / f"kmeans_k{k}.joblib"
    preprocessing_path = run_dir / "models" / "preprocessor.joblib"
    context = load_model_cached(str(run_dir), int(k), model_path.stat().st_mtime_ns,
                                preprocessing_path.stat().st_mtime_ns)
    cluster_path = run_dir / f"cluster_k{k}.csv"
    historical = load_historical_cached(str(run_dir), int(k), cluster_path.stat().st_mtime_ns)
except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
    st.error(f"Non riesco a caricare il training: {exc}")
    st.stop()

trained_version = training_report.get("versioni", {}).get("sklearn")
if trained_version and trained_version != sklearn.__version__:
    st.warning(f"Modello salvato con scikit-learn {trained_version}; ambiente attuale {sklearn.__version__}. "
               "Per compatibilità usa la stessa versione del training.")

states = st.session_state.setdefault("simulations_by_training", {})
if run_name not in states:
    states[run_name] = empty_simulation()
state = states[run_name]
selected_key = f"simulation_selected_{run_name}"
revision_key = f"simulation_revision_{run_name}"


def save_state(new_state: dict, selected: str | None, message: str) -> None:
    states[run_name] = new_state
    st.session_state[selected_key] = selected
    st.session_state[revision_key] = st.session_state.get(revision_key, 0) + 1
    st.session_state["simulation_flash"] = message
    st.rerun()


flash = st.session_state.pop("simulation_flash", None)
if flash:
    st.success(flash)

try:
    predicted = predict_simulation(state, context)
except (ValueError, TypeError, KeyError) as exc:
    st.error(f"I clienti salvati nella simulazione non sono compatibili con questo training: {exc}")
    st.caption("Seleziona il training precedente oppure svuota la simulazione associata a questo training.")
    if st.button("Svuota la simulazione per questo training"):
        save_state(empty_simulation(), None, "Simulazione svuotata.")
    st.stop()

selected_id = st.session_state.get(selected_key)
ids = predicted["id_cliente"].tolist() if not predicted.empty else []
if selected_id not in ids:
    selected_id = ids[0] if ids else None
    st.session_state[selected_key] = selected_id

st.info(f"**Training:** {run_name} · **K={k}** · **Riferimento:** {context.reference:%d/%m/%Y} · "
        f"**Osservazione:** {context.observation_days} giorni")
st.caption("Gli inserimenti sono relativi alla data del training e rimangono nella sessione Streamlit. "
           "Cambiando K, le predizioni sono ricalcolate sul medesimo elenco di clienti simulati.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Clienti storici", len(historical))
m2.metric("Clienti simulati", len(ids))
m3.metric("Cluster disponibili", int(k))
m4.metric("Prossimo ID", f"NEW_CUSTOMER_{int(state['next_sequence']):03d}")

st.subheader("Clienti della simulazione")
if predicted.empty:
    st.info("Non hai ancora inserito clienti: usa il modulo in basso per iniziare.")
else:
    table_fields = [
        "id_cliente", "nome", "cognome", "zona", "categoria_preferita",
        "data_iscrizione", "numero_richieste", "importo_totale_speso",
        "giorni_ultima_richiesta", "cluster", "distanza_centroide",
    ]
    displayed = predicted[[c for c in table_fields if c in predicted.columns]]
    selection = st.dataframe(
        displayed, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        key=f"simulation_table_{run_name}_{st.session_state.get(revision_key, 0)}",
    )
    selected_rows = selection.selection.rows
    if selected_rows:
        row_number = int(selected_rows[0])
        if 0 <= row_number < len(displayed):
            selected_id = str(displayed.iloc[row_number]["id_cliente"])
            st.session_state[selected_key] = selected_id
    st.caption("Clicca una riga per evidenziarla nei grafici e modificarla. Gli ID rimangono stabili "
               "dopo eliminazioni o modifiche; i numeri non vengono riutilizzati.")

    one, two = st.columns([1, 3])
    with one:
        if st.button("Elimina cliente selezionato", disabled=selected_id is None):
            try:
                updated = delete_customer(state, selected_id)
                next_selection = updated["records"][0]["id_cliente"] if updated["records"] else None
                save_state(updated, next_selection, f"{selected_id} eliminato dalla simulazione.")
            except ValueError as exc:
                st.error(str(exc))
    with two:
        confirm_clear = st.checkbox("Confermo la rimozione di tutti i clienti simulati")
        if st.button("Svuota simulazione", disabled=not confirm_clear):
            save_state(empty_simulation(), None, "Tutti i clienti simulati sono stati rimossi.")

    st.subheader("Distribuzione dei cluster · scenario temporaneo")
    counts = cluster_scenario_summary(historical, predicted, int(k))
    st.dataframe(counts, hide_index=True, use_container_width=True)
    st.caption("'Totale scenario' = clienti storici + simulati. È una visualizzazione ipotetica: "
               "i centroidi, le medie e la silhouette del training non vengono ricalcolati.")

    st.subheader("Posizione dei nuovi clienti rispetto ai cluster")
    graph_col, view_col = st.columns(2)
    with graph_col:
        projection = st.radio("Proiezione", ["Richieste e spesa", "PCA 2D"], horizontal=True)
    with view_col:
        plot_mode = st.radio("Mostra", ["Cliente selezionato", "Tutti i simulati"], horizontal=True)
    try:
        fig = plot_customer_in_clusters(historical, predicted, context, selected_id, projection,
                                        show_all=(plot_mode == "Tutti i simulati"))
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    except (ValueError, KeyError, OSError) as exc:
        st.warning(f"Non riesco a rappresentare lo scatter: {exc}")
    st.caption("⭐ Stella grande: cliente selezionato. Stelle piccole: altri clienti simulati. "
               "Cerchi: clienti storici. Nella PCA, le X nere rappresentano i centroidi del training.")

    selected_customer = predicted.loc[predicted["id_cliente"] == selected_id].iloc[0]
    profile = context.profiles.get(str(int(selected_customer["cluster"])), {})
    st.markdown(f"**{selected_id} → Cluster {int(selected_customer['cluster'])}** · "
                f"{profile.get('clienti', '—')} clienti storici nel gruppo")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Richieste medie storiche", str(profile.get("media_richieste", "—")))
    p2.metric("Spesa media storica", f"€ {profile.get('media_spesa_euro', '—')}")
    p3.metric("Recency media", f"{profile.get('media_recency_giorni', '—')} giorni")
    p4.metric("Anzianità media", f"{profile.get('media_anzianita_giorni', '—')} giorni")
    st.download_button("Esporta tutti i clienti simulati (.csv)",
                       data=predicted.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name=f"clienti_simulati_k{k}.csv", mime="text/csv")

st.divider()
st.subheader("Gestione clienti simulati")
add_tab, edit_tab, csv_tab = st.tabs(["Aggiungi cliente", "Modifica selezionato", "Importa più clienti da CSV"])
zone_options = sorted(historical["zona"].dropna().astype(str).unique().tolist())
category_options = sorted(historical["categoria_preferita"].dropna().astype(str).unique().tolist())

with add_tab:
    raw_customer = render_customer_form(
        key=f"simulation_add_{run_name}_{st.session_state.get(revision_key, 0)}",
        reference=context.reference, observation_days=context.observation_days,
        zone_options=zone_options, category_options=category_options,
        submit_label="Aggiungi e classifica",
    )
    if raw_customer is not None:
        try:
            updated, new_ids = add_customers(state, pd.DataFrame([raw_customer]), context)
            save_state(updated, new_ids[-1], f"{new_ids[-1]} aggiunto e classificato.")
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Cliente non valido: {exc}")

with edit_tab:
    if selected_id is None:
        st.info("Aggiungi un cliente e selezionalo dalla tabella per modificarlo.")
    else:
        item = next(record for record in state["records"] if record["id_cliente"] == selected_id)
        st.markdown(f"**Modifica di {selected_id}**. Verrà ricalcolato il cluster, ma l'ID resterà invariato.")
        changed_customer = render_customer_form(
            key=f"simulation_edit_{run_name}_{selected_id}_{st.session_state.get(revision_key, 0)}",
            reference=context.reference, observation_days=context.observation_days,
            zone_options=zone_options, category_options=category_options,
            existing=item["payload"], submit_label="Salva modifica e ricalcola",
        )
        if changed_customer is not None:
            try:
                updated = edit_customer(state, selected_id, pd.DataFrame([changed_customer]), context)
                save_state(updated, selected_id, f"{selected_id} aggiornato e riclassificato.")
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"Modifica non valida: {exc}")

with csv_tab:
    example_date = (context.reference - pd.Timedelta(days=180)).strftime("%d/%m/%Y")
    example_csv = (
        "id_cliente;nome;cognome;email;zona;categoria_preferita;data_iscrizione;"
        "numero_richieste;importo_totale_speso;giorni_ultima_richiesta\n"
        f"E01;Mario;Rossi;new.customer.001@example.com;Roma;Configurazione backup;{example_date};2;100,00;14\n"
    )
    st.download_button("Scarica template CSV", example_csv.encode("utf-8-sig"),
                       file_name="template_clienti_simulati.csv", mime="text/csv")
    csv_file = st.file_uploader("Carica uno o più clienti", type="csv", key=f"simulation_upload_{run_name}")
    imported = None
    if csv_file is not None:
        try:
            imported = read_uploaded_csv(csv_file.getvalue())
            st.caption(f"File con {len(imported)} clienti. I vecchi ID, se presenti, saranno tracciati come origine.")
            st.dataframe(imported.head(10), use_container_width=True, hide_index=True)
        except ValueError as exc:
            st.error(str(exc))
    st.caption(f"Limite massimo di {MAX_BATCH_ROWS} clienti per simulazione. "
               "L'importazione si aggiunge ai record già presenti, non li sostituisce.")
    if st.button("Aggiungi tutti e classifica", disabled=imported is None, type="primary"):
        try:
            updated, new_ids = add_customers(state, imported, context)
            save_state(updated, new_ids[-1], f"Importati e classificati {len(new_ids)} clienti.")
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Caricamento rifiutato: {exc}")

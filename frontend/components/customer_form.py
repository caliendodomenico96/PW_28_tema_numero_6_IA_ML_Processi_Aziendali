# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================
"""Form Streamlit condiviso per aggiunta e modifica di un cliente simulato."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st


def render_customer_form(
    *, key: str, reference: pd.Timestamp, observation_days: int,
    zone_options: list[str], category_options: list[str],
    existing: dict | None = None, submit_label: str = "Aggiungi cliente",
) -> dict | None:
    """Ritorna un record solo al submit, senza salvare dati sul disco."""
    initial = existing or {}
    ref_date = reference.date()
    signup_default = pd.to_datetime(initial.get("data_iscrizione", ""), format="%d/%m/%Y", errors="coerce")
    signup_default = signup_default.date() if pd.notna(signup_default) else ref_date - timedelta(days=180)
    signup_default = min(signup_default, ref_date)
    zones = sorted(set(zone_options) | ({str(initial['zona'])} if initial.get('zona') else set()))
    categories = sorted(set(category_options) | ({str(initial['categoria_preferita'])} if initial.get('categoria_preferita') else set()))

    with st.form(key):
        st.caption("Anagrafica, zona e categoria sono descrittive: non entrano nelle feature K-Means.")
        c1, c2, c3 = st.columns(3)
        with c1:
            first = st.text_input("Nome", value=str(initial.get("nome", "")))
            last = st.text_input("Cognome", value=str(initial.get("cognome", "")))
        with c2:
            email = st.text_input("Email sintetica", value=str(initial.get("email", "")),
                                  help="Preferisci indirizzi example.com")
            zone = st.selectbox("Zona", zones, index=zones.index(initial.get("zona")) if initial.get("zona") in zones else 0)
        with c3:
            category = st.selectbox("Categoria preferita", categories,
                                    index=categories.index(initial.get("categoria_preferita"))
                                    if initial.get("categoria_preferita") in categories else 0)
            signup = st.date_input("Data iscrizione", value=signup_default, max_value=ref_date,
                                   format="DD/MM/YYYY")
        c4, c5, c6 = st.columns(3)
        with c4:
            requests = st.number_input(f"Richieste negli ultimi {observation_days} giorni", min_value=0,
                                       step=1, value=int(initial.get("numero_richieste", 2)))
        with c5:
            spending = st.number_input("Spesa nel periodo (€)", min_value=0.0,
                                       step=5.0, value=float(initial.get("importo_totale_speso", 100.0)),
                                       format="%.2f")
        with c6:
            recency = st.number_input("Giorni dall'ultima richiesta", min_value=0, max_value=10000,
                                      step=1, value=int(initial.get("giorni_ultima_richiesta", 14)))
        st.caption("Zero richieste richiede zero spesa e ultima richiesta fuori dalla finestra. "
                   "L'importo medio e la data dell'ultima richiesta vengono calcolati automaticamente.")
        submitted = st.form_submit_button(submit_label, type="primary")

    if not submitted:
        return None
    return {
        "nome": first.strip(), "cognome": last.strip(), "email": email.strip(),
        "zona": zone, "categoria_preferita": category,
        "data_iscrizione": signup.strftime("%d/%m/%Y"),
        "numero_richieste": int(requests),
        "importo_totale_speso": float(spending),
        "giorni_ultima_richiesta": int(recency),
    }

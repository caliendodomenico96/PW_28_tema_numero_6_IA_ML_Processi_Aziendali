# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


""" Sessione di simulazione multi-cliente, senza scritture sul dataset storico. """


from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from .inference import MAX_BATCH_ROWS, PredictionContext, predict_new_customers

# Campi conservati in sessione: niente centroidi, etichette o informazioni di training.
INPUT_FIELDS = (
    "nome", "cognome", "email", "zona", "categoria_preferita",
    "data_iscrizione", "numero_richieste", "importo_totale_speso",
    "giorni_ultima_richiesta",
)


def empty_simulation() -> dict[str, Any]:
    return {"records": [], "next_sequence": 1}


def _validated_payloads(raw: pd.DataFrame, context: PredictionContext) -> tuple[pd.DataFrame, list[dict]]:
    """Validazione atomica; prima si verifica tutto il batch, poi si aggiorna lo stato."""
    predictions = predict_new_customers(raw, context)
    payloads = []
    for _, row in predictions.iterrows():
        payload = {key: row[key] for key in INPUT_FIELDS}
        payload["numero_richieste"] = int(payload["numero_richieste"])
        payload["giorni_ultima_richiesta"] = int(payload["giorni_ultima_richiesta"])
        payload["importo_totale_speso"] = float(payload["importo_totale_speso"])
        payloads.append(payload)
    return predictions, payloads


def add_customers(
    state: dict[str, Any], raw: pd.DataFrame, context: PredictionContext
) -> tuple[dict[str, Any], list[str]]:
    """Aggiunge uno o più clienti assegnando ID incrementali non riciclati."""
    if len(state["records"]) + len(raw) > MAX_BATCH_ROWS:
        raise ValueError(f"Massimo {MAX_BATCH_ROWS} clienti contemporaneamente nella simulazione")
    predictions, payloads = _validated_payloads(raw, context)
    updated = deepcopy(state)
    assigned_ids: list[str] = []
    for (_, validated), payload in zip(predictions.iterrows(), payloads):
        seq = int(updated["next_sequence"])
        customer_id = f"NEW_CUSTOMER_{seq:03d}"
        origin = validated.get("id_cliente_origine")
        updated["records"].append({
            "id_cliente": customer_id,
            "payload": payload,
            "id_cliente_origine": "" if pd.isna(origin) else str(origin),
        })
        assigned_ids.append(customer_id)
        updated["next_sequence"] = seq + 1
    return updated, assigned_ids


def edit_customer(
    state: dict[str, Any], customer_id: str, raw: pd.DataFrame, context: PredictionContext
) -> dict[str, Any]:
    """Rivalida tutte le feature del cliente modificato, senza cambiarne l'ID."""
    if len(raw) != 1:
        raise ValueError("La modifica deve contenere esattamente un cliente")
    _, payloads = _validated_payloads(raw, context)
    updated = deepcopy(state)
    for record in updated["records"]:
        if record["id_cliente"] == customer_id:
            record["payload"] = payloads[0]
            return updated
    raise ValueError(f"Cliente inesistente nella simulazione: {customer_id}")


def delete_customer(state: dict[str, Any], customer_id: str) -> dict[str, Any]:
    updated = deepcopy(state)
    updated["records"] = [record for record in updated["records"] if record["id_cliente"] != customer_id]
    if len(updated["records"]) == len(state["records"]):
        raise ValueError(f"Cliente inesistente nella simulazione: {customer_id}")
    return updated


def predict_simulation(state: dict[str, Any], context: PredictionContext) -> pd.DataFrame:
    """Ricalcola i cluster con il K corrente, preservando gli ID della sessione."""
    records = state["records"]
    if not records:
        return pd.DataFrame()
    raw = pd.DataFrame([record["payload"] for record in records])
    predicted = predict_new_customers(raw, context)
    predicted["id_cliente"] = [record["id_cliente"] for record in records]
    origins = [record.get("id_cliente_origine", "") for record in records]
    if any(origins):
        predicted["id_cliente_origine"] = origins
    return predicted


def cluster_scenario_summary(historical: pd.DataFrame, predicted: pd.DataFrame, k: int) -> pd.DataFrame:
    """Conteggi originali, aggiunte temporanee e totali ipotetici; nessun refit."""
    before = historical["cluster"].value_counts()
    after = predicted["cluster"].value_counts() if not predicted.empty else pd.Series(dtype=int)
    return pd.DataFrame([
        {
            "cluster": cluster,
            "clienti_storici": int(before.get(cluster, 0)),
            "clienti_simulati": int(after.get(cluster, 0)),
            "totale_scenario": int(before.get(cluster, 0) + after.get(cluster, 0)),
        }
        for cluster in range(k)
    ])

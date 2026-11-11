# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Pulizia dei dati e feature engineering, con trasformazioni riproducibili."""


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .storage import json_read


def load_customer_data(csv_path: Path) -> tuple[pd.DataFrame, dict, pd.Timestamp]:
    """Legge il CSV e recupera la data di riferimento dai metadati della generazione."""
    csv_path = Path(csv_path)
    metadata_path = csv_path.with_suffix(".metadati.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadati necessari per data di riferimento: {metadata_path}")
    metadata = json_read(metadata_path)
    reference = pd.to_datetime(metadata["fine_osservazione_inclusa"], format="%d/%m/%Y", errors="raise")
    sep = metadata.get("separatore_csv", ",")
    if sep not in {",", ";"}:
        raise ValueError("Separatore CSV non valido nei metadati")
    df = pd.read_csv(csv_path, sep=sep, dtype={"id_cliente": "string"}, encoding="utf-8")
    if "id_cliente" not in df.columns:
        raise ValueError("Nel CSV manca id_cliente")
    if df["id_cliente"].isna().any() or df["id_cliente"].duplicated().any():
        raise ValueError("id_cliente mancante o duplicato: impossibile associare un cluster a ogni cliente")
    if len(df) < 5:
        raise ValueError("Il dataset contiene meno di 5 clienti")
    return df, metadata, reference


def build_feature_frame(
    dataframe: pd.DataFrame,
    reference: pd.Timestamp,
    features: tuple[str, ...],
    log1p_spesa: bool,
) -> tuple[pd.DataFrame, dict]:
    """Ricostruisce sempre le stesse feature; NON legge alcuna etichetta latente."""
    df = dataframe
    extracted = pd.DataFrame(index=df.index)
    details = {"righe": len(df), "imputazione": "mediana sul dataset di training", "features": list(features),
               "trasformazione_spesa_log1p": log1p_spesa, "valori_mancanti_per_feature": {},
               "valori_invalidi_o_negativi_per_feature": {}}
    for feature in features:
        if feature == "giorni_da_iscrizione":
            if "data_iscrizione" not in df:
                raise ValueError("Manca data_iscrizione, necessaria per giorni_da_iscrizione")
            dates = pd.to_datetime(df["data_iscrizione"], format="%d/%m/%Y", errors="coerce")
            values = (reference - dates).dt.days.astype("float64")
        else:
            if feature not in df:
                raise ValueError(f"Feature obbligatoria assente: {feature}")
            values = pd.to_numeric(df[feature].astype("string").str.replace(",", ".", regex=False), errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).astype("float64")
        invalid = values.lt(0)
        if feature == "numero_richieste":
            invalid = invalid | (values.notna() & (values % 1 != 0))
        details["valori_invalidi_o_negativi_per_feature"][feature] = int(invalid.sum())
        values = values.mask(invalid)
        details["valori_mancanti_per_feature"][feature] = int(values.isna().sum())
        if values.isna().all():
            raise ValueError(f"La feature {feature} ha tutti i valori mancanti")
        if values.isna().mean() > 0.25:
            raise ValueError(f"La feature {feature} supera il 25% di valori mancanti")
        if feature == "importo_totale_speso" and log1p_spesa:
            values = np.log1p(values)
        extracted[feature] = values
    details["percentuali_mancanti_per_feature"] = {
        k: round(v / len(df) * 100, 2) for k, v in details["valori_mancanti_per_feature"].items()
    }
    return extracted, details


def create_preprocessor() -> Pipeline:
    # Fit di imputazione e standardizzazione SOLO durante il training.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

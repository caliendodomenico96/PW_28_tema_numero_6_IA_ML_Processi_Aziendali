# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


""" Predizione di nuovi clienti con il preprocessor e K-Means già addestrati """


from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from libs.clustering.preprocessing import build_feature_frame
from libs.dashboard.data import read_training_report

REQUIRED_FIELDS = (
    "data_iscrizione", "numero_richieste", "importo_totale_speso", "giorni_ultima_richiesta"
)
MAX_BATCH_ROWS = 1000
MAX_UPLOAD_BYTES = 2_000_000


@dataclass(frozen=True)
class PredictionContext:
    run_dir: Path
    k: int
    report: dict
    reference: pd.Timestamp
    observation_days: int
    profiles: dict
    preprocessor: object
    model: object


def load_prediction_context(run_dir: Path, k: int) -> PredictionContext:
    """Carica esclusivamente modelli locali creati dal proprio training."""
    run_dir = Path(run_dir).resolve()
    report = read_training_report(run_dir)
    if int(k) not in [int(x) for x in report["k_modelli_salvati"]]:
        raise ValueError(f"K={k} non è presente fra i modelli salvati")
    source = Path(report["dataset_csv"])
    if not source.is_absolute():
        # training_dir = PROJECT_ROOT/results/kmeans/<timestamp>
        source = run_dir.parents[2] / source
    metadata_path = source.with_suffix(".metadati.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadati di origine non trovati: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)
    observation_days = int(metadata.get("configurazione", {}).get("giorni_osservazione", 90))
    if observation_days <= 0:
        raise ValueError("Finestra di osservazione non valida")
    reference = pd.to_datetime(report["data_riferimento"], format="%d/%m/%Y", errors="raise")
    profiles_path = run_dir / f"profili_cluster_k{k}.json"
    with profiles_path.open(encoding="utf-8") as f:
        profiles = json.load(f)
    model_path = run_dir / "models" / f"kmeans_k{k}.joblib"
    preproc_path = run_dir / "models" / "preprocessor.joblib"
    if not model_path.is_file() or not preproc_path.is_file():
        raise FileNotFoundError("Modello K-Means o preprocessing non disponibili")
    # joblib deserializza oggetti Python: aprire esclusivamente file locali fidati.
    preprocessor = joblib.load(preproc_path)
    model = joblib.load(model_path)
    if int(model.n_clusters) != int(k):
        raise ValueError("Il numero di cluster del modello non corrisponde al training")
    n_features = len(report["features"])
    if preprocessor.n_features_in_ != n_features or model.n_features_in_ != n_features:
        raise ValueError("Numero di feature non coerente tra report, preprocessor e modello")
    return PredictionContext(run_dir, int(k), report, reference, observation_days,
                             profiles, preprocessor, model)


def read_uploaded_csv(contents: bytes) -> pd.DataFrame:
    """CSV italiano con separatore , oppure ; e intestazione obbligatoria."""
    if not contents:
        raise ValueError("Il file CSV è vuoto")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV troppo grande (massimo 2 MB)")
    try:
        df = pd.read_csv(BytesIO(contents), sep=None, engine="python", dtype="string", encoding="utf-8-sig")
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Impossibile leggere il CSV: {exc}") from exc
    if len(df) == 0:
        raise ValueError("Il CSV non contiene clienti")
    if len(df) > MAX_BATCH_ROWS:
        raise ValueError(f"Per una singola predizione sono ammessi al massimo {MAX_BATCH_ROWS} clienti")
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError("Il CSV contiene intestazioni duplicate")
    return df


def validate_new_customers(data: pd.DataFrame, reference: pd.Timestamp, observation_days: int) -> pd.DataFrame:
    """Validazione esplicita, prima della pipeline ML che gestisce i NaN.

    La data di riferimento è quella del training, non la data corrente del server.
    I clienti senza alcun acquisto storico non appartengono al dominio simulato.
    """
    if data.empty:
        raise ValueError("Non ci sono righe da classificare")
    missing = [c for c in REQUIRED_FIELDS if c not in data.columns]
    if missing:
        raise ValueError(f"Colonne obbligatorie mancanti: {', '.join(missing)}")
    if len(data) > MAX_BATCH_ROWS:
        raise ValueError(f"Limite di {MAX_BATCH_ROWS} clienti per caricamento")
    df = data.copy().reset_index(drop=True)
    dates = pd.to_datetime(df["data_iscrizione"], format="%d/%m/%Y", errors="coerce")
    numerical = {}
    for column in REQUIRED_FIELDS[1:]:
        values = df[column].astype("string").str.strip().str.replace(",", ".", regex=False)
        numerical[column] = pd.to_numeric(values, errors="coerce")
    invalid = []
    days_signup = (reference - dates).dt.days
    for idx in range(len(df)):
        row_num = idx + 2  # intestazione CSV = riga 1
        nr = numerical["numero_richieste"].iloc[idx]
        spending = numerical["importo_totale_speso"].iloc[idx]
        recency = numerical["giorni_ultima_richiesta"].iloc[idx]
        age = days_signup.iloc[idx]
        if pd.isna(dates.iloc[idx]) or pd.isna(age) or age < 0:
            invalid.append(f"riga {row_num}: data_iscrizione non valida o successiva al riferimento")
            continue
        if any(pd.isna(v) or not np.isfinite(float(v)) for v in (nr, spending, recency)):
            invalid.append(f"riga {row_num}: richieste/spesa/recency assenti o non numeriche")
            continue
        if nr < 0 or nr != int(nr) or spending < 0 or recency < 0 or recency != int(recency):
            invalid.append(f"riga {row_num}: frequenza e recency devono essere interi non negativi; spesa >= 0")
            continue
        if recency > age:
            invalid.append(f"riga {row_num}: ultima richiesta precedente all'iscrizione")
            continue
        if nr == 0 and spending != 0:
            invalid.append(f"riga {row_num}: senza richieste nel periodo la spesa deve essere zero")
            continue
        if nr > 0 and spending <= 0:
            invalid.append(f"riga {row_num}: richieste effettuate ma spesa nulla (interventi pagati)")
            continue
        if nr == 0 and recency < observation_days:
            invalid.append(f"riga {row_num}: 0 richieste nel periodo, ma l'ultima risulta entro {observation_days} giorni")
            continue
        if nr > 0 and recency >= observation_days:
            invalid.append(f"riga {row_num}: richieste nel periodo, ma ultima richiesta fuori dai {observation_days} giorni")
            continue
    if invalid:
        preview = "; ".join(invalid[:8])
        if len(invalid) > 8:
            preview += f"; ... ({len(invalid) - 8} ulteriori righe)"
        raise ValueError(preview)
    df["data_iscrizione"] = dates.dt.strftime("%d/%m/%Y")
    for column in numerical:
        df[column] = numerical[column].astype(int if column != "importo_totale_speso" else float)

    # Le date e le medie nel CSV devono coincidere con le feature derivate.
    # Il nuovo cliente non entra nel dataset storico: i campi sono solo descrittivi.
    calculated_last_date = (reference - pd.to_timedelta(df["giorni_ultima_richiesta"], unit="D"))
    if "data_ultima_richiesta" in df.columns:
        reported_last_date = pd.to_datetime(df["data_ultima_richiesta"], format="%d/%m/%Y", errors="coerce")
        nonempty = df["data_ultima_richiesta"].notna() & (df["data_ultima_richiesta"].astype("string").str.strip() != "")
        if (nonempty & (reported_last_date.isna() | (reported_last_date != calculated_last_date))).any():
            raise ValueError("data_ultima_richiesta non coerente con giorni_ultima_richiesta")
    df["data_ultima_richiesta"] = calculated_last_date.dt.strftime("%d/%m/%Y")
    avg = np.where(
        df["numero_richieste"] > 0,
        df["importo_totale_speso"] / df["numero_richieste"].replace(0, np.nan),
        0.0,
    )
    avg = np.round(avg.astype(float), 2)
    if "importo_medio_speso" in df.columns:
        given = pd.to_numeric(df["importo_medio_speso"].astype("string").str.replace(",", ".", regex=False), errors="coerce")
        nonempty = df["importo_medio_speso"].notna() & (df["importo_medio_speso"].astype("string").str.strip() != "")
        if (nonempty & (given.isna() | ((given - avg).abs() > 0.011))).any():
            raise ValueError("importo_medio_speso non coerente con importo_totale_speso / numero_richieste")
    df["importo_medio_speso"] = avg

    for column in ("nome", "cognome", "email", "zona", "categoria_preferita"):
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()

    # Nuovi ID *sempre* progressivi, anche se un CSV caricato possiede vecchi ID.
    if "id_cliente" in df.columns:
        supplied = df["id_cliente"].fillna("").astype(str).str.strip()
        if supplied.ne("").any():
            df["id_cliente_origine"] = supplied
    df["id_cliente"] = [f"NEW_CUSTOMER_{idx:03d}" for idx in range(1, len(df) + 1)]
    return df


def predict_new_customers(raw: pd.DataFrame, context: PredictionContext) -> pd.DataFrame:
    """Riusa preprocessing (incluso log1p) e KMeans.predict senza modificare i modelli."""
    df = validate_new_customers(raw, context.reference, context.observation_days)
    features, _ = build_feature_frame(df, context.reference,
                                      tuple(context.report["features"]),
                                      bool(context.report["log1p_spesa"]))
    scaled = context.preprocessor.transform(features)
    if not np.isfinite(scaled).all():
        raise ValueError("La trasformazione delle feature ha prodotto valori non finiti")
    labels = context.model.predict(scaled)
    distances = context.model.transform(scaled)
    output = df.copy()
    output["cluster"] = labels.astype(int)
    output["segmento"] = "Cluster " + output["cluster"].astype(str)
    output["distanza_centroide"] = np.round(distances[np.arange(len(df)), labels], 4)
    return output

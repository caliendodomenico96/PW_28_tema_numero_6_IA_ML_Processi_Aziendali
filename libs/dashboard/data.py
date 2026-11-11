# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Read-only layer tra gli artefatti K-Means e la dashboard Streamlit.

Non ricalcola K-Means: usa esclusivamente gli output dei training completati.
"""


from pathlib import Path
import json
import pandas as pd

from .validation_reports import validation_catalog_signature


NUMERIC_COLUMNS = (
    "numero_richieste", "importo_totale_speso", "importo_medio_speso",
    "giorni_ultima_richiesta", "giorni_da_iscrizione",
)
DATE_COLUMNS = ("data_iscrizione", "data_ultima_richiesta")


def completed_training_runs(project_root: Path) -> list[Path]:
    root = Path(project_root) / "results" / "kmeans"
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and
         (p / "training_report.json").is_file()),
        key=lambda p: p.name,
        reverse=True,
    )


def read_training_report(run_dir: Path) -> dict:
    path = Path(run_dir) / "training_report.json"
    with path.open(encoding="utf-8") as file:
        report = json.load(file)
    if not report.get("k_modelli_salvati"):
        raise ValueError("Il training selezionato non include modelli K-Means salvati")
    return report


def load_cluster_results(run_dir: Path, k: int) -> tuple[pd.DataFrame, dict]:
    """Legge le assegnazioni precomputate e i profili, verificando la struttura."""
    run_dir = Path(run_dir)
    report = read_training_report(run_dir)
    if int(k) not in [int(item) for item in report["k_modelli_salvati"]]:
        raise ValueError(f"K={k} non è fra i modelli salvati in {run_dir.name}")

    csv_path = run_dir / f"cluster_k{k}.csv"
    profile_path = run_dir / f"profili_cluster_k{k}.json"
    if not csv_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError(f"Mancano assegnazioni o profili per K={k}")

    df = pd.read_csv(csv_path, dtype={"id_cliente": "string"}, encoding="utf-8")
    with profile_path.open(encoding="utf-8") as file:
        profiles = json.load(file)
    required = {"id_cliente", "cluster", "zona", "categoria_preferita", "data_iscrizione",
                "numero_richieste", "importo_totale_speso", "giorni_ultima_richiesta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne assenti dal CSV: {sorted(missing)}")
    if df["id_cliente"].isna().any() or df["id_cliente"].duplicated().any():
        raise ValueError("ID cliente mancanti o duplicati")

    for field in DATE_COLUMNS:
        if field in df:
            df[field] = pd.to_datetime(df[field], format="%d/%m/%Y", errors="coerce")
    reference = pd.to_datetime(report["data_riferimento"], format="%d/%m/%Y", errors="raise")
    df["giorni_da_iscrizione"] = (reference - df["data_iscrizione"]).dt.days
    for col in NUMERIC_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cluster"] = pd.to_numeric(df["cluster"], errors="raise").astype("int64")
    df["segmento"] = "Cluster " + df["cluster"].astype(str)
    if len(df) != int(report["n_clienti"]):
        raise ValueError("Numero di record diverso da quello nel training_report.json")
    return df, profiles


def filter_clients(df: pd.DataFrame, zones: list[str] | None = None,
                   categories: list[str] | None = None,
                   date_field: str = "data_iscrizione",
                   start_date=None, end_date=None) -> pd.DataFrame:
    """Filtra solo la vista; le etichette cluster sono immutabili."""
    if date_field not in DATE_COLUMNS:
        raise ValueError(f"Data non supportata: {date_field}")
    mask = pd.Series(True, index=df.index)
    if zones is not None:
        mask &= df["zona"].isin(zones)
    if categories is not None:
        mask &= df["categoria_preferita"].isin(categories)
    if start_date is not None:
        mask &= df[date_field] >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= df[date_field] <= pd.Timestamp(end_date)
    return df.loc[mask].copy()


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["cluster", "clienti", "richieste_medie", "spesa_media_euro", "spesa_totale_euro",
               "recency_media_giorni", "anzianita_media_giorni", "percentuale_clienti"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    grouped = df.groupby("cluster", sort=True).agg(
        clienti=("id_cliente", "size"),
        richieste_medie=("numero_richieste", "mean"),
        spesa_media_euro=("importo_totale_speso", "mean"),
        spesa_totale_euro=("importo_totale_speso", "sum"),
        recency_media_giorni=("giorni_ultima_richiesta", "mean"),
        anzianita_media_giorni=("giorni_da_iscrizione", "mean"),
    ).reset_index()
    grouped["percentuale_clienti"] = 100.0 * grouped["clienti"] / len(df)
    return grouped[columns]


def available_dataset_folders(project_root: Path) -> list[Path]:
    """Generazioni complete disponibili, indipendenti dagli esperimenti K-Means."""
    data_dir = Path(project_root) / "data"
    if not data_dir.is_dir():
        return []
    return sorted(
        (folder for folder in data_dir.iterdir()
         if folder.is_dir()
         and (folder / "customer_history_dataset.csv").is_file()
         and (folder / "customer_history_dataset.metadati.json").is_file()),
        key=lambda folder: folder.name,
        reverse=True,
    )


def training_source_path(project_root: Path, report: dict) -> Path:
    """Percorso del dataset collegato al training; non ricostruito dal tag UTC."""
    path = Path(report["dataset_csv"])
    if not path.is_absolute():
        path = Path(project_root) / path
    return path


def training_run_status(project_root: Path, run_dir: Path) -> tuple[bool, str]:
    """Controlli leggeri per non mostrare run inutilizzabili se cancellati a mano."""
    try:
        report = read_training_report(run_dir)
        dataset = training_source_path(project_root, report)
        if not dataset.is_file() or not dataset.with_suffix(".metadati.json").is_file():
            return False, "Il dataset o i suoi metadati sono stati rimossi"
        models = list(report["k_modelli_salvati"])
        if not models:
            return False, "Nessun modello salvato"
        if not (run_dir / "models" / "preprocessor.joblib").is_file():
            return False, "Preprocessing mancante"
        for k in models:
            if not all((run_dir / path).is_file() for path in (
                f"cluster_k{k}.csv",
                f"profili_cluster_k{k}.json",
                f"models/kmeans_k{k}.joblib",
            )):
                return False, f"Artefatti K={k} incompleti"
        return True, ""
    except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        return False, f"Training non leggibile: {exc}"


def usable_training_runs(project_root: Path) -> list[Path]:
    """I run validi per entrambe le pagine: inferenza inclusa."""
    return [run for run in completed_training_runs(project_root)
            if training_run_status(project_root, run)[0]]


def dashboard_catalog_snapshot(project_root: Path) -> tuple:
    """Rileva aggiunte E rimozioni di dataset e training, senza caching."""
    runs = completed_training_runs(project_root)
    statuses = tuple((run.name, *training_run_status(project_root, run)) for run in runs)
    folders = tuple(folder.name for folder in available_dataset_folders(project_root))
    validations = tuple((run.name, validation_catalog_signature(run)) for run in runs)
    return statuses, folders, validations

# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Caricamento e validazione della configurazione K-Means (solo INI)."""


from dataclasses import dataclass
from pathlib import Path
import configparser


FEATURES_CONSENTITE = {
    "numero_richieste", "importo_totale_speso",
    "giorni_ultima_richiesta", "giorni_da_iscrizione",
}


@dataclass(frozen=True)
class KMeansConfig:
    dataset_csv: str
    output_dir: Path
    features: tuple[str, ...]
    k_min: int
    k_max: int
    k_confronto: tuple[int, ...]
    random_state: int
    n_init: int
    log1p_spesa: bool
    training_run: str


def load_kmeans_config(config_path: Path, project_root: Path, ini=None) -> KMeansConfig:
    config_path = Path(config_path)
    project_root = Path(project_root)
    if ini is None:
        ini = configparser.ConfigParser(interpolation=None)
    if not ini.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"Configurazione K-Means non trovata: {config_path}")
    for section in ("KMEANS", "VALIDATION"):
        if not ini.has_section(section):
            raise ValueError(f"Sezione obbligatoria mancante: [{section}]")

    features = tuple(x.strip() for x in ini.get("KMEANS", "features").split(",") if x.strip())
    if not features or len(set(features)) != len(features):
        raise ValueError("[KMEANS] features non può essere vuoto o avere duplicati")
    unknown = set(features) - FEATURES_CONSENTITE
    if unknown:
        raise ValueError(f"Feature non ammesse: {sorted(unknown)}. Le feature anagrafiche non sono consentite.")

    k_min = ini.getint("KMEANS", "k_min")
    k_max = ini.getint("KMEANS", "k_max")
    k_confronto = tuple(int(x.strip()) for x in ini.get("KMEANS", "k_confronto").split(",") if x.strip())
    if not 2 <= k_min < k_max:
        raise ValueError("k_min deve essere >= 2 e inferiore a k_max")
    if len(set(k_confronto)) < 2:
        raise ValueError("Indicare almeno due K distinti in k_confronto")
    if any(not k_min <= k <= k_max for k in k_confronto):
        raise ValueError("I K di confronto devono essere nell'intervallo k_min..k_max")
    if ini.getint("KMEANS", "n_init") < 1:
        raise ValueError("n_init deve essere almeno 1")

    output_dir = Path(ini.get("KMEANS", "output_dir").strip()).expanduser()
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if not output_dir.name:
        raise ValueError("output_dir non valido")

    dataset_csv = ini.get("KMEANS", "dataset_csv").strip()
    training_run = ini.get("VALIDATION", "training_run").strip()
    if not dataset_csv or not training_run:
        raise ValueError("dataset_csv e training_run non possono essere vuoti")

    return KMeansConfig(
        dataset_csv=dataset_csv,
        output_dir=output_dir.resolve(),
        features=features,
        k_min=k_min,
        k_max=k_max,
        k_confronto=tuple(sorted(set(k_confronto))),
        random_state=ini.getint("KMEANS", "random_state"),
        n_init=ini.getint("KMEANS", "n_init"),
        log1p_spesa=ini.getboolean("KMEANS", "log1p_spesa"),
        training_run=training_run,
    )
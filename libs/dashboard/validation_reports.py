# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================
""" Lettura read-only dei risultati di validation per la dashboard """


import json
from pathlib import Path


def completed_validation_runs(training_dir: Path) -> list[Path]:
    """Restituisce le validation concluse, dalla più recente alla più vecchia.

    Il report viene scritto per ultimo dalla pipeline: evita cartelle parziali.
    """
    validations_root = Path(training_dir) / "validations"
    if not validations_root.is_dir():
        return []
    return sorted(
        (folder for folder in validations_root.iterdir()
         if folder.is_dir() and (folder / "validation_report.json").is_file()),
        key=lambda folder: folder.name,
        reverse=True,
    )


def read_validation_report(validation_dir: Path) -> dict:
    """Carica il JSON, controllandone i campi minimi per il confronto."""
    with (Path(validation_dir) / "validation_report.json").open(encoding="utf-8") as file:
        report = json.load(file)
    if not isinstance(report, dict) or not isinstance(report.get("metriche"), list):
        raise ValueError("Report di validazione privo delle metriche")
    return report


def latest_validation_for_k(training_dir: Path, k: int) -> tuple[Path, dict] | None:
    """Trova l'ultima validation COMPLETA contenente la silhouette del K scelto."""
    for validation_dir in completed_validation_runs(training_dir):
        try:
            report = read_validation_report(validation_dir)
            ks = {int(metric["k"]) for metric in report["metriche"]}
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if int(k) in ks and (validation_dir / f"silhouette_k{int(k)}.png").is_file():
            return validation_dir, report
    return None


def validation_catalog_signature(training_dir: Path) -> tuple:
    """Firma minimale per rilevare anche aggiunte/rimozioni con il refresh a 10 s."""
    signature = []
    for validation_dir in completed_validation_runs(training_dir):
        report = validation_dir / "validation_report.json"
        try:
            signature.append((validation_dir.name, report.stat().st_mtime_ns,
                              tuple(sorted(p.name for p in validation_dir.glob("silhouette_k*.png")))))
        except OSError:
            continue
    return tuple(signature)


def validation_metrics_rows(report: dict) -> list[dict]:
    """Restituisce le metriche in formato tabellare senza ricalcolarle."""
    result = []
    for metric in report["metriche"]:
        result.append({
            "K": int(metric["k"]),
            "Silhouette": float(metric["silhouette"]),
            "Davies-Bouldin": float(metric["davies_bouldin"]),
            "Calinski-Harabasz": float(metric["calinski_harabasz"]),
            "Clienti cluster min.": int(metric["cluster_minimo"]),
        })
    return sorted(result, key=lambda row: row["K"])

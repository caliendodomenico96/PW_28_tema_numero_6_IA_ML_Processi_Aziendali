# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Path, identificatori UTC e serializzazione riutilizzabili."""


import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def unique_utc_dir(parent: Path) -> Path:
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
        candidate = parent / timestamp
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError("Non è stato possibile creare una cartella UTC univoca")


def json_write(path: Path, payload: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def json_read(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_dataset(project_root: Path, selection: str) -> Path:
    project_root = Path(project_root)
    if selection.lower() == "latest":
        candidates = [p for p in (project_root / "data").glob("*/customer_history_dataset.csv")
                      if p.with_suffix(".metadati.json").is_file()]
        if not candidates:
            raise FileNotFoundError("Non trovo dataset in data/<timestamp_UTC>/ con metadati JSON")
        return sorted(candidates, key=lambda p: p.parent.name)[-1].resolve()
    path = Path(selection).expanduser()
    path = path if path.is_absolute() else project_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV non trovato: {path}")
    return path.resolve()


def resolve_training_run(config, project_root: Path) -> Path:
    if config.training_run.lower() == "latest":
        candidates = [p for p in config.output_dir.iterdir() if p.is_dir() and (p / "training_report.json").is_file()] \
            if config.output_dir.exists() else []
        if not candidates:
            raise FileNotFoundError("Non è presente un training completo in results/kmeans")
        return sorted(candidates, key=lambda p: p.name)[-1]
    path = Path(config.training_run).expanduser()
    if not path.is_absolute():
        # Percorso dichiarato rispetto alla root del progetto, non a results/kmeans.
        path = Path(project_root) / path
    if not (path / "training_report.json").is_file():
        raise FileNotFoundError(f"Training non trovato: {path}")
    return path.resolve()

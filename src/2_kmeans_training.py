# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Preprocessing + training + scelta motivata di K (interamente da file INI)."""



import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.utils.helpers import *
from libs.clustering.configuration import load_kmeans_config  # noqa: E402
from libs.clustering.training import train_kmeans  # noqa: E402


if __name__ == "__main__":

    config_path = PROJECT_ROOT / "config" / "kmeans_config.ini"
    config_ini = get_config(config_path)

    logging_parsing("2_kmeans_training", config_ini)
    cfg = load_kmeans_config(config_path, PROJECT_ROOT, config_ini)
    run_dir = train_kmeans(PROJECT_ROOT, cfg)
    logging.info(f"Training K-Means completato: {run_dir}")

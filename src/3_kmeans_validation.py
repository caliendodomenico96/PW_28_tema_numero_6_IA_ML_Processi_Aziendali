# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Validazione separata dei modelli già addestrati (interamente da file INI)."""


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.utils.helpers import *
from libs.clustering.configuration import load_kmeans_config
from libs.clustering.validation import validate_kmeans


if __name__ == "__main__":

    config_path = PROJECT_ROOT / "config" / "kmeans_config.ini"
    config_ini = get_config(config_path)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = load_kmeans_config(config_path, PROJECT_ROOT, config_ini)
    validation_dir = validate_kmeans(PROJECT_ROOT, cfg)
    logging.info(f"Validazione K-Means completata: {validation_dir}")

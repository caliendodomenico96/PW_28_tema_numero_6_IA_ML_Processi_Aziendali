# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


"""Generatore di dati sintetici per la segmentazione dei clienti."""


import hashlib
import json
import logging
import os.path
import platform
import pandas as pd
from data_profiling import ProfileReport
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.dataset.dataset_tool import (  # noqa: E402
    config_validator,
    dataset_generation,
    date_output_format,
    load_dataset_configuration,
    riepilogo_comportamenti,
)
from libs.utils.helpers import (  # noqa: E402
    logging_parsing,
    parsing_config_output,
    store_csv,
    store_dataset_recap,
)


CAMPI_CLIENTI_BASE = [
    "id_cliente", "zona", "data_iscrizione", "numero_richieste",
    "importo_totale_speso", "importo_medio_speso", "giorni_ultima_richiesta",
    "categoria_preferita", "data_ultima_richiesta",
]
CAMPI_ANAGRAFICA = ["nome", "cognome", "email"]
CAMPI_TRANSAZIONI = ["id_richiesta", "id_cliente", "data_richiesta", "categoria", "importo_euro"]


if __name__ == "__main__":

    config_path = PROJECT_ROOT / "config" / "dataset_generator_config.ini"
    ini, cfg = load_dataset_configuration(config_path)
    config_validator(cfg)
    output, richieste_path, metadati_path, separatore, includi_anagrafica = parsing_config_output(
        PROJECT_ROOT, config_path, ini
    )

    # Un unico timestamp UTC per tutti i file di una stessa generazione.
    istante_generazione_utc = datetime.now(timezone.utc)
    timestamp = istante_generazione_utc.strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    cartella_esecuzione = output.parent / timestamp
    output = cartella_esecuzione / output.name
    metadati_path = cartella_esecuzione / metadati_path.name
    if richieste_path is not None:
        richieste_path = cartella_esecuzione / richieste_path.name

    destinazioni = [output, metadati_path]
    if richieste_path is not None:
        destinazioni.append(richieste_path)
    if len(set(destinazioni)) != len(destinazioni):
        raise ValueError("I file di output devono avere nomi differenti nella cartella della generazione")
    cartella_esecuzione.mkdir(parents=True, exist_ok=False)

    logging_parsing("1_dataset_tool_generator", ini)
    logging.info("Generazione di %s clienti; seed=%s; configurazione=%s",
                 cfg["n_clienti"], cfg["seed"], config_path)
    if not 300 <= cfg["n_clienti"] <= 600:
        logging.warning("La traccia richiede una numerosità compresa tra 300 e 600 clienti")

    clienti, richieste = dataset_generation(cfg, includi_anagrafica)
    campi_clienti = (["id_cliente", *CAMPI_ANAGRAFICA, *CAMPI_CLIENTI_BASE[1:]]
                     if includi_anagrafica else CAMPI_CLIENTI_BASE)
    output_dataset_path = store_csv(output, clienti, campi_clienti, separatore)
    if richieste_path is not None:
        store_csv(richieste_path, richieste, CAMPI_TRANSAZIONI, separatore)

    riferimento = date.fromisoformat(cfg["data_riferimento"])
    comportamenti = riepilogo_comportamenti(clienti, cfg)
    metadati = {
        "versione_generatore": "2.5.0",
        "timestamp_generazione_utc": istante_generazione_utc.isoformat(
            timespec="microseconds").replace("+00:00", "Z"),
        "versione_python": platform.python_version(),
        "configurazione": {**cfg, "data_riferimento": date_output_format(riferimento)},
        "contesto_geografico": {"nazione": ini.get("GEOGRAFIA", "nazione").strip()},
        "separatore_csv": separatore,
        "anagrafica_sintetica": {
            "inclusa": includi_anagrafica,
            "dominio_email": "example.com" if includi_anagrafica else None,
            "uso_previsto": "Identificazione e presentazione; esclusa dal clustering",
        },
        "inizio_osservazione_incluso": (
                riferimento - timedelta(days=cfg["giorni_osservazione"] - 1)
        ).strftime("%d/%m/%Y"),
        "fine_osservazione_inclusa": date_output_format(riferimento),
        "n_clienti": len(clienti),
        "n_interventi_storici": len(richieste),
        "n_clienti_senza_richieste_nel_periodo": sum(
            c["numero_richieste"] == 0 for c in clienti),
        "distribuzione_comportamenti_osservati": comportamenti,
        "sha256_csv": hashlib.sha256(output.read_bytes()).hexdigest(),
        "note": [
            "Dati completamente sintetici, privi di informazioni personali reali.",
            "Nomi e cognomi sono combinazioni fittizie; email univoche sul dominio example.com.",
            "Una richiesta è un intervento completato e pagato; almeno un acquisto storico per cliente.",
            "Numero richieste, spesa totale e media usano la stessa finestra inclusiva.",
            "Media zero senza interventi nel periodo: convenzione, non prezzo osservato.",
            "Recency e categoria preferita usano la storia completa.",
            "Le fasce nel JSON derivano da dati osservati e NON rappresentano etichette vere di clustering.",
            "Recenti e ticket elevato sono attributi trasversali, non profili esclusivi.",
            "Inattivo nel periodo significa nessuna richiesta negli ultimi giorni_osservazione giorni.",
            "Nessuna etichetta latente di profilo è esportata nel CSV clienti.",
        ],
    }
    metadati_path.write_text(json.dumps(metadati, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")

    logging.info("Cartella generazione (UTC): %s", cartella_esecuzione)
    logging.info("CSV clienti: %s", output)
    logging.info("JSON metadati: %s", metadati_path)
    if richieste_path is not None:
        logging.info("CSV interventi storici: %s", richieste_path)
    logging.info("Distribuzione richieste osservate: %s",
                 {k: v["clienti"] for k, v in comportamenti["fasce_frequenza_esclusive"].items()})

    # Salva il riepilogo JSON (senza output in console).
    recap_path = store_dataset_recap(
        clienti, richieste, metadati, output,
        metadati_path=metadati_path, richieste_path=richieste_path,
    )
    logging.info("JSON recap: %s", recap_path)

    df = pd.read_csv(output_dataset_path)

    profile = ProfileReport(
        df,
        title="Customer Segmentation - EDA",
        explorative=True
    )

    profile.to_file(os.path.join(str(output_dataset_path).split("customer_history_dataset.csv")[0],"dataset_profiling.html"))
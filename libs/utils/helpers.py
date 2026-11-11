# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


""" Helpers functions """


import time
import os
import configparser
from pathlib import Path
import logging
from datetime import datetime
import csv
import json
from collections import Counter
from decimal import Decimal


def get_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def logging_parsing(script_name, config):
    severity_log = config["LOG"]["severity"]
    if config["LOG"]["store_log_file"] == "yes":
        log_file_name = datetime.now().strftime(script_name + '_%H_%M_%S_%d_%m_%Y.log')
        Path('log').mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=eval(severity_log), filename=os.path.join('log', log_file_name), filemode='w',
                            format='[%(levelname)s %(funcName)s():%(lineno)d %(asctime)s] %(message)s')
    else:
        logging.basicConfig(level=eval(severity_log),
                            format='[%(levelname)s %(funcName)s():%(lineno)d %(asctime)s] %(message)s')

    logging.Formatter.converter = time.gmtime


def store_csv(output_path, righe, campi, separatore=","):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=campi, delimiter=separatore, lineterminator="\n")
        writer.writeheader()
        writer.writerows(righe)
    return output_path


def project_path(project_root, value):
    """Risolve i percorsi relativi rispetto alla root del progetto."""
    project_root = Path(project_root)
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def parsing_config_output(project_root, config_path, ini):
    """Prepara i file da scrivere usando soltanto la sezione [OUTPUT] dell'INI."""
    project_root = Path(project_root)
    config_path = Path(config_path)
    sezione = "OUTPUT"
    obbligatori = {"csv_clienti", "esporta_richieste", "separatore_csv"}
    ammessi = obbligatori | {"csv_richieste", "includi_anagrafica"}
    sconosciuti = set(ini[sezione]) - ammessi
    mancanti = obbligatori - set(ini[sezione])
    if sconosciuti or mancanti:
        raise ValueError(
            f"[{sezione}]: chiavi sconosciute {sorted(sconosciuti)}; "
            f"mancanti {sorted(mancanti)}"
        )

    csv_clienti = ini.get(sezione, "csv_clienti").strip()
    if not csv_clienti:
        raise ValueError("[OUTPUT] csv_clienti non può essere vuoto")

    output = project_path(project_root, csv_clienti)

    esporta_richieste = ini.getboolean(sezione, "esporta_richieste")
    csv_richieste = ini.get(sezione, "csv_richieste", fallback="").strip()
    if esporta_richieste and not csv_richieste:
        raise ValueError("[OUTPUT] csv_richieste è obbligatorio quando esporta_richieste = yes")
    richieste_path = project_path(project_root, csv_richieste) if esporta_richieste else None

    separatore = ini.get(sezione, "separatore_csv").strip()
    if separatore not in (",", ";"):
        raise ValueError("[OUTPUT] separatore_csv deve essere ',' oppure ';'")

    metadati_path = output.with_suffix(".metadati.json")
    destinazioni = [output, metadati_path]
    if richieste_path is not None:
        destinazioni.append(richieste_path)
    if len(destinazioni) != len(set(destinazioni)):
        raise ValueError("I percorsi di output devono essere distinti")
    if output.suffix.lower() != ".csv" or (richieste_path and richieste_path.suffix.lower() != ".csv"):
        raise ValueError("I file dati devono avere estensione .csv")

    sources = {config_path.resolve(), Path(__file__).resolve(), (project_root / "libs/utils/helpers.py").resolve()}
    if any(p in sources for p in destinazioni):
        raise ValueError("Un output non può sovrascrivere configurazione o codice")

    includi_anagrafica = ini.getboolean(sezione, "includi_anagrafica", fallback=True)
    return output, richieste_path, metadati_path, separatore, includi_anagrafica


def store_dataset_recap(clienti, richieste, metadati, output,
                        metadati_path=None, richieste_path=None):
    """Salva un recap JSON centrato sui quattro comportamenti della traccia.

    I conteggi sono osservazioni sul CSV, non etichette per K-Means.
    Recenti è una caratteristica trasversale alle fasce di frequenza.
    Restituisce il percorso del file JSON senza modificare il dataset.
    """
    output = Path(output)
    recap_path = output.with_suffix(".recap.json")

    totale_clienti = len(clienti)
    richieste_periodo = sum(int(c["numero_richieste"]) for c in clienti)
    spesa_periodo = sum(
        (Decimal(str(c["importo_totale_speso"])) for c in clienti),
        Decimal("0.00"),
    )
    clienti_con_richieste = sum(int(c["numero_richieste"]) > 0 for c in clienti)
    ticket_medio = spesa_periodo / richieste_periodo if richieste_periodo else Decimal("0")
    spesa_media_cliente = spesa_periodo / totale_clienti if totale_clienti else Decimal("0")
    richieste_medie_cliente = richieste_periodo / totale_clienti if totale_clienti else 0

    def euro_json(numero):
        # Un numero nel JSON, senza simbolo di valuta e con due decimali.
        return float(Decimal(numero).quantize(Decimal("0.01")))

    zone = Counter(c["zona"] for c in clienti)
    categorie = Counter(c["categoria_preferita"] for c in clienti)
    zona_top = sorted(zone.items(), key=lambda x: (-x[1], x[0]))[0] if zone else None
    categoria_top = sorted(categorie.items(), key=lambda x: (-x[1], x[0]))[0] if categorie else None

    osservati = metadati["distribuzione_comportamenti_osservati"]
    fasce = osservati["fasce_frequenza_esclusive"]
    coorti = osservati["coorti_trasversali"]
    criteri = osservati["criteri"]

    # La logica originale distingue tre fasce positive: occasionale,
    # attivo intermedio e molto attivo. "Recente" si basa invece
    # sulla data d'iscrizione e può sovrapporsi a tutte le fasce.
    inattivi = fasce["inattivi_0_richieste_nel_periodo"]
    occasionali = next(v for k, v in fasce.items() if k.startswith("occasionali_"))
    attivi = next(v for k, v in fasce.items() if k.startswith("attivi_"))
    molto_attivi = next(v for k, v in fasce.items() if k.startswith("molto_attivi_"))
    recenti = coorti["recenti_iscritti_nel_periodo_definito"]
    ticket_elevato = coorti["ticket_medio_elevato_nel_periodo"]
    soglie_frequenza = [t.strip() for t in criteri["fasce_frequenza_richieste"].split(";")]

    def con_criterio(dati, criterio):
        return {"clienti": dati["clienti"], "percentuale": dati["percentuale"], "criterio": criterio}

    # Quattro indicatori richiesti dalla traccia in primo piano.
    # Le ulteriori analisi non vengono confuse con i requisiti didattici.
    recap = {
        "timestamp_generazione_utc": metadati.get("timestamp_generazione_utc"),
        "periodo_osservazione": {
            "inizio_incluso": metadati["inizio_osservazione_incluso"],
            "fine_inclusa": metadati["fine_osservazione_inclusa"],
        },
        "riepilogo_generale": {
            "clienti_generati": totale_clienti,
            "clienti_con_richieste_nel_periodo": clienti_con_richieste,
            "clienti_senza_richieste_nel_periodo": totale_clienti - clienti_con_richieste,
            "interventi_nel_periodo": richieste_periodo,
            "interventi_nella_storia": len(richieste),
            "media_richieste_per_cliente_nel_periodo": round(richieste_medie_cliente, 2),
            "spesa_complessiva_nel_periodo_euro": euro_json(spesa_periodo),
            "spesa_media_per_cliente_nel_periodo_euro": euro_json(spesa_media_cliente),
            "ticket_medio_per_intervento_nel_periodo_euro": euro_json(ticket_medio),
        },
        "comportamenti_richiesti_dalla_traccia": {
            "molto_attivi": con_criterio(
                molto_attivi, f"Almeno {soglie_frequenza[3].removeprefix('almeno ').strip()} richieste nel periodo"
            ),
            "occasionali": con_criterio(
                occasionali, f"{soglie_frequenza[1]} richieste nel periodo"
            ),
            "recenti": con_criterio(
                recenti,
                f"Iscritti da meno di {criteri['recenti_se_iscritti_da_meno_di_giorni']} giorni",
            ),
            "inattivi": con_criterio(
                inattivi, f"0 richieste negli ultimi {criteri['finestra_giorni']} giorni"
            ),
        },
        "analisi_aggiuntive": {
            "attivi_intermedi": con_criterio(
                attivi, f"{soglie_frequenza[2]} richieste nel periodo"
            ),
            "ticket_medio_elevato": con_criterio(
                ticket_elevato,
                f"Ticket medio almeno {criteri['ticket_medio_elevato_da_euro']} euro, con richieste nel periodo",
            ),
        },
        "metodologia_comportamenti": {
            "finestra_osservazione_giorni": criteri["finestra_giorni"],
            "fasce_di_frequenza_esclusive": ["inattivi", "occasionali", "attivi_intermedi", "molto_attivi"],
            "recenti_trasversale": True,
            "note": (
                "Le quattro categorie della traccia non sono mutuamente esclusive: "
                "un cliente recente può appartenere a una fascia di frequenza. "
                "Perciò non sommare le loro percentuali."
            ),
        },
        "distribuzione_geografica": dict(sorted(zone.items())),
        "distribuzione_categorie_preferite": dict(sorted(categorie.items())),
        "citta_piu_rappresentata": (
            {"nome": zona_top[0], "clienti": zona_top[1]} if zona_top else None
        ),
        "categoria_preferita_piu_comune": (
            {"nome": categoria_top[0], "clienti": categoria_top[1]} if categoria_top else None
        ),
        "file_generati": {
            "csv_clienti": output.name,
            "json_metadati": Path(metadati_path).name if metadati_path is not None else None,
            "csv_richieste_storiche": (
                Path(richieste_path).name if richieste_path is not None else None
            ),
            "json_recap": recap_path.name,
        },
        "note": [
            "I quattro comportamenti richiesti dalla traccia sono riportati per primi.",
            "Recenti e ticket medio elevato sono caratteristiche trasversali sovrapponibili.",
            "La fascia intermedia attivi è un'analisi aggiuntiva, non richiesta dalla traccia.",
            "Le classificazioni descrittive non costituiscono etichette reali per K-Means.",
            "Gli importi del riepilogo si riferiscono alla finestra di osservazione.",
        ],
    }

    # Salva nella stessa directory UTC del CSV senza sovrascrivere file già presenti.
    # La cartella viene creata dal main al momento della generazione.
    with recap_path.open("x", encoding="utf-8") as file:
        json.dump(recap, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return recap_path
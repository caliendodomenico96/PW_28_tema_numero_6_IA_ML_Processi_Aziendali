# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================


""" Dataset Tool functions """


import configparser
from datetime import date, datetime, timedelta, timezone
from collections import Counter
from pathlib import Path
import copy
import math
import random
from decimal import Decimal, ROUND_HALF_UP
from libs.dataset.anagrafica import genera_anagrafica, rng_anagrafica
from libs.utils.math import lognorm



INTERI = ("n_clienti", "seed", "giorni_osservazione", "giorni_attivita_azienda", "giorni_cliente_nuovo")
DECIMALI = ("quota_nuovi_clienti", "variabilita_frequenza", "variabilita_ticket_cliente",
                "variabilita_ticket_richiesta", "probabilita_categoria_preferita")
CONFIG_KEYS = set(INTERI + DECIMALI) | {
    "data_riferimento", "zone", "categorie", "profili_simulazione", "soglie_descrittive", "valore_economico"
}

def load_dataset_configuration(percorso):

    ini = configparser.ConfigParser(interpolation=None)
    ini.optionxform = str

    with Path(percorso).open(encoding="utf-8-sig") as file:
        ini.read_file(file)
    basi = {"LOG", "DATASET_GENERATOR", "OUTPUT", "GEOGRAFIA", "ZONE", "SEMANTIC_ANALYSIS", "ANALISI_COMPORTAMENTI", "VALORE_ECONOMICO"}

    for sezione in ("DATASET_GENERATOR", "OUTPUT", "GEOGRAFIA", "ZONE"):
        if not ini.has_section(sezione):
            raise ValueError(f"Sezione INI mancante: [{sezione}]")

    for sezione in ini.sections():
        if sezione not in basi and not sezione.startswith(("CATEGORIA:", "PROFILO:")):
            raise ValueError(f"Sezione INI sconosciuta: [{sezione}]")

    extra = set(ini["DATASET_GENERATOR"]) - set(INTERI + DECIMALI) - {"data_riferimento"}
    if extra:
        raise ValueError(f"Parametri DATASET_GENERATOR sconosciuti: {sorted(extra)}")
    cfg = {nome: ini.getint("DATASET_GENERATOR", nome) for nome in INTERI}
    cfg.update({nome: ini.getfloat("DATASET_GENERATOR", nome) for nome in DECIMALI})
    cfg["data_riferimento"] = datetime.strptime(ini.get("DATASET_GENERATOR", "data_riferimento").strip(), "%d-%m-%Y").date().isoformat()
    cfg["zone"] = {nome: float(peso) for nome, peso in ini.items("ZONE")}
    sez_analisi = "ANALISI_COMPORTAMENTI"
    if not ini.has_section(sez_analisi):
        raise ValueError(f"Sezione INI mancante: [{sez_analisi}]")
    chiavi_analisi = {
        "richieste_min_attivi", "richieste_min_molto_attivi", "ticket_medio_elevato_euro"
    }
    if set(ini[sez_analisi]) != chiavi_analisi:
        raise ValueError(f"[{sez_analisi}]: chiavi richieste {sorted(chiavi_analisi)}")
    cfg["soglie_descrittive"] = {
        "richieste_min_attivi": ini.getint(sez_analisi, "richieste_min_attivi"),
        "richieste_min_molto_attivi": ini.getint(sez_analisi, "richieste_min_molto_attivi"),
        "ticket_medio_elevato_euro": ini.getfloat(sez_analisi, "ticket_medio_elevato_euro"),
    }
    if not ini.has_section("VALORE_ECONOMICO"):
        raise ValueError("Sezione INI mancante: [VALORE_ECONOMICO]")
    chiavi_valore = {"quota_clienti_ticket_premium", "moltiplicatore_ticket_premium"}
    if set(ini["VALORE_ECONOMICO"]) != chiavi_valore:
        raise ValueError(f"[VALORE_ECONOMICO]: chiavi richieste {sorted(chiavi_valore)}")
    cfg["valore_economico"] = {
        "quota_clienti_ticket_premium": ini.getfloat("VALORE_ECONOMICO", "quota_clienti_ticket_premium"),
        "moltiplicatore_ticket_premium": ini.getfloat("VALORE_ECONOMICO", "moltiplicatore_ticket_premium"),
    }
    cfg["categorie"], cfg["profili_simulazione"] = {}, {}

    for sezione in ini.sections():
        if sezione.startswith("CATEGORIA:"):
            nome = sezione.split(":", 1)[1].strip()
            campi = {"peso", "fattore_prezzo"}
            if set(ini[sezione]) - campi or nome in cfg["categorie"]:
                raise ValueError(f"Categoria duplicata o campi sconosciuti: [{sezione}]")
            cfg["categorie"][nome] = {k: ini.getfloat(sezione, k) for k in sorted(campi)}
        elif sezione.startswith("PROFILO:"):
            nome = sezione.split(":", 1)[1].strip()
            campi = {"peso", "richieste_mese", "ticket_medio_euro"}
            opzionali = {"inattivita_giorni_min", "inattivita_giorni_max"}
            if set(ini[sezione]) - campi - opzionali or nome in cfg["profili_simulazione"]:
                raise ValueError(f"Profilo duplicato o campi sconosciuti: [{sezione}]")
            valori = {k: ini.getfloat(sezione, k) for k in sorted(campi)}
            for k in sorted(opzionali):
                if ini.has_option(sezione, k):
                    valori[k] = ini.getint(sezione, k)
            cfg["profili_simulazione"][nome] = valori

    if not ini.get("GEOGRAFIA", "nazione").strip():
        raise ValueError("GEOGRAFIA.nazione non può essere vuota")

    return ini, cfg


def config_validator(config):
    """ Rifiuta parametri incoerenti invece di correggerli """

    sconosciuti = set(config) - set(CONFIG_KEYS)
    mancanti = set(CONFIG_KEYS) - set(config)
    if sconosciuti or mancanti:
        raise ValueError(f"Chiavi sconosciute: {sorted(sconosciuti)}; mancanti: {sorted(mancanti)}")
    for nome in ("n_clienti", "giorni_osservazione", "giorni_attivita_azienda", "giorni_cliente_nuovo"):
        _numero(config[nome], nome, minimo=1, intero=True)
    _numero(config["seed"], "seed", intero=True)
    date.fromisoformat(config["data_riferimento"])
    if config["giorni_attivita_azienda"] < config["giorni_osservazione"]:
        raise ValueError("giorni_attivita_azienda deve essere >= giorni_osservazione")
    if config["giorni_cliente_nuovo"] >= config["giorni_attivita_azienda"]:
        raise ValueError("giorni_cliente_nuovo deve essere < giorni_attivita_azienda")
    for nome in ("quota_nuovi_clienti", "probabilita_categoria_preferita"):
        _numero(config[nome], nome, massimo=1)
    for nome in ("variabilita_frequenza", "variabilita_ticket_cliente", "variabilita_ticket_richiesta"):
        _numero(config[nome], nome, massimo=2)
    soglie = config["soglie_descrittive"]
    chiavi_soglie = {"richieste_min_attivi", "richieste_min_molto_attivi", "ticket_medio_elevato_euro"}
    if not isinstance(soglie, dict) or set(soglie) != chiavi_soglie:
        raise ValueError(f"soglie_descrittive: richieste {sorted(chiavi_soglie)}")
    _numero(soglie["richieste_min_attivi"], "richieste_min_attivi", minimo=2, intero=True)
    _numero(soglie["richieste_min_molto_attivi"], "richieste_min_molto_attivi",
            minimo=soglie["richieste_min_attivi"] + 1, intero=True)
    _numero(soglie["ticket_medio_elevato_euro"], "ticket_medio_elevato_euro", minimo=0.01)
    valore = config["valore_economico"]
    if not isinstance(valore, dict) or set(valore) != {
        "quota_clienti_ticket_premium", "moltiplicatore_ticket_premium"
    }:
        raise ValueError("valore_economico: parametri premium mancanti o sconosciuti")
    _numero(valore["quota_clienti_ticket_premium"], "quota_clienti_ticket_premium", massimo=1)
    _numero(valore["moltiplicatore_ticket_premium"], "moltiplicatore_ticket_premium", minimo=1)
    # La nomenclatura della simulazione segue i comportamenti richiesti dalla traccia.
    profili_attesi = {"molto_attivi", "abituali", "occasionali", "inattivi"}
    if not profili_attesi <= set(config["profili_simulazione"]):
        raise ValueError(f"profili_simulazione: obbligatori {sorted(profili_attesi)}")
    if not (config["profili_simulazione"]["inattivi"].get("inattivita_giorni_min", 0)
            >= config["giorni_osservazione"]):
        raise ValueError("Il profilo inattivi deve fermarsi almeno da giorni_osservazione giorni")
    _pesi(config["zone"], "zone")
    for sezione, obbligatori in (
        ("categorie", {"peso", "fattore_prezzo"}),
        ("profili_simulazione", {"peso", "richieste_mese", "ticket_medio_euro"}),
    ):
        if not isinstance(config[sezione], dict) or not config[sezione]:
            raise ValueError(f"{sezione}: dizionario non vuoto richiesto")
        for nome, valori in config[sezione].items():
            if not isinstance(valori, dict) or not obbligatori <= set(valori):
                raise ValueError(f"{sezione}.{nome}: campi obbligatori {sorted(obbligatori)}")
            ammessi = obbligatori | ({"inattivita_giorni_min", "inattivita_giorni_max"} if sezione == "profili_simulazione" else set())
            if set(valori) - ammessi:
                raise ValueError(f"{sezione}.{nome}: campi sconosciuti {sorted(set(valori) - ammessi)}")
            for campo in obbligatori - {"peso"}:
                _numero(valori[campo], f"{nome}.{campo}", minimo=0.000001)
            if "inattivita_giorni_min" in valori or "inattivita_giorni_max" in valori:
                lo = valori.get("inattivita_giorni_min", 0)
                hi = valori.get("inattivita_giorni_max", lo)
                _numero(lo, f"{nome}.inattivita_giorni_min", intero=True)
                _numero(hi, f"{nome}.inattivita_giorni_max", minimo=lo, intero=True)
                if hi >= config["giorni_attivita_azienda"]:
                    raise ValueError(f"{nome}: inattivita massima deve essere < giorni_attivita_azienda")
        _pesi({k: v["peso"] for k, v in config[sezione].items()}, sezione)


def _numero(valore, nome, minimo=0, massimo=None, intero=False):
    if isinstance(valore, bool) or not isinstance(valore, (int, float)):
        raise ValueError(f"{nome}: richiesto un numero")
    if intero and not isinstance(valore, int):
        raise ValueError(f"{nome}: richiesto un intero")
    if not math.isfinite(valore) or valore < minimo or (massimo is not None and valore > massimo):
        raise ValueError(f"{nome}: valore fuori intervallo")


def _pesi(pesi, nome):
    if not isinstance(pesi, dict) or not pesi:
        raise ValueError(f"{nome}: richiesto un dizionario non vuoto")
    for etichetta, peso in pesi.items():
        if not isinstance(etichetta, str) or not etichetta.strip():
            raise ValueError(f"{nome}: nomi non vuoti richiesti")
        _numero(peso, f"{nome}.{etichetta}")
    if not math.isfinite(sum(pesi.values())) or sum(pesi.values()) <= 0:
        raise ValueError(f"{nome}: almeno un peso deve essere positivo")


def date_output_format(valore: date) -> str:
    """Mostra le date in formato italiano; i calcoli rimangono su oggetti date."""
    return valore.strftime("%d/%m/%Y")


def dataset_generation(config, includi_anagrafica=True):
    """Restituisce (clienti, richieste); nessuna etichetta latente viene esportata.

    Una riga cliente riassume richieste completate/pagate nell'ultima finestra.
    Le richieste restituite coprono invece l'intera storia simulata.
    """
    cfg = copy.deepcopy(config)
    config_validator(cfg)
    rng = random.Random(cfg["seed"])
    # Una sequenza distinta isola la dimensione economica da frequenza e date.
    rng_premium = random.Random(f"premium:{cfg['seed']}")
    rng_nomi = rng_anagrafica(cfg["seed"]) if includi_anagrafica else None
    riferimento = date.fromisoformat(cfg["data_riferimento"])
    inizio_finestra = riferimento - timedelta(days=cfg["giorni_osservazione"] - 1)
    clienti, richieste = [], []
    pesi_profili = {k: v["peso"] for k, v in cfg["profili_simulazione"].items()}
    pesi_categorie = {k: v["peso"] for k, v in cfg["categorie"].items()}

    for indice in range(1, cfg["n_clienti"] + 1):
        id_cliente = f"C{indice:06d}"
        profilo = cfg["profili_simulazione"][rng.choices(list(pesi_profili), weights=list(pesi_profili.values()), k=1)[0]]
        stop_min = profilo.get("inattivita_giorni_min", 0)
        stop_max = profilo.get("inattivita_giorni_max", stop_min)
        giorni_stop = rng.randint(stop_min, stop_max)
        if giorni_stop > 0:
            anzianita = rng.randint(giorni_stop + 1, cfg["giorni_attivita_azienda"])
        elif rng.random() < cfg["quota_nuovi_clienti"]:
            anzianita = rng.randint(0, cfg["giorni_cliente_nuovo"] - 1)
        else:
            anzianita = rng.randint(cfg["giorni_cliente_nuovo"], cfg["giorni_attivita_azienda"])
        iscrizione = riferimento - timedelta(days=anzianita)
        fine_attiva = riferimento - timedelta(days=giorni_stop)
        zona = rng.choices(list(cfg["zone"]), weights=list(cfg["zone"].values()), k=1)[0]
        categoria_target = rng.choices(list(pesi_categorie), weights=list(pesi_categorie.values()), k=1)[0]
        rate_giorno = lognorm(rng, profilo["richieste_mese"], cfg["variabilita_frequenza"]) / 30.4375
        ticket_cliente = lognorm(rng, profilo["ticket_medio_euro"], cfg["variabilita_ticket_cliente"])
        if rng_premium.random() < cfg["valore_economico"]["quota_clienti_ticket_premium"]:
            ticket_cliente *= cfg["valore_economico"]["moltiplicatore_ticket_premium"]
        giorni_disponibili = (fine_attiva - iscrizione).days
        # Coorte di clienti paganti: almeno una prima richiesta entro 7 giorni.
        tempo = float(rng.randint(0, min(7, giorni_disponibili)))
        storia = []
        while tempo < giorni_disponibili + 1:
            data_richiesta = iscrizione + timedelta(days=int(tempo))
            alternative = {k: v for k, v in pesi_categorie.items() if k != categoria_target and v > 0}
            categoria = categoria_target
            if alternative and rng.random() > cfg["probabilita_categoria_preferita"]:
                categoria = rng.choices(list(alternative), weights=list(alternative.values()), k=1)[0]
            media = ticket_cliente * cfg["categorie"][categoria]["fattore_prezzo"]
            centesimi = max(1, round(lognorm(rng, media, cfg["variabilita_ticket_richiesta"]) * 100))
            storia.append((data_richiesta, categoria, centesimi))
            richieste.append({
                "id_richiesta": f"R{len(richieste) + 1:09d}", "id_cliente": id_cliente,
                "data_richiesta": date_output_format(data_richiesta), "categoria": categoria,
                "importo_euro": f"{centesimi / 100:.2f}",
            })
            tempo += rng.expovariate(rate_giorno)

        recenti = [t for t in storia if t[0] >= inizio_finestra]
        numero = len(recenti)
        totale_cent = sum(t[2] for t in recenti)
        medio_euro = (Decimal(totale_cent) / (100 * numero)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ) if numero else Decimal("0.00")
        ultima = max(t[0] for t in storia)
        conteggi = Counter(t[1] for t in storia)
        # Categoria preferita = categoria piu' usata nell'intera storia.
        # Parita' risolta alfabeticamente, per non usare etichette nascoste.
        categoria_preferita = min(conteggi, key=lambda k: (-conteggi[k], k))
        cliente = {
            "id_cliente": id_cliente, "zona": zona, "data_iscrizione": date_output_format(iscrizione),
            "numero_richieste": numero, "importo_totale_speso": f"{totale_cent / 100:.2f}",
            "importo_medio_speso": f"{medio_euro:.2f}",
            "giorni_ultima_richiesta": (riferimento - ultima).days,
            "categoria_preferita": categoria_preferita, "data_ultima_richiesta": date_output_format(ultima),
        }
        if rng_nomi is not None:
            # Inseriti soltanto come campi descrittivi: non incidono sul modello.
            cliente.update(genera_anagrafica(rng_nomi, id_cliente))
        clienti.append(cliente)
    return clienti, richieste

def riepilogo_comportamenti(clienti, config):
    """Conta segmenti OSSERVATI, senza leggere i profili latenti del simulatore.

    Le fasce di frequenza sono mutuamente esclusive. Le coorti per anzianità,
    inattività e ticket possono sovrapporsi alle fasce e tra loro.
    """
    riferimento = date.fromisoformat(config["data_riferimento"])
    giorni_periodo = config["giorni_osservazione"]
    soglia_recente = config["giorni_cliente_nuovo"]
    soglie = config["soglie_descrittive"]
    attivi_min = soglie["richieste_min_attivi"]
    molto_attivi_min = soglie["richieste_min_molto_attivi"]
    ticket_alto = Decimal(str(soglie["ticket_medio_elevato_euro"]))
    n = len(clienti)
    fascia_occasionale = f"occasionali_1_{attivi_min - 1}"
    fascia_attivi = f"attivi_{attivi_min}_{molto_attivi_min - 1}"
    fascia_molto_attivi = f"molto_attivi_{molto_attivi_min}_o_piu"
    frequenza = {
        "inattivi_0_richieste_nel_periodo": 0,
        fascia_occasionale: 0,
        fascia_attivi: 0,
        fascia_molto_attivi: 0,
    }
    trasversali = {
        "recenti_iscritti_nel_periodo_definito": 0,
        "ticket_medio_elevato_nel_periodo": 0,
    }
    for cliente in clienti:
        numero = cliente["numero_richieste"]
        if numero == 0:
            frequenza["inattivi_0_richieste_nel_periodo"] += 1
        elif numero < attivi_min:
            frequenza[fascia_occasionale] += 1
        elif numero < molto_attivi_min:
            frequenza[fascia_attivi] += 1
        else:
            frequenza[fascia_molto_attivi] += 1

        anzianita = (riferimento - datetime.strptime(cliente["data_iscrizione"], "%d/%m/%Y").date()).days
        if anzianita < soglia_recente:
            trasversali["recenti_iscritti_nel_periodo_definito"] += 1
        if numero and Decimal(cliente["importo_medio_speso"]) >= ticket_alto:
            trasversali["ticket_medio_elevato_nel_periodo"] += 1

    def quota(conteggi):
        return {
            nome: {"clienti": valore, "percentuale": round(100 * valore / n, 1) if n else 0.0}
            for nome, valore in conteggi.items()
        }

    return {
        "criteri": {
            "finestra_giorni": giorni_periodo,
            "recenti_se_iscritti_da_meno_di_giorni": soglia_recente,
            "inattivi_se_ultima_richiesta_da_almeno_giorni": giorni_periodo,
            "fasce_frequenza_richieste": (
                f"0; 1-{attivi_min - 1}; {attivi_min}-{molto_attivi_min - 1}; almeno {molto_attivi_min}"
            ),
            "ticket_medio_elevato_da_euro": f"{ticket_alto:.2f}",
            "nota": "Inattivi = zero richieste nella finestra (recency almeno pari ai giorni osservati); recenti e ticket elevato sono attributi sovrapponibili. Sono classificazioni descrittive, non profili latenti né cluster K-Means.",
        },
        "fasce_frequenza_esclusive": quota(frequenza),
        "coorti_trasversali": quota(trasversali),
    }

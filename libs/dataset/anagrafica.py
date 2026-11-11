# ==============================================================================
# PROJECT WORK - Corso di Laurea L31
#
# Domenico Caliendo
# ==============================================================================

"""Generazione di anagrafiche inventate per demo di dati sintetici.

Le email usano example.com, dominio riservato per esempi e documentazione.
I dati anagrafici sono indipendenti dal processo casuale delle transazioni.
"""

import random
import re
import unicodedata


NOMI = (
    "Alessandro", "Andrea", "Antonio", "Davide", "Edoardo", "Federico",
    "Filippo", "Francesco", "Gabriele", "Giacomo", "Giovanni", "Giulio",
    "Leonardo", "Lorenzo", "Luca", "Marco", "Matteo", "Michele",
    "Nicola", "Paolo", "Pietro", "Riccardo", "Roberto", "Simone",
    "Alessandra", "Alice", "Anna", "Beatrice", "Camilla", "Chiara",
    "Elena", "Elisa", "Emma", "Federica", "Francesca", "Gaia",
    "Giada", "Giulia", "Greta", "Ilaria", "Laura", "Martina",
    "Matilde", "Sara", "Sofia", "Valentina", "Veronica", "Vittoria",
)

COGNOMI = (
    "Barbieri", "Basile", "Bianchi", "Bruno", "Carbone", "Colombo",
    "Conti", "Costa", "D'Amico", "D'Angelo", "De Luca", "De Santis",
    "Di Pietro", "Esposito", "Ferrari", "Ferraro", "Fiore", "Fontana",
    "Gallo", "Gentile", "Giordano", "Greco", "Leone", "Lombardi",
    "Longo", "Mancini", "Marchetti", "Marino", "Martini", "Moretti",
    "Neri", "Palmieri", "Parisi", "Pellegrini", "Ricci", "Rinaldi",
    "Rizzo", "Romano", "Rossi", "Russo", "Santoro", "Serra",
    "Testa", "Valentini", "Villa", "Vitale",
)

DOMINIO_EMAIL_FITTIZIO = "example.com"


def _slug_email(testo: str) -> str:
    """Normalizza nomi composti e apostrofi per un indirizzo email leggibile."""
    testo = unicodedata.normalize("NFKD", testo)
    testo = testo.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", ".", testo).strip(".")


def genera_anagrafica(rng: random.Random, id_cliente: str) -> dict[str, str]:
    """Crea nome, cognome ed email fittizi; ID incluso per univocità."""
    nome = rng.choice(NOMI)
    cognome = rng.choice(COGNOMI)
    username = f"{_slug_email(nome)}.{_slug_email(cognome)}.{id_cliente.lower()}"
    return {
        "nome": nome,
        "cognome": cognome,
        "email": f"{username}@{DOMINIO_EMAIL_FITTIZIO}",
    }


def rng_anagrafica(seed: int) -> random.Random:
    """Seed separato: aggiungere l'anagrafica non altera le metriche dei clienti."""
    return random.Random(f"anagrafica:{seed}")

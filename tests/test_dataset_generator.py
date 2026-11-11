"""Test di consistenza dati sintetici, eseguibili con unittest standard library."""

import csv
import unittest
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from libs.dataset.dataset_tool import (
    config_validator,
    dataset_generation,
    load_dataset_configuration,
    riepilogo_comportamenti,
)

ROOT = Path(__file__).resolve().parent.parent


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.cfg = load_dataset_configuration(ROOT / "config" / "dataset_generator_config.ini")
        config_validator(cls.cfg)
        cls.clienti, cls.richieste = dataset_generation(cls.cfg, True)

    def test_modello_coerente_con_traccia(self):
        self.assertEqual(
            set(self.cfg["profili_simulazione"]),
            {"molto_attivi", "abituali", "occasionali", "inattivi"},
        )
        d = riepilogo_comportamenti(self.clienti, self.cfg)
        self.assertEqual(set(d["coorti_trasversali"]), {
            "recenti_iscritti_nel_periodo_definito", "ticket_medio_elevato_nel_periodo"
        })
        self.assertEqual(
            d["fasce_frequenza_esclusive"]["inattivi_0_richieste_nel_periodo"]["clienti"],
            sum(c["giorni_ultima_richiesta"] >= self.cfg["giorni_osservazione"] for c in self.clienti),
        )
        self.assertGreater(d["coorti_trasversali"]["ticket_medio_elevato_nel_periodo"]["clienti"], 20)
        # Il ticket alto e' trasversale: non identifica una sola fascia di frequenza.
        soglia = Decimal(str(self.cfg["soglie_descrittive"]["ticket_medio_elevato_euro"]))
        premium = [c for c in self.clienti if int(c["numero_richieste"]) > 0
                   and Decimal(c["importo_medio_speso"]) >= soglia]
        self.assertTrue(any(1 <= c["numero_richieste"] <= 2 for c in premium))
        self.assertTrue(any(c["numero_richieste"] >= 3 for c in premium))

    def test_dimensione_e_identificativi(self):
        self.assertEqual(len(self.clienti), self.cfg["n_clienti"])
        ids = [c["id_cliente"] for c in self.clienti]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(len(set(c["email"] for c in self.clienti)), len(ids))
        self.assertTrue(all(c["email"].endswith("@example.com") for c in self.clienti))

    def test_riproducibilita_e_anagrafica(self):
        c1, t1 = dataset_generation(self.cfg, True)
        self.assertEqual(c1, self.clienti)
        self.assertEqual(t1, self.richieste)
        c2, t2 = dataset_generation(self.cfg, False)
        self.assertEqual(t2, self.richieste)
        for c_old, c_new in zip(self.clienti, c2):
            self.assertEqual({k: v for k, v in c_old.items() if k not in {"nome", "cognome", "email"}}, c_new)

    def test_aggregazioni_temporali_e_importi(self):
        fine = date.fromisoformat(self.cfg["data_riferimento"])
        inizio = fine - timedelta(days=self.cfg["giorni_osservazione"] - 1)
        storico = defaultdict(list)
        ids_richieste = []
        for t in self.richieste:
            data_richiesta = datetime.strptime(t["data_richiesta"], "%d/%m/%Y").date()
            self.assertLessEqual(data_richiesta, fine)
            self.assertGreater(Decimal(t["importo_euro"]), 0)
            storico[t["id_cliente"]].append((data_richiesta, t["categoria"], Decimal(t["importo_euro"])))
            ids_richieste.append(t["id_richiesta"])
        self.assertEqual(len(ids_richieste), len(set(ids_richieste)))
        for c in self.clienti:
            dati = storico[c["id_cliente"]]
            self.assertTrue(dati)
            iscrizione = datetime.strptime(c["data_iscrizione"], "%d/%m/%Y").date()
            ultima = max(r[0] for r in dati)
            self.assertGreaterEqual(min(r[0] for r in dati), iscrizione)
            self.assertEqual(c["data_ultima_richiesta"], ultima.strftime("%d/%m/%Y"))
            self.assertEqual(c["giorni_ultima_richiesta"], (fine - ultima).days)
            subset = [r for r in dati if r[0] >= inizio]
            n = len(subset)
            totale = sum((r[2] for r in subset), Decimal("0.00"))
            media = (totale / n).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) if n else Decimal("0.00")
            self.assertEqual(c["numero_richieste"], n)
            self.assertEqual(Decimal(c["importo_totale_speso"]), totale)
            self.assertEqual(Decimal(c["importo_medio_speso"]), media)
            conteggi = Counter(r[1] for r in dati)
            categoria = min(conteggi, key=lambda k: (-conteggi[k], k))
            self.assertEqual(c["categoria_preferita"], categoria)

    def test_fasce_osservate(self):
        d = riepilogo_comportamenti(self.clienti, self.cfg)
        fasce = d["fasce_frequenza_esclusive"]
        self.assertEqual(sum(x["clienti"] for x in fasce.values()), len(self.clienti))
        # Campione di riferimento: controlla che emergano, senza imporre
        # alcuna quota per singolo cliente, tutte le fasce previste.
        self.assertGreater(fasce["molto_attivi_6_o_piu"]["clienti"], 30)
        self.assertGreater(fasce["attivi_3_5"]["clienti"], 30)
        self.assertGreater(fasce["occasionali_1_2"]["clienti"], 100)
        self.assertGreater(fasce["inattivi_0_richieste_nel_periodo"]["clienti"], 50)
        self.assertGreater(d["coorti_trasversali"]["recenti_iscritti_nel_periodo_definito"]["clienti"], 30)
        self.assertGreater(d["coorti_trasversali"]["ticket_medio_elevato_nel_periodo"]["clienti"], 20)
        # Il ticket alto e' trasversale: non identifica una sola fascia di frequenza.
        soglia = Decimal(str(self.cfg["soglie_descrittive"]["ticket_medio_elevato_euro"]))
        premium = [c for c in self.clienti if int(c["numero_richieste"]) > 0
                   and Decimal(c["importo_medio_speso"]) >= soglia]
        self.assertTrue(any(1 <= c["numero_richieste"] <= 2 for c in premium))
        self.assertTrue(any(c["numero_richieste"] >= 3 for c in premium))

    def test_riepilogo_json_aderente_traccia(self):
        import json
        import tempfile
        from libs.utils.helpers import store_dataset_recap

        comportamenti = riepilogo_comportamenti(self.clienti, self.cfg)
        riferimento = date.fromisoformat(self.cfg["data_riferimento"])
        metadati = {
            "timestamp_generazione_utc": "2026-11-09T17:30:00.000000Z",
            "inizio_osservazione_incluso": (
                riferimento - timedelta(days=self.cfg["giorni_osservazione"] - 1)
            ).strftime("%d/%m/%Y"),
            "fine_osservazione_inclusa": riferimento.strftime("%d/%m/%Y"),
            "distribuzione_comportamenti_osservati": comportamenti,
        }
        originali = [dict(c) for c in self.clienti]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "clienti.csv"
            recap_path = store_dataset_recap(self.clienti, self.richieste, metadati, output)
            self.assertEqual(recap_path, output.with_suffix(".recap.json"))
            recap = json.loads(recap_path.read_text(encoding="utf-8"))
            richiesti = recap["comportamenti_richiesti_dalla_traccia"]
            self.assertEqual(set(richiesti), {"molto_attivi", "occasionali", "recenti", "inattivi"})
            self.assertEqual(richiesti["recenti"]["clienti"],
                             comportamenti["coorti_trasversali"]["recenti_iscritti_nel_periodo_definito"]["clienti"])
            fasce = comportamenti["fasce_frequenza_esclusive"]
            self.assertEqual(richiesti["molto_attivi"]["clienti"], fasce["molto_attivi_6_o_piu"]["clienti"])
            self.assertEqual(richiesti["inattivi"]["clienti"], fasce["inattivi_0_richieste_nel_periodo"]["clienti"])
            self.assertEqual(richiesti["occasionali"]["clienti"], fasce["occasionali_1_2"]["clienti"])
            self.assertEqual(recap["analisi_aggiuntive"]["attivi_intermedi"]["clienti"], fasce["attivi_3_5"]["clienti"])
            self.assertEqual(
                richiesti["inattivi"]["clienti"] + richiesti["occasionali"]["clienti"] +
                richiesti["molto_attivi"]["clienti"] + recap["analisi_aggiuntive"]["attivi_intermedi"]["clienti"],
                len(self.clienti),
            )
            with self.assertRaises(FileExistsError):
                store_dataset_recap(self.clienti, self.richieste, metadati, output)
        self.assertEqual(self.clienti, originali)

    def test_soglie_analitiche_non_cambiano_csv(self):
        import copy
        altro = copy.deepcopy(self.cfg)
        altro["soglie_descrittive"]["richieste_min_attivi"] = 2
        altro["soglie_descrittive"]["richieste_min_molto_attivi"] = 5
        altro["soglie_descrittive"]["ticket_medio_elevato_euro"] = 50.0
        config_validator(altro)
        self.assertEqual(dataset_generation(altro, True), (self.clienti, self.richieste))
        altre_fasce = riepilogo_comportamenti(self.clienti, altro)["fasce_frequenza_esclusive"]
        self.assertIn("attivi_2_4", altre_fasce)
        self.assertIn("molto_attivi_5_o_piu", altre_fasce)


if __name__ == "__main__":
    unittest.main()

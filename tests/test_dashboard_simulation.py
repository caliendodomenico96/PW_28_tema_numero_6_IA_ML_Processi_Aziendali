"""Test del registro multi-cliente (senza Streamlit e senza refit)."""
import unittest
from pathlib import Path

import pandas as pd

from libs.dashboard.data import completed_training_runs, load_cluster_results
from libs.dashboard.inference import load_prediction_context
from libs.dashboard.simulation import (
    add_customers, cluster_scenario_summary, delete_customer, edit_customer,
    empty_simulation, predict_simulation,
)
from libs.dashboard.visualization import plot_customer_in_clusters

ROOT = Path(__file__).resolve().parents[1]


def sample(first="Marco", requests=2, spending=100.0, recency=14, **extra):
    result = {
        "nome": first, "cognome": "Esempio", "email": "cliente@example.com",
        "zona": "Roma", "categoria_preferita": "Configurazione backup",
        "data_iscrizione": "01/06/2026", "numero_richieste": requests,
        "importo_totale_speso": spending, "giorni_ultima_richiesta": recency,
    }
    return dict(result, **extra)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = completed_training_runs(ROOT)[0]
        cls.ctx3 = load_prediction_context(cls.training, 3)
        cls.ctx4 = load_prediction_context(cls.training, 4)
        cls.historical, _ = load_cluster_results(cls.training, 4)

    def test_ids_stable_after_delete_append_and_edit(self):
        initial = empty_simulation()
        first, ids = add_customers(initial, pd.DataFrame([sample("Marco"), sample("Anna")]), self.ctx4)
        self.assertEqual(ids, ["NEW_CUSTOMER_001", "NEW_CUSTOMER_002"])
        self.assertEqual(initial, empty_simulation())  # Nessuna mutazione in-place.
        second = delete_customer(first, "NEW_CUSTOMER_001")
        self.assertEqual(first["records"][0]["id_cliente"], "NEW_CUSTOMER_001")
        third, ids2 = add_customers(second, pd.DataFrame([sample("Lucia")]), self.ctx4)
        self.assertEqual(ids2, ["NEW_CUSTOMER_003"])
        self.assertEqual(third["next_sequence"], 4)
        self.assertEqual(set(predict_simulation(third, self.ctx4).id_cliente),
                         {"NEW_CUSTOMER_002", "NEW_CUSTOMER_003"})
        fourth = edit_customer(third, "NEW_CUSTOMER_002", pd.DataFrame([
            sample("Anna", requests=8, spending=800, recency=3)
        ]), self.ctx4)
        row = predict_simulation(fourth, self.ctx4).set_index("id_cliente")
        self.assertEqual(row.loc["NEW_CUSTOMER_002", "numero_richieste"], 8)
        self.assertEqual(row.loc["NEW_CUSTOMER_002", "nome"], "Anna")
        self.assertEqual(third["records"][0]["payload"]["numero_richieste"], 2)

    def test_csv_batch_ids_and_origin(self):
        csv = pd.DataFrame([dict(sample("Mario"), id_cliente="E001"),
                            dict(sample("Paola"), id_cliente="E002")])
        state, ids = add_customers(empty_simulation(), csv, self.ctx4)
        result = predict_simulation(state, self.ctx4)
        self.assertEqual(result.id_cliente.tolist(), ids)
        self.assertEqual(result.id_cliente_origine.tolist(), ["E001", "E002"])

    def test_atomic_failure_does_not_add_partial_batch(self):
        initial, _ = add_customers(empty_simulation(), pd.DataFrame([sample()]), self.ctx4)
        invalid = pd.DataFrame([sample("OK"), sample("Bad", requests=0, spending=12, recency=100)])
        with self.assertRaises(ValueError):
            add_customers(initial, invalid, self.ctx4)
        self.assertEqual(initial["next_sequence"], 2)
        self.assertEqual(len(initial["records"]), 1)
        with self.assertRaises(ValueError):
            edit_customer(initial, "NEW_CUSTOMER_001", invalid.iloc[1:], self.ctx4)
        self.assertEqual(initial["records"][0]["payload"]["numero_richieste"], 2)

    def test_k_switch_recomputes_predictions_without_refit(self):
        initial, _ = add_customers(empty_simulation(), pd.DataFrame([sample()]), self.ctx4)
        labels3 = predict_simulation(initial, self.ctx3)
        labels4 = predict_simulation(initial, self.ctx4)
        self.assertEqual(labels3.loc[0, "id_cliente"], labels4.loc[0, "id_cliente"])
        self.assertTrue(0 <= labels3.loc[0, "cluster"] < 3)
        self.assertTrue(0 <= labels4.loc[0, "cluster"] < 4)
        self.assertEqual(len(initial["records"]), 1)

    def test_scenario_counts_and_graphs_preserve_history(self):
        state, ids = add_customers(empty_simulation(), pd.DataFrame([
            sample("Marco"), sample("Anna", requests=8, spending=780, recency=2),
            sample("Enzo", requests=0, spending=0, recency=115),
        ]), self.ctx4)
        simulated = predict_simulation(state, self.ctx4)
        hist_copy = self.historical.copy(deep=True)
        table = cluster_scenario_summary(self.historical, simulated, 4)
        self.assertEqual(table.clienti_storici.sum(), 600)
        self.assertEqual(table.clienti_simulati.sum(), 3)
        self.assertEqual(table.totale_scenario.sum(), 603)
        for mode in ("Richieste e spesa", "PCA 2D"):
            fig = plot_customer_in_clusters(self.historical, simulated, self.ctx4, ids[1], mode, show_all=True)
            self.assertEqual(fig.data[-1].marker.symbol, "star")
            self.assertEqual(len(fig.data[-2].x), 2)
            self.assertEqual(sum(len(trace.x) for trace in fig.data if trace.name.startswith("Cluster ")), 600)
        pd.testing.assert_frame_equal(self.historical, hist_copy)

    def test_delete_nonexistent_customer(self):
        with self.assertRaisesRegex(ValueError, "inesistente"):
            delete_customer(empty_simulation(), "NEW_CUSTOMER_999")


if __name__ == "__main__":
    unittest.main()

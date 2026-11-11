"""Test della classificazione di nuovi clienti senza Streamlit o retraining."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from libs.dashboard.data import completed_training_runs, load_cluster_results
from libs.dashboard.visualization import plot_customer_in_clusters
from libs.dashboard.inference import (
    load_prediction_context, predict_new_customers,
    read_uploaded_csv, validate_new_customers,
)

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_dir = completed_training_runs(ROOT)[0]
        cls.context3 = load_prediction_context(cls.training_dir, 3)
        cls.context4 = load_prediction_context(cls.training_dir, 4)

    def test_reproduces_labels_on_existing_clients(self):
        # Il preprocessor e il modello devono restituire le stesse etichette
        # già assegnate durante il training, senza usare 'cluster' come feature.
        for k, context in [(3, self.context3), (4, self.context4)]:
            source, _ = load_cluster_results(self.training_dir, k)
            sample = source.iloc[:35].copy()
            sample['data_iscrizione'] = sample['data_iscrizione'].dt.strftime('%d/%m/%Y')
            result = predict_new_customers(sample, context)
            np.testing.assert_array_equal(result['cluster'].to_numpy(), sample['cluster'].to_numpy())
            self.assertTrue(np.isfinite(result['distanza_centroide']).all())
            self.assertTrue((result['distanza_centroide'] >= 0).all())

    def test_upload_csv_with_semicolon_and_italian_decimal(self):
        contents = (
            'id_cliente;data_iscrizione;numero_richieste;importo_totale_speso;giorni_ultima_richiesta\n'
            'E01;01/06/2026;2;100,50;14\n'
            'E02;01/03/2026;0;0,00;130\n'
        ).encode('utf-8-sig')
        imported = read_uploaded_csv(contents)
        result = predict_new_customers(imported, self.context4)
        self.assertEqual(len(result), 2)
        self.assertEqual(float(result.loc[0, 'importo_totale_speso']), 100.50)
        self.assertEqual(set(result['id_cliente']), {'NEW_CUSTOMER_001', 'NEW_CUSTOMER_002'})
        self.assertEqual(list(result['id_cliente_origine']), ['E01', 'E02'])
        self.assertTrue(result['cluster'].between(0, 3).all())

    def test_rejects_inconsistent_data(self):
        good = pd.DataFrame([{
            'data_iscrizione': '01/06/2026', 'numero_richieste': 2,
            'importo_totale_speso': 90.0, 'giorni_ultima_richiesta': 15,
        }])
        cases = [
            ({'data_iscrizione': '31/12/2026'}, 'data_iscrizione'),
            ({'numero_richieste': 0, 'importo_totale_speso': 90}, 'spesa'),
            ({'numero_richieste': 0, 'importo_totale_speso': 0, 'giorni_ultima_richiesta': 10}, '0 richieste'),
            ({'numero_richieste': 2, 'giorni_ultima_richiesta': 120}, 'ultima richiesta fuori'),
            ({'numero_richieste': -1}, 'interi non negativi'),
            ({'numero_richieste': 1.25}, 'interi non negativi'),
        ]
        for change, msg in cases:
            with self.subTest(change=change):
                invalid = good.assign(**change)
                with self.assertRaisesRegex(ValueError, msg):
                    validate_new_customers(invalid, self.context4.reference, 90)

    def test_complete_customer_fields_and_derived_columns(self):
        raw = pd.DataFrame([{
            'nome': 'Elena', 'cognome': 'Bianchi', 'email': 'elena@example.com',
            'zona': 'Roma', 'categoria_preferita': 'Configurazione backup',
            'data_iscrizione': '01/06/2026', 'numero_richieste': 2,
            'importo_totale_speso': 100, 'giorni_ultima_richiesta': 14,
        }])
        result = predict_new_customers(raw, self.context4)
        self.assertEqual(result.loc[0, 'id_cliente'], 'NEW_CUSTOMER_001')
        self.assertEqual(result.loc[0, 'zona'], 'Roma')
        self.assertEqual(result.loc[0, 'categoria_preferita'], 'Configurazione backup')
        self.assertEqual(result.loc[0, 'nome'], 'Elena')
        self.assertEqual(result.loc[0, 'data_ultima_richiesta'], '12/10/2026')
        self.assertEqual(result.loc[0, 'importo_medio_speso'], 50.0)

    def test_rejects_inconsistent_derived_fields(self):
        base = pd.DataFrame([{
            'data_iscrizione': '01/06/2026', 'numero_richieste': 2,
            'importo_totale_speso': 100.0, 'giorni_ultima_richiesta': 14,
            'data_ultima_richiesta': '12/10/2026', 'importo_medio_speso': 50.0,
        }])
        predict_new_customers(base, self.context4)
        with self.assertRaisesRegex(ValueError, 'data_ultima_richiesta'):
            predict_new_customers(base.assign(data_ultima_richiesta='01/10/2026'), self.context4)
        with self.assertRaisesRegex(ValueError, 'importo_medio_speso'):
            predict_new_customers(base.assign(importo_medio_speso=98), self.context4)

    def test_interactive_chart_contains_originals_and_new_customer(self):
        historical, _ = load_cluster_results(self.training_dir, 4)
        raw = pd.DataFrame([{
            'data_iscrizione': '01/06/2026', 'numero_richieste': 2,
            'importo_totale_speso': 100.0, 'giorni_ultima_richiesta': 14,
            'zona': 'Roma', 'categoria_preferita': 'Configurazione backup',
        }])
        predicted = predict_new_customers(raw, self.context4)
        historic_before = historical.copy(deep=True)
        for projection in ('Richieste e spesa', 'PCA 2D'):
            chart = plot_customer_in_clusters(historical, predicted, self.context4, 'NEW_CUSTOMER_001', projection)
            self.assertEqual(chart.data[-1].marker.symbol, 'star')
            self.assertIn('NEW_CUSTOMER_001', chart.data[-1].name)
            self.assertEqual(sum(len(trace.x) for trace in chart.data if trace.name.startswith('Cluster ')), len(historical))
        pd.testing.assert_frame_equal(historical, historic_before)

    def test_does_not_modify_existing_artifacts(self):
        files = [self.training_dir / 'models' / 'preprocessor.joblib',
                 self.training_dir / 'models' / 'kmeans_k4.joblib',
                 self.training_dir / 'cluster_k4.csv']
        before = [p.read_bytes() for p in files]
        raw = pd.DataFrame([{
            'data_iscrizione': '01/06/2026', 'numero_richieste': 2,
            'importo_totale_speso': 100, 'giorni_ultima_richiesta': 15,
        }])
        result = predict_new_customers(raw, self.context4)
        self.assertEqual(result.loc[0, 'id_cliente'], 'NEW_CUSTOMER_001')
        self.assertEqual(before, [p.read_bytes() for p in files])


if __name__ == '__main__':
    unittest.main()

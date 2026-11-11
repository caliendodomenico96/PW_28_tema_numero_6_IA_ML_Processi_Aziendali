"""Test automatici del secondo punto del Project Work (libreria standard unittest)."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from libs.clustering.configuration import load_kmeans_config
from libs.clustering.preprocessing import build_feature_frame, create_preprocessor, load_customer_data
from libs.clustering.storage import json_read, resolve_dataset
from libs.clustering.training import train_kmeans
from libs.clustering.validation import validate_kmeans

ROOT = Path(__file__).resolve().parents[1]


class TestKMeansPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_kmeans_config(ROOT / "config" / "kmeans_config.ini", ROOT)
        cls.dataset = resolve_dataset(ROOT, "latest")
        cls.df, cls.metadata, cls.reference = load_customer_data(cls.dataset)

    def test_data_columns_and_reference(self):
        self.assertEqual(len(self.df), 600)
        self.assertEqual(str(self.reference.date()), "2026-10-26")
        self.assertEqual(self.df.id_cliente.nunique(), len(self.df))

    def test_preprocessing_scaling(self):
        data, report = build_feature_frame(self.df, self.reference, self.config.features, True)
        self.assertEqual(list(data.columns), list(self.config.features))
        transformed = create_preprocessor().fit_transform(data)
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_allclose(transformed.mean(axis=0), np.zeros(4), atol=1e-10)
        np.testing.assert_allclose(transformed.std(axis=0), np.ones(4), atol=1e-10)
        self.assertEqual(sum(report["valori_mancanti_per_feature"].values()), 0)

    def test_missing_and_negative_are_imputed(self):
        sample = self.df.copy()
        sample.loc[0, "numero_richieste"] = None
        sample.loc[1, "importo_totale_speso"] = -20
        sample.loc[2, "data_iscrizione"] = "data-non-valida"
        sample.loc[3, "giorni_ultima_richiesta"] = -1
        data, report = build_feature_frame(sample, self.reference, self.config.features, True)
        self.assertEqual(report["valori_mancanti_per_feature"]["numero_richieste"], 1)
        self.assertEqual(report["valori_invalidi_o_negativi_per_feature"]["importo_totale_speso"], 1)
        self.assertEqual(report["valori_mancanti_per_feature"]["giorni_da_iscrizione"], 1)
        transformed = create_preprocessor().fit_transform(data)
        self.assertTrue(np.isfinite(transformed).all())

    def test_too_many_missing_values_rejected(self):
        sample = self.df.copy()
        sample.loc[:200, "numero_richieste"] = None
        with self.assertRaisesRegex(ValueError, "25%"):
            build_feature_frame(sample, self.reference, self.config.features, True)

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp)
            path = sub / "customer_history_dataset.csv"
            self.df.head(20).assign(id_cliente="DUPLICATO").to_csv(path, index=False)
            path.with_suffix(".metadati.json").write_text(json.dumps({
                "fine_osservazione_inclusa": "26/10/2026", "separatore_csv": ","
            }))
            with self.assertRaisesRegex(ValueError, "duplicato"):
                load_customer_data(path)

    def test_training_validation_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = replace(self.config, dataset_csv=str(self.dataset),
                          output_dir=Path(tmp) / "kmeans", k_max=4, k_confronto=(3, 4),
                          training_run="latest")
            train_path = train_kmeans(ROOT, cfg)
            report = json_read(train_path / "training_report.json")
            self.assertEqual(report["k_confronto"], [3, 4])
            self.assertTrue((train_path / "grafico_gomito.png").exists())
            self.assertTrue((train_path / "models" / "kmeans_k4.joblib").exists())
            self.assertEqual(len(pd.read_csv(train_path / "cluster_k4.csv")), len(self.df))
            self.assertTrue((train_path / "heatmap_centroidi_k4.png").exists())
            self.assertTrue((train_path / "scatter_pca_k4.png").exists())
            self.assertTrue((train_path / "boxplot_feature_k4.png").exists())
            validation_path = validate_kmeans(ROOT, cfg)
            validation_report = json_read(validation_path / "validation_report.json")
            self.assertTrue(validation_report["assegnazioni_modelli_verificate"])
            self.assertEqual(len(validation_report["metriche"]), 2)
            self.assertTrue((validation_path / "silhouette_k3.png").exists())
            self.assertTrue((validation_path / "silhouette_k4.png").exists())
            # I grafici descrittivi e i profili devono esistere solo nel training.
            for k in (3, 4):
                self.assertFalse((validation_path / f"heatmap_centroidi_k{k}.png").exists())
                self.assertFalse((validation_path / f"scatter_pca_k{k}.png").exists())
                self.assertFalse((validation_path / f"boxplot_feature_k{k}.png").exists())
                self.assertFalse((validation_path / f"profili_cluster_k{k}.json").exists())
            self.assertEqual(validation_report["grafici_validation"],
                             ["silhouette_k3.png", "silhouette_k4.png"])
            self.assertTrue((validation_path / "confronto_k.csv").exists())

    def test_features_with_personal_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kmeans_config.ini"
            text = (ROOT / "config" / "kmeans_config.ini").read_text()
            text = text.replace("numero_richieste, importo_totale_speso, giorni_ultima_richiesta, giorni_da_iscrizione",
                                "nome, numero_richieste")
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "Feature non ammesse"):
                load_kmeans_config(path, ROOT)


if __name__ == "__main__":
    unittest.main()

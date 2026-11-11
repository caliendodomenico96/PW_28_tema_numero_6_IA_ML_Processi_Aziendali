"""Test di integrazione read-only dei risultati di validation."""

import json
import tempfile
import unittest
from pathlib import Path

from libs.dashboard.data import dashboard_catalog_snapshot
from libs.dashboard.validation_reports import (
    completed_validation_runs, latest_validation_for_k,
    read_validation_report, validation_metrics_rows, validation_catalog_signature,
)


class DashboardValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "results" / "kmeans" / "run_a"
        self.train.mkdir(parents=True)
        (self.train / "training_report.json").write_text(
            json.dumps({"dataset_csv": "data/missing.csv", "k_modelli_salvati": [4]}), encoding="utf-8")
        self.validations = self.train / "validations"
        self.validations.mkdir()

    def create_validation(self, name, ks=(3, 4), complete=True):
        folder = self.validations / name
        folder.mkdir()
        metrics = [dict(k=k, silhouette=0.5, davies_bouldin=0.8,
                        calinski_harabasz=600., cluster_minimo=10) for k in ks]
        for k in ks:
            (folder / f"silhouette_k{k}.png").write_bytes(b"PNG")
        if complete:
            (folder / "validation_report.json").write_text(
                json.dumps({"metriche": metrics}), encoding="utf-8")
        return folder

    def test_ignores_incomplete_validations(self):
        self.create_validation("2026-11-10", complete=False)
        self.assertEqual(completed_validation_runs(self.train), [])
        self.assertIsNone(latest_validation_for_k(self.train, 4))

    def test_selects_latest_valid_for_selected_k(self):
        older = self.create_validation("2026-11-09", ks=(3, 4))
        self.create_validation("2026-11-10", ks=(3,))
        chosen = latest_validation_for_k(self.train, 4)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen[0], older)
        self.assertEqual(len(validation_metrics_rows(chosen[1])), 2)
        self.assertEqual(latest_validation_for_k(self.train, 2), None)

    def test_deletion_and_addition_affect_catalog_signature(self):
        before = dashboard_catalog_snapshot(self.root)
        folder = self.create_validation("2026-11-09")
        changed = dashboard_catalog_snapshot(self.root)
        self.assertNotEqual(before, changed)
        (folder / "silhouette_k4.png").unlink()
        self.assertNotEqual(changed, dashboard_catalog_snapshot(self.root))
        self.assertIsNone(latest_validation_for_k(self.train, 4))
        self.assertIsNotNone(latest_validation_for_k(self.train, 3))

    def test_report_content(self):
        folder = self.create_validation("2026-11-09", ks=(4,))
        report = read_validation_report(folder)
        self.assertEqual(validation_metrics_rows(report)[0]["K"], 4)


if __name__ == "__main__":
    unittest.main()

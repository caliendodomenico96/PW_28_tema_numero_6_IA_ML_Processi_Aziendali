"""Test del layer read-only del frontend, senza dipendenze Streamlit."""
import unittest
import tempfile
import json
from pathlib import Path
from libs.dashboard.data import (
    completed_training_runs, read_training_report, load_cluster_results,
    filter_clients, cluster_summary, available_dataset_folders,
    training_run_status, usable_training_runs, dashboard_catalog_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


class DashboardDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_dir = completed_training_runs(ROOT)[0]
        cls.report = read_training_report(cls.training_dir)

    def test_read_results_and_summary(self):
        for k in self.report["k_modelli_salvati"]:
            df, profiles = load_cluster_results(self.training_dir, int(k))
            self.assertEqual(len(df), self.report["n_clienti"])
            self.assertEqual(df.cluster.nunique(), k)
            self.assertEqual(int(cluster_summary(df).clienti.sum()), len(df))
            self.assertEqual(len(profiles), k)

    def test_filters_do_not_change_clusters(self):
        k = int(self.report["k_modelli_salvati"][0])
        df, _ = load_cluster_results(self.training_dir, k)
        zone = df.zona.iloc[0]
        category = df.categoria_preferita.iloc[0]
        filtered = filter_clients(df, zones=[zone], categories=[category])
        self.assertTrue((filtered.zona == zone).all())
        self.assertTrue((filtered.categoria_preferita == category).all())
        self.assertTrue(filtered.cluster.equals(df.loc[filtered.index, "cluster"]))

    def test_new_training_run_discovered_without_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs_dir = root / "results" / "kmeans"
            runs_dir.mkdir(parents=True)
            old = runs_dir / "2026-11-09T12-00-00Z"
            new = runs_dir / "2026-11-09T12-01-00Z"
            old.mkdir()
            new.mkdir()
            (old / "training_report.json").write_text("{}", encoding="utf-8")
            self.assertEqual([p.name for p in completed_training_runs(root)], [old.name])
            # La nuova directory da sola non è considerata un training completo.
            self.assertEqual([p.name for p in completed_training_runs(root)], [old.name])
            (new / "training_report.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                [p.name for p in completed_training_runs(root)],
                [new.name, old.name],
            )

    def test_refresh_catalog_on_added_and_deleted_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / 'results' / 'kmeans' / '2026-11-09T12-00-00Z'
            data_dir = root / 'data' / '2026-11-09T11-00-00Z'
            run_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            csv = data_dir / 'customer_history_dataset.csv'
            meta = data_dir / 'customer_history_dataset.metadati.json'
            csv.write_text('id_cliente\nX\n', encoding='utf-8')
            meta.write_text('{}', encoding='utf-8')
            (run_dir / 'training_report.json').write_text(json.dumps({
                'dataset_csv': str(csv.relative_to(root)), 'k_modelli_salvati': [4]
            }), encoding='utf-8')
            (run_dir / 'models').mkdir()
            for name in ('models/preprocessor.joblib', 'models/kmeans_k4.joblib',
                         'cluster_k4.csv', 'profili_cluster_k4.json'):
                (run_dir / name).write_text('placeholder', encoding='utf-8')
            self.assertTrue(training_run_status(root, run_dir)[0])
            self.assertEqual(usable_training_runs(root), [run_dir])
            before = dashboard_catalog_snapshot(root)
            extra = root / 'data' / '2026-11-09T11-10-00Z'
            extra.mkdir()
            (extra / csv.name).write_text('id_cliente\nA\n', encoding='utf-8')
            (extra / meta.name).write_text('{}', encoding='utf-8')
            self.assertNotEqual(dashboard_catalog_snapshot(root), before)
            self.assertEqual(len(available_dataset_folders(root)), 2)
            extra.rename(root / 'data' / '2026-11-09T11-15-00Z')
            self.assertNotEqual(dashboard_catalog_snapshot(root), before)
            csv.unlink()
            self.assertFalse(training_run_status(root, run_dir)[0])
            self.assertEqual(usable_training_runs(root), [])
            self.assertNotEqual(dashboard_catalog_snapshot(root), before)

    def test_period_filter(self):
        k = int(self.report["k_modelli_salvati"][0])
        df, _ = load_cluster_results(self.training_dir, k)
        earliest = df.data_iscrizione.min().date()
        subset = filter_clients(df, date_field="data_iscrizione", start_date=earliest, end_date=earliest)
        self.assertGreater(len(subset), 0)
        self.assertTrue((subset.data_iscrizione.dt.date == earliest).all())


if __name__ == "__main__":
    unittest.main()

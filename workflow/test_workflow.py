import json
from pathlib import Path
import sqlite3
import unittest
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from workflow.calibration import evaluate
from workflow.intake_api import build_app

ROOT = Path(__file__).resolve().parents[1]


class InventoryWorkflowTest(unittest.TestCase):
    def test_import_report_retry_retrieval_and_database(self):
        with TemporaryDirectory() as directory:
            db = Path(directory) / "run.sqlite3"
            with TestClient(build_app(db)) as client:
                for extension in ("json", "csv"):
                    inventory = {
                        "format": extension,
                        "text": (ROOT / f"dist/docs/scenario.{extension}").read_text(encoding="utf-8"),
                    }
                    initial = client.post("/reports", json=inventory)
                    self.assertEqual(initial.status_code, 200)
                    self.assertTrue(initial.json()["created"])
                    report = initial.json()["report"]
                    self.assertEqual(report["planned"]["kg"], 121)
                    self.assertEqual(report["nearest_first"]["kg"], 86)
                    self.assertFalse(client.post("/reports", json=inventory).json()["created"])
                    self.assertEqual(client.get("/reports/" + report["report_id"]).json(), report)
                rejected = client.post("/reports", json={"format":"json","text":'{"lots":[],"orders":[]}'})
                self.assertEqual(rejected.status_code, 422)
                self.assertEqual(client.get("/reports/unknown").status_code, 404)
            with sqlite3.connect(db) as connection:
                rows = connection.execute("SELECT report_json FROM reports").fetchall()
            self.assertEqual(len(rows), 2)
            self.assertTrue(all("text" not in json.loads(row[0]) for row in rows))

    def test_local_calibrator_reduces_brier_on_synthetic_shift(self):
        result = evaluate()
        self.assertEqual(result["data"], "synthetic")
        self.assertEqual(result["target_holdout_rows"], 500)
        self.assertLess(result["adapted"]["brier"], result["baseline"]["brier"])


if __name__ == "__main__":
    unittest.main()

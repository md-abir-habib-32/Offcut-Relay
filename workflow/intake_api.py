"""Local inventory validation and reproducible allocation reports."""
from argparse import ArgumentParser
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

NODE_SCRIPT = Path(__file__).with_name("scenario-analysis.cjs")
INPUT_LIMIT = 32_000
ALGORITHM_REVISION = "browser-planner-v1"


class Inventory(BaseModel):
    format: Literal["csv", "json"]
    text: str = Field(min_length=1, max_length=INPUT_LIMIT)
    priority: float = Field(default=0.2, ge=0, le=1)


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=10)
    db.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id TEXT PRIMARY KEY,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return db


def find_report(path: Path, report_id: str):
    with closing(connect(path)) as db:
        row = db.execute("SELECT report_json FROM reports WHERE report_id = ?", (report_id,)).fetchone()
    return json.loads(row[0]) if row else None


def run_report(path: Path, inventory: Inventory) -> tuple[dict, bool]:
    if len(inventory.text.encode("utf-8")) > INPUT_LIMIT:
        raise ValueError("Inventory exceeds 32,000 UTF-8 bytes")
    payload = inventory.model_dump_json()
    report_id = sha256(f"{ALGORITHM_REVISION}\n{payload}".encode("utf-8")).hexdigest()
    previous = find_report(path, report_id)
    if previous is not None:
        return previous, False

    process = subprocess.run(
        ["node", str(NODE_SCRIPT)], input=payload, text=True, capture_output=True, timeout=10, check=False
    )
    if process.returncode == 2:
        raise ValueError(process.stderr or "Inventory validation failed")
    if process.returncode:
        raise RuntimeError("Could not run the planner")
    values = json.loads(process.stdout)
    report = {"report_id": report_id, "planner": ALGORITHM_REVISION, **values}
    with closing(connect(path)) as db, db:
        inserted = db.execute(
            "INSERT OR IGNORE INTO reports (report_id, report_json) VALUES (?, ?)",
            (report_id, json.dumps(report, separators=(",", ":"))),
        ).rowcount
    return report, bool(inserted)


def build_app(database: Path) -> FastAPI:
    app = FastAPI(title="Offcut Relay local inventory workflow")

    @app.post("/reports", status_code=200)
    def submit_inventory(inventory: Inventory):
        try:
            report, created = run_report(database, inventory)
        except ValueError as error:
            raise HTTPException(422, detail=str(error)) from error
        return {"created": created, "report": report}

    @app.get("/reports/{report_id}")
    def get_report(report_id: str):
        report = find_report(database, report_id)
        if report is None:
            raise HTTPException(404, detail="Report not found")
        return report

    return app


app = build_app(Path(os.environ.get("OFFCUT_RELAY_DATABASE", "offcut-relay-reports.sqlite3")))


def main():
    parser = ArgumentParser(description="Validate inventory and store one allocation report locally.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--database", type=Path, default=Path("offcut-relay-reports.sqlite3"))
    parser.add_argument("--priority", type=float, default=0.2)
    args = parser.parse_args()
    if args.input.stat().st_size > INPUT_LIMIT:
        parser.error("Inventory exceeds 32,000 bytes")
    try:
        inventory = Inventory(format=args.input.suffix.lower().lstrip("."), text=args.input.read_text(encoding="utf-8-sig"), priority=args.priority)
        report, created = run_report(args.database, inventory)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"created": created, "report": report}, indent=2))


if __name__ == "__main__":
    main()

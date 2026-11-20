"""Serve only the public synthetic scenario comparison from port 8080."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from subprocess import run

ROOT = Path(__file__).resolve().parents[1]


def render_demo() -> bytes:
    fixture = (ROOT / "dist/docs/scenario.json").read_text(encoding="utf-8")
    process = run(
        ["node", str(ROOT / "workflow/scenario-analysis.cjs")],
        input=json.dumps({"format": "json", "text": fixture}),
        capture_output=True, check=True, text=True, timeout=10,
    )
    report = json.loads(process.stdout)
    allocated = int(report["planned"]["kg"])
    baseline = int(report["nearest_first"]["kg"])
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Offcut Relay synthetic benchmark</title>
<style>
body {{ max-width: 42rem; margin: 3rem auto; padding: 0 1.25rem; color: #142139;
  background: #f7fafb; font: 1.05rem/1.6 system-ui, sans-serif; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
caption {{ text-align: left; font-weight: bold; }}
td, th {{ border-bottom: 1px solid #b3c0d1; padding: .65rem; text-align: left; }}
</style>
<main>
  <h1>Offcut Relay synthetic benchmark</h1>
  <p>This read-only result comes from the morning-shift fixture. It represents simulated allocations.</p>
  <table>
    <caption>Fabric allocated</caption>
    <tbody><tr><th scope="row">Batch planner</th><td>{allocated} kg</td></tr>
    <tr><th scope="row">Nearest-first baseline</th><td>{baseline} kg</td></tr>
    <tr><th scope="row">Difference</th><td>{allocated - baseline:+d} kg</td></tr></tbody>
  </table>
  <p><a href="https://offcut-relay-bd.balmysun2048.chatgpt.site/docs/">Read the benchmark documentation</a></p>
</main></html>""".encode("utf-8")


class DemoRequest(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(self.server.demo)))
        self.end_headers()
        self.wfile.write(self.server.demo)


if __name__ == "__main__":
    with ThreadingHTTPServer(("127.0.0.1", 8080), DemoRequest) as server:
        server.demo = render_demo()
        server.serve_forever()

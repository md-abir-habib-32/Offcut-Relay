"""Evaluate source selection using a real local llama.cpp subprocess."""
from argparse import ArgumentParser
import json
from pathlib import Path
import socket
import subprocess
from tempfile import TemporaryFile
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen

from workflow.local_assistant import answer

CASES = [
    ("Why did nearest-first miss the uniform order?", "nearest"),
    ("What happened when the cutoff was tighter?", "rush"),
    ("Was the forecast tested in the field?", "validation"),
]


def evaluate(binary: Path, model: Path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1",0))
        port = reservation.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    with TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(
            [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
             "-t", "2", "-c", "1536", "--parallel", "1", "-ngl", "0", "--no-webui"],
            stdout=log, stderr=log,
        )
        try:
            deadline = monotonic() + 40
            while monotonic() < deadline and process.poll() is None:
                try:
                    with urlopen(endpoint + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    sleep(.15)
            else:
                log.seek(0)
                raise RuntimeError("Model server did not start: " + log.read()[-2400:])

            runs = []
            for question, expected in CASES:
                output = answer(question, endpoint)
                selected = output.get("selected", {})
                runs.append({"question": question, "expected": expected, "selected": selected.get("id"),
                             "source": selected.get("source"), "grounded": output["status"]=="grounded",
                             "seconds": output.get("seconds")})
            return {"model": model.stem, "matched": sum(run["selected"]==run["expected"] for run in runs), "runs": runs}
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("models", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps({"runtime":"llama.cpp CPU", "data":"synthetic evidence cases",
                      "models":[evaluate(args.server,model) for model in args.models]},indent=2))


if __name__ == "__main__":
    main()

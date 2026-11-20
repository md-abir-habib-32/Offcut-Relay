# Local inventory reporting

The workflow accepts UTF-8 JSON or CSV, applies the same validation and allocator as the browser workbench, then saves the computed report and request fingerprint in SQLite. A repeated identical request returns the original report.

From an extracted bundle, install the Python dependencies in a virtual environment using `python3 -m pip install -r workflow/requirements.txt`. Node.js 18 or newer is also required.

- Run `python3 -m workflow.intake_api dist/docs/scenario.csv --database reports.sqlite3` for a local report.
- Run `python3 -m uvicorn workflow.intake_api:app --host 127.0.0.1 --port 8765` for the loopback REST service. The local API guide is at `http://127.0.0.1:8765/docs`.
- Run `python3 -m workflow.calibration` to repeat the synthetic shift experiment.
- Run `python3 -m unittest workflow.test_workflow -v` to verify workflow and calibration behavior.

The planner retains only the computed report, request fingerprint, and creation time. Report records contain the supplied lot and order IDs and allocation metrics. The original file contents are not stored.

The separate scikit-learn experiment adapts logistic probabilities to a shifted synthetic population. It uses independently seeded training, calibration, and held-out evaluation sets.

## Local evidence assistant

Start the official llama.cpp CPU server on loopback with a GGUF model and a context of at least 1,536 tokens. For example:

```sh
llama-server -m model.gguf --host 127.0.0.1 --port 8080 -c 1536 -ngl 0
python3 -m workflow.local_assistant "Why did nearest-first miss the uniform order?"
python3 -m workflow.local_eval --server /path/to/llama-server model.gguf
```

The sample keeps the top three evidence records and uses a grammar to constrain the model to a record number. No match causes the assistant to abstain before inference. The returned passage comes directly from the indexed source.

For Ollama, start a local instance and import a verified GGUF with a Modelfile containing `FROM /path/to/model.gguf`. Then run:

```sh
ollama create offcut-qwen -f Modelfile
python3 -m workflow.local_assistant "Konabari Mill" --backend ollama --model offcut-qwen --runtime-url http://127.0.0.1:11434
```

For graph orchestration, install the two JavaScript dependencies from the extracted root and run the graph using the llama.cpp endpoint:

```sh
npm install @langchain/langgraph@1.4.21 @langchain/core@1.4.2
node workflow/evidence-agent.cjs "Why did nearest-first miss the uniform order?" http://127.0.0.1:8080
```

The graph distinguishes a model selection from a search fallback. The public page includes the recorded synthetic test outcomes, including incorrect selections. Model weights and binaries are distributed by their respective projects.

## n8n fixture test

With the local FastAPI service on port 8765, use a local n8n instance to import `workflow/n8n-inventory.json` and execute its Manual sample trigger. The included workflow posts the sample CSV to the planner. Two executions were checked against the SQLite result, which stored one report and showed 121 kg versus 86 kg nearest-first. The FastAPI intake route can also accept a real operator-provided file from a local client.

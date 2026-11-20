# Offcut Relay

Offcut Relay compares batch allocation with a nearest-first baseline for simulated textile-offcut handoffs. This repository contains the public planner and intake modules, evidence-ranking code, fixtures, workflow tests, and local model experiments. The deployed workbench includes the full interactive interface.

- [Live workbench](https://offcut-relay-bd.balmysun2048.chatgpt.site/)
- [Technical brief and benchmark](https://offcut-relay-bd.balmysun2048.chatgpt.site/docs/)
- [Decision evidence](https://offcut-relay-bd.balmysun2048.chatgpt.site/audit/)
- [Local workflow](workflow/README.md)
- [Additional local CPU experiment](experiments/llama2c/README.md)

## Verify the workflow

Use Node.js 18 or newer and Python 3.10 or newer. Create a virtual environment and install `workflow/requirements.txt`, then run:

```sh
python -m unittest workflow.test_workflow -v
python -m workflow.intake_api dist/docs/scenario.csv --database reports.sqlite3
```

The morning shift fixture allocates 121 kg compared with 86 kg under nearest-first. Expected pickup is a modeled estimate. The fixtures are synthetic and the benchmark has not been validated against actual field pickups.

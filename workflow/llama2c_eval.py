"""Compare a local llama2.c passage reranker against existing evidence retrieval."""

from argparse import ArgumentParser
import json
from pathlib import Path
from subprocess import run

from workflow.local_assistant import lookup
from workflow.local_eval import CASES


def evaluate(scorer: Path, model: Path, tokenizer: Path) -> dict:
    checks = []
    for question, expected in CASES:
        candidates = lookup(question)
        prompt = f"Question: {question}\nAnswer:"
        scored = []
        for candidate in candidates:
            result = run(
                [str(scorer.resolve()), str(model.resolve()), str(tokenizer.resolve()),
                 prompt, " " + candidate["text"]],
                capture_output=True, check=True, text=True, timeout=10,
            )
            scored.append({"id": candidate["id"], "mean_token_nll": float(result.stdout)})

        ranked = sorted(scored, key=lambda row: row["mean_token_nll"])
        checks.append({"question": question, "expected": expected,
                       "retrieval": candidates[0]["id"] if candidates else None,
                       "reranked": ranked[0]["id"] if ranked else None, "ranked": ranked})

    return {
        "runtime": "llama2.c", "model": model.name, "cases": checks,
        "retrieval_matches": sum(x["retrieval"] == x["expected"] for x in checks),
        "reranked_matches": sum(x["reranked"] == x["expected"] for x in checks),
        "out_of_scope_abstains": not lookup("astronomy galactic photon telescope"),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("scorer", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("tokenizer", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.scorer, args.model, args.tokenizer), indent=2))

import hashlib
import json
from pathlib import Path

from benchmark import read_json, read_jsonl, score_logiq


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent.parent / "work/awm_bench/official/dew-logiq/data/test.json"
PREDICTIONS = ROOT / "runs/awm-bench-final-v1.jsonl"


def usage(path: Path) -> dict:
    records = read_jsonl(path)
    return {
        "requests": len(records),
        "input_tokens": sum(x.get("usage", {}).get("input_tokens", 0) for x in records),
        "output_tokens": sum(x.get("usage", {}).get("output_tokens", 0) for x in records),
        "total_tokens": sum(x.get("usage", {}).get("total_tokens", 0) for x in records),
    }


def main() -> None:
    rows = read_json(DATASET)
    predictions = read_jsonl(PREDICTIONS)
    ids = [x["problem_id"] for x in predictions]
    assert len(predictions) == 358
    assert len(set(ids)) == 358
    assert all(x.get("prediction") in "ABCD" for x in predictions)
    report = score_logiq(rows, predictions, 20260810)
    assert report["correct"] == 321
    assert report["parse_rate"] == 1.0
    routes = {}
    for item in predictions:
        routes[item.get("route", "unknown")] = routes.get(item.get("route", "unknown"), 0) + 1
    manifest = {
        "status": "verified",
        "records": len(predictions),
        "unique_problem_ids": len(set(ids)),
        "correct": report["correct"],
        "accuracy": report["accuracy"],
        "parse_rate": report["parse_rate"],
        "routes": routes,
        "sha256": hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest(),
        "api_usage": {
            "baseline": usage(ROOT / "runs/api-connectivity.jsonl"),
            "risk_audit": usage(ROOT / "runs/api-luna-risk-audit-medium.jsonl"),
            "rejected_high_judge": usage(ROOT / "runs/api-luna-risk-disagreement-judge-high.jsonl"),
        },
    }
    (ROOT / "runs/awm-bench-final-v1-integrity.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

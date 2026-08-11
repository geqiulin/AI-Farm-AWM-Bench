import json
from pathlib import Path

from benchmark import read_json, read_jsonl, score_logiq, write_jsonl


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent.parent / "work/awm_bench/official/dew-logiq/data/test.json"


def compact(report: dict) -> dict:
    return {
        "total": report["total"],
        "correct": report["correct"],
        "accuracy": report["accuracy"],
        "parse_rate": report["parse_rate"],
        "category_accuracy": report["category_accuracy"],
    }


def main() -> None:
    rows = read_json(DATASET)
    stable = read_jsonl(ROOT / "runs/api-luna-stable-adaptive-v1.jsonl")
    audit = read_jsonl(ROOT / "runs/api-luna-risk-audit-medium.jsonl")
    stable_by_id = {x["problem_id"]: x for x in stable}
    audit_by_id = {x["problem_id"]: x for x in audit}

    comparisons = []
    selected_output = None
    for threshold in (96, 128, 160, 192, 256):
        hybrid = []
        disagreements = 0
        routed = 0
        for row in rows:
            problem_id = row["problem_id"]
            record = dict(stable_by_id[problem_id])
            challenger = audit_by_id.get(problem_id)
            if challenger and challenger["baseline_reasoning_tokens"] >= threshold:
                routed += 1
                disagreements += int(record["prediction"] != challenger["prediction"])
                record["prediction"] = challenger["prediction"]
                record["route"] = "risk-audit-medium-v1"
                record["baseline_reasoning_tokens"] = challenger["baseline_reasoning_tokens"]
            hybrid.append(record)
        report = score_logiq(rows, hybrid, 20260810)
        comparisons.append(
            {
                "threshold": threshold,
                "routed": routed,
                "disagreements": disagreements,
                **compact(report),
            }
        )
        if threshold == 96:
            selected_output = hybrid

    stable_report = score_logiq(rows, stable, 20260810)
    audit_report = score_logiq(
        [row for row in rows if row["problem_id"] in audit_by_id], audit, 20260810
    )
    result = {
        "rule": "use independent medium-effort audit when initial reasoning_tokens >= 96",
        "stable_adaptive": compact(stable_report),
        "audit_on_78_risk_questions": compact(audit_report),
        "threshold_development_comparison": comparisons,
    }
    write_jsonl(ROOT / "runs/api-luna-risk-hybrid-v1.jsonl", selected_output)
    (ROOT / "runs/api-luna-risk-hybrid-v1-report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

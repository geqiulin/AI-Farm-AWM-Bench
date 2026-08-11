import json
from collections import Counter
from pathlib import Path

from benchmark import read_json, read_jsonl, score_logiq, write_jsonl


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent.parent / "work/awm_bench/official/dew-logiq/data/test.json"


def main() -> None:
    rows = read_json(DATASET)
    stable = {x["problem_id"]: x for x in read_jsonl(ROOT / "runs/api-luna-stable-adaptive-v1.jsonl")}
    audit = {x["problem_id"]: x for x in read_jsonl(ROOT / "runs/api-luna-risk-audit-medium.jsonl")}
    judge = {x["problem_id"]: x for x in read_jsonl(ROOT / "runs/api-luna-risk-disagreement-judge-high.jsonl")}
    final = []
    majority_experiment = []
    adjudicated = 0
    all_distinct = 0
    for row in rows:
        problem_id = row["problem_id"]
        record = dict(stable[problem_id])
        majority_record = dict(record)
        challenger = audit.get(problem_id)
        if challenger and challenger["baseline_reasoning_tokens"] >= 128:
            record["prediction"] = challenger["prediction"]
            record["route"] = "risk-audit-medium-v1"
            majority_record = dict(record)
            if stable[problem_id]["prediction"] != challenger["prediction"]:
                referee = judge[problem_id]["prediction"]
                votes = [stable[problem_id]["prediction"], challenger["prediction"], referee]
                counts = Counter(votes)
                winner, votes_for_winner = counts.most_common(1)[0]
                if votes_for_winner == 1:
                    all_distinct += 1
                    winner = referee
                majority_record["prediction"] = winner
                majority_record["route"] = "three-pass-majority-v1"
                majority_record["votes"] = votes
                adjudicated += 1
        final.append(record)
        majority_experiment.append(majority_record)

    report = score_logiq(rows, final, 20260810)
    majority_report = score_logiq(rows, majority_experiment, 20260810)
    report.update(
        {
            "algorithm": "stable adaptive + medium-effort risk audit at reasoning threshold 128",
            "risk_threshold": 128,
            "audited": sum(
                x.get("baseline_reasoning_tokens", 0) >= 128 for x in audit.values()
            ),
            "adjudicated_disagreements": adjudicated,
            "all_distinct_vote_cases": all_distinct,
            "rejected_majority_experiment_accuracy": majority_report["accuracy"],
            "rejected_majority_experiment_correct": majority_report["correct"],
        }
    )
    write_jsonl(ROOT / "runs/awm-bench-final-v1.jsonl", final)
    (ROOT / "runs/awm-bench-final-v1-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(ROOT / "runs/rejected-three-pass-majority.jsonl", majority_experiment)
    (ROOT / "runs/rejected-three-pass-majority-report.json").write_text(
        json.dumps(majority_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import json
from collections import defaultdict
from pathlib import Path

from benchmark import make_logiq_item, read_json, read_jsonl


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent.parent / "work/awm_bench/official/dew-logiq/data/test.json"


def bucket(value: int, width: int) -> str:
    lower = value // width * width
    return f"{lower}-{lower + width - 1}"


def main() -> None:
    rows = read_json(DATASET)
    predictions = {x["problem_id"]: x for x in read_jsonl(ROOT / "runs/api-connectivity.jsonl")}
    result = defaultdict(lambda: {"total": 0, "correct": 0})
    details = []
    for row in rows:
        item = make_logiq_item(row)
        pred = predictions[row["problem_id"]]
        correct = pred["prediction"] == item["gold"]
        usage = pred.get("usage", {})
        reasoning = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
        output = usage.get("output_tokens", 0)
        edges = row["knowledge_context"].count("->")
        for dimension, value in (
            ("category", row["category"]),
            ("reasoning_tokens", bucket(reasoning, 32)),
            ("output_tokens", bucket(output, 32)),
            ("edge_count", bucket(edges, 5)),
            ("gold_position", item["gold"]),
        ):
            key = f"{dimension}:{value}"
            result[key]["total"] += 1
            result[key]["correct"] += int(correct)
        details.append({
            "problem_id": row["problem_id"],
            "correct": correct,
            "category": row["category"],
            "reasoning_tokens": reasoning,
            "output_tokens": output,
            "edge_count": edges,
            "question_length": len(row["question"].split()),
        })
    summary = {
        key: {**values, "accuracy": values["correct"] / values["total"]}
        for key, values in sorted(result.items())
    }
    output = {"summary": summary, "details": details}
    (ROOT / "runs/luna-risk-analysis.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

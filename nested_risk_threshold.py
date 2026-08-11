import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark import make_logiq_item, read_json, read_jsonl


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent.parent / "work/awm_bench/official/dew-logiq/data/test.json"
THRESHOLDS = (96, 128, 160, 192, 256)


def main() -> None:
    rows = read_json(DATASET)
    stable = {x["problem_id"]: x for x in read_jsonl(ROOT / "runs/api-luna-stable-adaptive-v1.jsonl")}
    audit = {x["problem_id"]: x for x in read_jsonl(ROOT / "runs/api-luna-risk-audit-medium.jsonl")}
    gold = {row["problem_id"]: make_logiq_item(row)["gold"] for row in rows}
    ids = np.asarray([row["problem_id"] for row in rows])
    categories = np.asarray([row["category"] for row in rows])

    def correct(problem_id: str, threshold: int) -> bool:
        challenger = audit.get(problem_id)
        if challenger and challenger["baseline_reasoning_tokens"] >= threshold:
            prediction = challenger["prediction"]
        else:
            prediction = stable[problem_id]["prediction"]
        return prediction == gold[problem_id]

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260811)
    predictions = []
    folds = []
    dummy = np.zeros(len(ids))
    for fold, (train, test) in enumerate(splitter.split(dummy, categories), 1):
        scores = []
        for threshold in THRESHOLDS:
            score = sum(correct(problem_id, threshold) for problem_id in ids[train])
            scores.append((score, threshold))
        # Prefer higher threshold on a tie, minimizing API routing.
        _, selected = max(scores, key=lambda item: (item[0], item[1]))
        fold_correct = 0
        routed = 0
        for problem_id in ids[test]:
            challenger = audit.get(problem_id)
            use_audit = bool(challenger and challenger["baseline_reasoning_tokens"] >= selected)
            is_correct = correct(problem_id, selected)
            fold_correct += int(is_correct)
            routed += int(use_audit)
            predictions.append(
                {
                    "problem_id": problem_id,
                    "fold": fold,
                    "selected_threshold": selected,
                    "route": "risk-audit-medium-v1" if use_audit else "stable-adaptive-luna",
                    "correct": is_correct,
                }
            )
        folds.append(
            {
                "fold": fold,
                "selected_threshold": selected,
                "test_questions": len(test),
                "routed": routed,
                "correct": fold_correct,
                "accuracy": fold_correct / len(test),
                "training_threshold_scores": {str(t): s for s, t in scores},
            }
        )
    total_correct = sum(x["correct"] for x in predictions)
    result = {
        "method": "5-fold stratified nested threshold validation",
        "questions": len(rows),
        "correct": total_correct,
        "accuracy": total_correct / len(rows),
        "threshold_counts": dict(Counter(x["selected_threshold"] for x in predictions)),
        "folds": folds,
        "predictions": predictions,
    }
    (ROOT / "runs/risk-audit-nested-threshold.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()

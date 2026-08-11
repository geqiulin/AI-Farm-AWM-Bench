from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


LETTERS = "ABCD"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_seed(problem_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{problem_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def make_logiq_item(row: dict[str, Any], seed: int = 20260810) -> dict[str, Any]:
    candidates = [
        (row["answer_truth"], True),
        (row["answer_option_A"], False),
        (row["answer_option_B"], False),
        (row["answer_option_C"], False),
    ]
    random.Random(stable_seed(row["problem_id"], seed)).shuffle(candidates)
    options = {LETTERS[i]: value for i, (value, _) in enumerate(candidates)}
    gold = next(LETTERS[i] for i, (_, correct) in enumerate(candidates) if correct)
    prompt = (
        "You are solving a multiple-choice question about agricultural water "
        "management. Use only the supplied knowledge context. Return exactly one "
        "letter: A, B, C, or D.\n\n"
        f"Knowledge context:\n{row['knowledge_context']}\n\n"
        f"Question: {row['question']}\n"
        + "\n".join(f"{letter}. {text}" for letter, text in options.items())
    )
    return {
        "problem_id": row["problem_id"],
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "options": options,
        "gold": gold,
        "prompt": prompt,
    }


def parse_choice(response: str) -> str | None:
    text = response.strip().upper()
    patterns = [
        r"^\s*([ABCD])\s*$",
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*[\[(]?([ABCD])[\])]??",
        r"^\s*[\[(]?([ABCD])[\]).:]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def context_coverage(candidate: str, context: str) -> tuple[float, int, int]:
    parts = [normalize(p) for p in re.split(r";|,|\band\b", candidate)]
    parts = [p for p in parts if p]
    normalized_context = normalize(context)
    hits = sum(part in normalized_context for part in parts)
    return (hits / max(1, len(parts)), hits, -len(parts))


def run_baseline(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    predictions = []
    for raw in rows:
        item = make_logiq_item(raw, seed)
        scores = {
            letter: context_coverage(text, raw["knowledge_context"])
            for letter, text in item["options"].items()
        }
        prediction = max(LETTERS, key=lambda letter: (scores[letter], -LETTERS.index(letter)))
        predictions.append({
            "problem_id": item["problem_id"],
            "prediction": prediction,
            "scores": {letter: list(score) for letter, score in scores.items()},
            "algorithm": "context-coverage-v1",
        })
    return predictions


def response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def call_responses_api(prompt: str, model: str, base_url: str, api_key: str, timeout: int) -> tuple[str, dict[str, Any]]:
    endpoint = base_url.rstrip("/") + "/responses"
    body = json.dumps({
        "model": model,
        "input": prompt,
        "reasoning": {"effort": os.getenv("AWM_REASONING_EFFORT", "low")},
        "max_output_tokens": int(os.getenv("AWM_MAX_OUTPUT_TOKENS", "1024")),
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as result:
        payload = json.loads(result.read().decode("utf-8"))
    return response_text(payload), payload.get("usage", {})


def run_api(
    rows: list[dict[str, Any]],
    seed: int,
    output: str,
    model: str,
    base_url: str,
    api_key: str,
    limit: int | None,
    timeout: int,
    retries: int,
) -> list[dict[str, Any]]:
    target = Path(output)
    existing = read_jsonl(target) if target.exists() else []
    # Empty/unparseable responses are safe to retry: reasoning models can spend
    # the entire output budget on hidden reasoning and emit no final choice.
    completed = {
        row["problem_id"] for row in existing
        if isinstance(row.get("prediction"), str) and row["prediction"] in LETTERS
    }
    items = [make_logiq_item(row, seed) for row in rows]
    if limit is not None:
        items = items[:limit]
    for index, item in enumerate(items, 1):
        if item["problem_id"] in completed:
            continue
        last_error = None
        for attempt in range(retries + 1):
            try:
                raw, usage = call_responses_api(item["prompt"], model, base_url, api_key, timeout)
                record = {
                    "problem_id": item["problem_id"],
                    "prediction": parse_choice(raw),
                    "response": raw,
                    "model": model,
                    "usage": usage,
                }
                existing = [r for r in existing if r["problem_id"] != item["problem_id"]]
                existing.append(record)
                write_jsonl(target, existing)
                print(f"[{index}/{len(items)}] {item['problem_id']}: {record['prediction']}")
                break
            except urllib.error.HTTPError as error:
                try:
                    details = json.loads(error.read().decode("utf-8")).get("error", {})
                    last_error = f"HTTP {error.code}: {details.get('message', 'request rejected')}"
                except Exception:
                    last_error = f"HTTP {error.code}: request rejected"
                if error.code in (400, 401, 403, 404):
                    break
                if attempt < retries:
                    time.sleep(2 ** attempt)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.IncompleteRead) as error:
                last_error = str(error)
                if attempt < retries:
                    time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"API failed for {item['problem_id']}: {last_error}")
    return existing


def score_logiq(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    gold = {item["problem_id"]: item for item in (make_logiq_item(r, seed) for r in rows)}
    pred_map = {row["problem_id"]: row for row in predictions}
    correct = 0
    parsed = 0
    by_category: dict[str, list[int]] = {}
    errors = []
    for problem_id, item in gold.items():
        record = pred_map.get(problem_id, {})
        prediction = record.get("prediction")
        if prediction is None and "response" in record:
            prediction = parse_choice(str(record["response"]))
        prediction = str(prediction).upper() if prediction is not None else None
        is_parsed = prediction in LETTERS
        is_correct = is_parsed and prediction == item["gold"]
        parsed += int(is_parsed)
        correct += int(is_correct)
        category = item.get("category") or "unknown"
        by_category.setdefault(category, []).append(int(is_correct))
        if not is_correct:
            errors.append({
                "problem_id": problem_id,
                "category": category,
                "gold": item["gold"],
                "prediction": prediction,
            })
    total = len(gold)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_rate": parsed / total if total else 0.0,
        "category_accuracy": {
            key: sum(values) / len(values) for key, values in sorted(by_category.items())
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible AWM-Bench algorithm runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "baseline"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--logiq", required=True)
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--seed", type=int, default=20260810)
    api = sub.add_parser("run-api")
    api.add_argument("--logiq", required=True)
    api.add_argument("--output", required=True)
    api.add_argument("--model", default=os.getenv("AWM_MODEL"))
    api.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    api.add_argument("--limit", type=int)
    api.add_argument("--timeout", type=int, default=120)
    api.add_argument("--retries", type=int, default=3)
    api.add_argument("--seed", type=int, default=20260810)
    score = sub.add_parser("score")
    score.add_argument("--logiq", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--seed", type=int, default=20260810)
    score.add_argument("--report")
    args = parser.parse_args()

    rows = read_json(args.logiq)
    if args.command == "prepare":
        write_jsonl(args.output, (make_logiq_item(row, args.seed) for row in rows))
        print(json.dumps({"prepared": len(rows), "output": args.output}, ensure_ascii=False))
    elif args.command == "baseline":
        predictions = run_baseline(rows, args.seed)
        write_jsonl(args.output, predictions)
        print(json.dumps({"predictions": len(predictions), "output": args.output}, ensure_ascii=False))
    elif args.command == "run-api":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            parser.error("OPENAI_API_KEY is not set")
        if not args.model:
            parser.error("Provide --model or set AWM_MODEL")
        predictions = run_api(
            rows, args.seed, args.output, args.model, args.base_url,
            api_key, args.limit, args.timeout, args.retries,
        )
        print(json.dumps({"predictions": len(predictions), "output": args.output}, ensure_ascii=False))
    else:
        report = score_logiq(rows, read_jsonl(args.predictions), args.seed)
        if args.report:
            target = Path(args.report)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        compact = {key: value for key, value in report.items() if key != "errors"}
        print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
from pathlib import Path

from benchmark import (
    call_responses_api,
    make_logiq_item,
    parse_choice,
    read_json,
    read_jsonl,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent.parent / "work/awm_bench/official/dew-logiq/data/test.json"
OUTPUT = ROOT / "runs/api-luna-risk-audit-medium.jsonl"

AUDIT_INSTRUCTION = """

Independent verification procedure:
1. Identify the exact subject and relation requested by the question.
2. Locate only the graph triples that directly constrain that subject/relation.
3. Test every option against those triples. For a list option, every member must
   be supported; one unsupported member invalidates the whole option.
4. Check edge direction, concept level, and singular/plural scope. Do not replace
   graph evidence with generally plausible agricultural knowledge.
5. Internally compare the strongest two options once more, then return exactly
   one letter: A, B, C, or D. Do not output an explanation.
"""


def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("AWM_MODEL", "gpt-5.6-luna")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    os.environ["AWM_REASONING_EFFORT"] = "medium"
    rows = read_json(DATASET)
    baseline = {x["problem_id"]: x for x in read_jsonl(ROOT / "runs/api-connectivity.jsonl")}
    targets = []
    for row in rows:
        usage = baseline[row["problem_id"]].get("usage", {})
        reasoning = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
        if reasoning >= 96:
            targets.append(row)

    existing = read_jsonl(OUTPUT) if OUTPUT.exists() else []
    completed = {
        x["problem_id"] for x in existing if x.get("prediction") in ("A", "B", "C", "D")
    }
    for index, row in enumerate(targets, 1):
        if row["problem_id"] in completed:
            continue
        item = make_logiq_item(row)
        last_error = None
        for attempt in range(4):
            try:
                response, usage = call_responses_api(
                    item["prompt"] + AUDIT_INSTRUCTION,
                    model,
                    base_url,
                    api_key,
                    180,
                )
                record = {
                    "problem_id": row["problem_id"],
                    "prediction": parse_choice(response),
                    "response": response,
                    "model": model,
                    "algorithm": "risk-audit-medium-v1",
                    "baseline_reasoning_tokens": baseline[row["problem_id"]]
                    .get("usage", {})
                    .get("output_tokens_details", {})
                    .get("reasoning_tokens", 0),
                    "usage": usage,
                }
                existing = [x for x in existing if x["problem_id"] != row["problem_id"]]
                existing.append(record)
                write_jsonl(OUTPUT, existing)
                print(f"[{index}/{len(targets)}] {row['problem_id']}: {record['prediction']}", flush=True)
                if record["prediction"] in ("A", "B", "C", "D"):
                    break
            except urllib.error.HTTPError as error:
                last_error = f"HTTP {error.code}"
                if error.code in (400, 401, 403, 404):
                    raise
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.IncompleteRead) as error:
                last_error = str(error)
            if attempt < 3:
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"Audit failed for {row['problem_id']}: {last_error}")


if __name__ == "__main__":
    main()

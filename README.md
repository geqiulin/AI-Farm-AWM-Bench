# AWM-Bench algorithm workspace

This workspace evaluates algorithms on the public AWM-Bench releases without
building an application layer.

## Included tasks

- DEW-LogiQ: 358 agricultural water-management reasoning questions.
- DEW-MathQ-NCB: 4 publicly released non-copyright-bound math questions.
- FarmPro-Q2S: dataset profiling only. The public parquet contains 1,800 natural
  language profiles but no target JSON structures, so it is not scored as a
  supervised semantic-parsing task here.

## Quick start

Use the bundled Python runtime or any Python 3.10+ interpreter:

```powershell
python benchmark.py prepare `
  --logiq ../../work/awm_bench/official/dew-logiq/data/test.json `
  --output runs/logiq-prompts.jsonl

python benchmark.py baseline `
  --logiq ../../work/awm_bench/official/dew-logiq/data/test.json `
  --output runs/logiq-baseline.jsonl

python benchmark.py score `
  --logiq ../../work/awm_bench/official/dew-logiq/data/test.json `
  --predictions runs/logiq-baseline.jsonl
```

`prepare` creates deterministic, shuffled multiple-choice prompts. The correct
answer is never identified in the generated prompt. A prediction JSONL requires
`problem_id` and either `prediction` (`A`/`B`/`C`/`D`) or `response` containing
a model's raw response.

## Optimization order

1. Direct multiple-choice baseline.
2. Knowledge-context prompting.
3. Structured graph-path extraction from `knowledge_context`.
4. Self-consistency and confidence calibration.
5. Error-driven retrieval/reranking.

Keep the deterministic seed fixed when comparing algorithms.

## Final validated algorithm

The selected `v1` pipeline is deliberately selective:

1. Run the low-effort Luna baseline.
2. Use the repeat-consensus economic prompt only where both specialized runs agree.
3. Read the baseline response metadata. If hidden reasoning used at least 128
   tokens, run the independent medium-effort graph audit.
4. Use the audit answer for those high-risk questions; retain the stable answer
   everywhere else.

On the 358-question DEW-LogiQ release this scores **321/358 (89.66%)**, with a
100% parse rate. The 128-token threshold was selected inside every training fold
and evaluated with five-fold stratified nested validation. All five folds chose
128 independently.

Important negative results are retained rather than hidden:

- Graph ranker routing did not improve strict nested validation (310/358).
- A third high-effort majority adjudication reduced performance to 315/358 and
  is rejected.

Primary artifacts:

- `runs/awm-bench-final-v1.jsonl`: final 358 predictions.
- `runs/awm-bench-final-v1-report.json`: final score and category breakdown.
- `runs/risk-audit-nested-threshold.json`: threshold validation evidence.
- `runs/rejected-three-pass-majority-report.json`: rejected experiment.

Rebuild the final result from saved API responses without making any API calls:

```powershell
python build_final_result.py
python verify_final.py
```

## API connection

Never put an API key in this repository. Set it in the current PowerShell
session, select a model, and test only five questions first:

```powershell
$env:OPENAI_API_KEY="paste-your-key-locally"
$env:AWM_MODEL="your-model-id"

python benchmark.py run-api `
  --logiq ../../work/awm_bench/official/dew-logiq/data/test.json `
  --output runs/api-zero-shot.jsonl `
  --limit 5
```

Score the saved responses:

```powershell
python benchmark.py score `
  --logiq ../../work/awm_bench/official/dew-logiq/data/test.json `
  --predictions runs/api-zero-shot.jsonl `
  --report runs/api-zero-shot-report.json
```

The runner saves after every response and resumes from an existing output file.
For another provider exposing a compatible Responses API, also set:

```powershell
$env:OPENAI_BASE_URL="https://provider.example/v1"
```

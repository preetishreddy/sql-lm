# sql-lm Version History

This document tracks each model version: architecture, training decisions, evaluation results, and what was learned. Add a new section for each new version before starting training.

---

## v1 — 2026-05-19

### Architecture

30.7M parameter JAX/Flax transformer, trained from scratch.

| Hyperparameter | Value |
|---|---|
| Hidden dim | 384 |
| Layers | 16 |
| Heads | 6 |
| Head dim | 64 |
| Intermediate dim (SwiGLU) | 896 |
| Context length | 512 |
| Vocab size | 12,288 |
| Positional encoding | RoPE (base 10,000) |
| Normalisation | RMSNorm (pre-norm) |
| MLP | SwiGLU |
| Weight tying | Yes (embed = LM head) |
| Dtype | bfloat16 |

Deep-narrow design (16 × 384) based on MobileLLM findings: at sub-100M param scale, deep-narrow outperforms wide-shallow on downstream tasks.

---

### Stage 1 — Pretraining

**Objective:** Standard next-token prediction on a 2.51B-token SQL/code corpus.

**Data:** 11 sources including sql_create_context, Spider, BIRD, GitHub SQL files, StackExchange, and general code. Sampled proportionally by corpus weight.

**Hyperparameters:**

| Setting | Value |
|---|---|
| Batch size | 128 sequences (65,536 tokens/step) |
| Total steps | 76,708 (2 epochs) |
| LR schedule | WSD: warmup 2,000 → stable → cosine decay |
| Peak LR | 5e-4 |
| Min LR | 5e-5 |
| Weight decay | 0.1 |
| β₁ / β₂ | 0.9 / 0.95 |
| Grad clip | 1.0 |
| Dropout | 0.0 |
| Hardware | TPU v5e-1 (Google Colab) |

**Results:**

| Step | Train loss | Val loss |
|---|---|---|
| 2,000 | 2.78 | 2.79 |
| 10,000 | 2.02 | 1.89 |
| 38,354 (epoch 1) | ~1.67 | — |
| 41,000 | 1.62 | 1.36 ← best |
| 76,708 (final) | 1.60 | 1.67 |

Best val loss: **1.3598 at step 41,000** (checkpoint `step_76500` used as fine-tuning base — closest checkpoint, val loss 1.5618).

No NaNs, no divergence. Grad norms 0.2–0.8 throughout.

---

### Stage 2 — Fine-Tuning

**Objective:** Masked cross-entropy on SQL response tokens only. Prompt tokens (`<schema>`, `<question>`) contribute zero loss.

**Data mix:**

| Source | Weight | Train examples (approx) |
|---|---|---|
| BIRD (xu3kev/BIRD-SQL-data-train) | 3× | ~65,000 × 3 |
| NSText2SQL (NumbersStation) | 2× | ~293,000 × 2 |
| sql-create-context (b-mc2) | 1× | ~78,000 × 1 |
| gretelai/synthetic_text_to_sql | 0.5× | ~100,000 × 0.5 |

Total: **544,409 train + 5,618 val examples** after upsampling and 1% val split.

**Input format:**
```
<schema>{CREATE TABLE ...}</schema><question>{natural language}</question><sql>{target SQL}</sql>
```
Loss mask = 1 from `<sql>` through `</sql>` + EOS. Everything before contributes 0.

**Hyperparameters:**

| Setting | Value |
|---|---|
| Batch size | 32 |
| Total steps | 3,000 |
| Warmup steps | 100 |
| LR schedule | Linear warmup → cosine decay |
| Peak LR | 1e-5 |
| Min LR | 1e-6 |
| Weight decay | 0.01 |
| Grad clip | 1.0 |
| Dropout | 0.0 |

**Results:**

| Step | Val loss |
|---|---|
| 250 | 0.2449 |
| 500 | 0.1664 |
| 750 | 0.1453 |
| 1,000 | 0.1398 |
| 1,500 | 0.1370 |
| 2,000 | 0.1358 |
| 2,250 | **0.1355** ← best |
| 2,500 | 0.1358 |
| 3,000 (final) | 0.1386 |

Best checkpoint: **`ft_step_02500`** (val loss 0.1358). Used for all evaluation below.

---

### Evaluation — gretelai/synthetic_text_to_sql test split

**Why gretelai:** Spider and WikiSQL loading scripts are deprecated on HuggingFace. gretelai has a clean `test` split never seen during training (`train` only was used in fine-tuning). Schema format (`sql_context`, `sql_prompt`, `sql`) matches the training prompt format exactly.

**Evaluation setup:**
- Split: `test` (5,851 examples)
- Decoding: greedy (temperature=0, top_p=1.0)
- Max new tokens: 150
- Execution accuracy: in-memory SQLite — compare `frozenset` of result rows between predicted and gold SQL

**Results:**

| Metric | Count | % |
|---|---|---|
| Exact Match (EM) | 1,204 / 5,851 | 20.6% |
| Execution Accuracy (EX) | 2,496 / 5,851 | 42.7% |
| Exec errors (invalid SQL) | 1,740 / 5,851 | 29.7% |
| Wrong result (valid SQL, wrong rows) | 1,615 / 5,851 | 27.6% |

When the model generates syntactically valid SQL (70.3% of examples), it produces the correct result set **60.7%** of the time.

**EX accuracy by sequence length:**

| Sequence length | Examples | EX accuracy |
|---|---|---|
| 0–128 tokens | 1,596 | 65.9% |
| 128–256 tokens | 3,488 | 38.7% |
| 256–384 tokens | 694 | 13.3% |
| 384–512 tokens | 70 | 2.9% |
| > 512 tokens | 3 | 0.0% |

Median sequence length: 163 tokens. Median schema length: 99 tokens. Only 3 examples exceed the 512-token context limit — context truncation is not a factor.

---

### Failure Analysis

**EM wrong / EX correct (1,362 examples — 23.3% of total):**
These are not real failures. The model generates semantically correct SQL that differs from the gold string due to:
- Missing column aliases (`AVG(x)` vs `AVG(x) AS avg_x`)
- `JOIN` vs `INNER JOIN` (identical semantics)

**EX wrong — identified failure patterns:**

| Pattern | Example | Root cause |
|---|---|---|
| Wrong column name | `donation_year` instead of `year` | Schema grounding — model guesses instead of reading schema |
| Wrong aggregation | `COUNT(*)` instead of `SUM(amount)` | Semantic misread of question |
| Missing WHERE clause | Drops `revenue_source` filter in multi-condition query | Multi-condition attention failure |
| Complex query breakdown | Wrong JOIN logic in subqueries, window functions | Query complexity beyond training coverage |
| Dynamic date expressions | Hardcodes `'2022-03-01'` instead of `DATEADD(month,-1,GETDATE())` | No exposure to dynamic SQL patterns |

**Primary bottleneck:** Schema grounding. The model frequently invents plausible-sounding column names rather than reading them from the provided `<schema>` block. This accounts for a significant fraction of EX failures and worsens with longer/more complex schemas.

**Secondary bottleneck:** Query complexity. EX accuracy drops from 66% on simple queries to under 3% on the 384–512 token range, which corresponds to multi-table schemas with complex SQL (nested subqueries, window functions, multi-condition filters). This is a training data coverage problem, not a context length problem.

Output truncation is not a factor: only 19 examples (0.3%) have gold SQL longer than 150 tokens.

---

### What to improve in v2

| Change | Expected impact |
|---|---|
| BIRD weight: 3× → 5× | More complex multi-table schemas; directly addresses schema grounding and complexity failures |
| Add dropout 0.1 during fine-tuning | Reduce exec error rate (29.7% invalid SQL) |
| gretelai weight: 0.5× → 0.25× | Already represented in test set; diminishing returns |
| Keep NSText2SQL at 2× | Good JOIN coverage |
| Longer fine-tuning run (5,000 steps) | Val loss was still improving at step 2,250 |

Primary target: push the 128–256 token bucket (3,488 examples, 38.7% EX) — it's the most populated range and most tractable with better training data.

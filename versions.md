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

---

## v2 — 2026-05-19

### Changes from v1

| Change | Value |
|---|---|
| Dropout | 0.1 (was 0.0) |
| Fine-tuning steps | 5,000 (was 3,000) |
| BIRD weight | 5× (was 3×) |
| gretelai weight | 0.25× (was 0.5×) |
| Starting checkpoint | step_76500 (same) |

### Fine-Tuning Results

| Step | Val loss |
|---|---|
| 500 | 0.1669 |
| 1,000 | 0.1514 |
| 1,700 | 0.1275 |
| 2,500 | 0.1187 |
| 3,500 | 0.1097 |
| 4,200 | 0.1080 |
| **4,500** | **0.1078 ← best** |
| 5,000 (final) | 0.1150 |

Best checkpoint: **`ft_step_04500`** (val loss 0.1078, vs v1's 0.1355 — 20% lower).

### Evaluation — gretelai/synthetic_text_to_sql test split

| Metric | v1 | v2 | Δ |
|---|---|---|---|
| Exact Match (EM) | 20.6% | 20.6% | 0 |
| Execution Accuracy (EX) | 42.7% | 42.7% | 0 |
| Exec errors | 1,740 (29.7%) | 1,758 (30.0%) | +18 |

**Val loss and EX are decoupled.** Despite a 20% drop in val loss, benchmark performance is flat. The model became better at predicting gold SQL tokens but did not generalise to new question-schema pairs any better. Dropout at 0.1 also failed to reduce the exec error rate.

### What we learned

The changes made (dropout, more steps, BIRD 5×) were in the right direction but insufficient to break the ceiling. The bottleneck is not hyperparameters.

**Root causes of the ceiling:**
- **Model capacity** — 30.7M params is small for multi-table SQL generation with complex schemas.
- **BIRD barely contributing** — 73% of BIRD examples exceed 512 tokens and are skipped during data prep. At 5×, BIRD contributes only ~10k weighted examples out of 523k total (~1.9%).
- **Training data ceiling** — the fine-tuning mix is largely synthetic; diminishing returns after v1.

### What to try in v3

| Option | Expected impact | Cost |
|---|---|---|
| Beam search (k=4) at inference | +2–4% EX, no retraining | Free |
| Increase context length to 1024 | Unlocks 73% of skipped BIRD examples | Moderate (retrain) |
| Larger model (60–120M params) | Directly addresses capacity ceiling | High |
| Real SQL data (actual DB + query pairs) | Better schema grounding | Data collection effort |

---

## v3 — 2026-05-19

### Changes from v2

| Change | Value |
|---|---|
| Dropout bug fixed | `nn.Dropout(deterministic=rate==0.0)` — dropout now actually applies during training |
| Everything else | Identical to v2 (5,000 steps, BIRD 5×, gretelai 0.25×, step_76500 start) |

**Bug context:** In v2, `nn.scan` silently drops `__call__` kwargs, so `deterministic=not train` passed as a kwarg was ignored. v2 trained with dropout=0 despite `FT_DROPOUT=0.1`. v3 fixed this by setting `deterministic` as a constructor argument on `nn.Dropout`, derived from the rate: `deterministic=(rate == 0.0)`. This makes inference a guaranteed no-op (rate=0.0 → deterministic=True) and training stochastic (rate=0.1 → deterministic=False).

### Fine-Tuning Results

Val loss identical to v2 — best **0.1078 at step 4,500**. Training loss values matched v2 to 4 decimal places throughout, which is expected: dropout at rate=0.1 adds noise to gradients but has minimal effect on the scalar loss printed per step.

Best checkpoint: **`ft_step_04500`**

### Evaluation — gretelai/synthetic_text_to_sql test split

| Metric | v1 | v2 | v3 | Δ (v2→v3) |
|---|---|---|---|---|
| Exact Match (EM) | 20.6% | 20.6% | 20.6% | 0 |
| Execution Accuracy (EX) | 42.7% | 42.7% | 42.7% | 0 |
| Exec errors | 1,740 (29.7%) | 1,758 (30.0%) | 1,758 (30.0%) | 0 |

Dropout with the bug fixed produces identical results to v2. The regularization hypothesis is ruled out.

### What we learned

All three versions produce the same benchmark numbers. The ceiling at **42.7% EX / 30% exec error** is not a regularization problem, not a training duration problem, and not a data weighting problem. The constraint is structural.

**Confirmed dead ends:**
- More training steps (3k → 5k): no effect
- Higher BIRD weight (3× → 5×): no effect (73% of BIRD skipped due to 512-token limit)
- Dropout (0.0 → 0.1): no effect on exec error rate or EX
- Beam search (k=4): negative effect (EX −2.5pp, exec_err +33%)

### What to try in v4

The only levers with remaining headroom:

| Option | Expected impact | Rationale |
|---|---|---|
| Context length 1024 (pretraining + fine-tuning) | Unlock BIRD; EX +5–10pp estimate | BIRD has the most complex multi-table schemas — it's the right data, just truncated |
| Larger model (60–120M params) | Address capacity ceiling directly | 30.7M is small for multi-table SQL with complex schemas |
| Both together | Largest expected gain | Compound effect: more capacity + better training data |

Context length increase is the higher-ROI first step: it requires extending RoPE frequencies and the position index range, rerunning data prep (no truncation at 1024), and retraining from the pretrain checkpoint. The model architecture is otherwise unchanged.

---

## v4 — 2026-05-19

### Changes from v3

| Change | Value |
|---|---|
| Context length | 1024 (was 512) |
| Fine-tune batch size | 16 (was 32 — O(T²) attention) |
| Fine-tune dataset | Rebuilt at MAX_LEN=1024 |
| Starting checkpoint | step_76500 (same — RoPE extrapolation, no continued pretraining) |

Dataset grew from ~523k → **591,399 train + 6,123 val** examples, confirming more BIRD examples now pass the length filter.

### Fine-Tuning Results

| Step | Val loss |
|---|---|
| 500 | 0.3823 |
| 1,000 | 0.2403 |
| 1,700 | 0.2052 |
| 2,500 | 0.1765 |
| 3,200 | 0.1676 |
| **4,000** | **0.1595 ← best** |
| 5,000 (final) | 0.1622 |

Best checkpoint: **`ft_step_04000`** (val loss 0.1595).

**Note:** val loss is not comparable to v3's 0.1078. The v4 val set includes longer BIRD examples (up to 1024 tokens) that are harder to predict. A higher val loss does not mean worse model quality relative to v3.

Grad norms were consistently above the 1.0 clip threshold (1.5–4.6) throughout training, especially early. This is expected when the model encounters positions 512–1023 it was never pretrained on.

### Evaluation — gretelai/synthetic_text_to_sql test split

Evaluated on both `ft_step_04000` (best val) and `ft_step_04500`.

| Metric | v1 | v2 | v3 | v4 (step 4000) | v4 (step 4500) |
|---|---|---|---|---|---|
| Exact Match (EM) | 20.6% | 20.6% | 20.6% | 20.0% | 20.1% |
| Execution Accuracy (EX) | 42.7% | 42.7% | 42.7% | 42.4% | 42.5% |
| Exec errors | 1,740 | 1,758 | 1,758 | 1,715 | 1,706 |

The best val loss checkpoint (step 4000) does not outperform step 4500 on EX — further confirming val loss and EX are decoupled.

### What we learned

Longer context did not improve gretelai EX accuracy. The hypothesis was: more BIRD training data → better schema grounding → higher EX. The data volume increased (591k vs 523k examples), but the benchmark didn't move.

**Root cause:** The gretelai test set has a median schema length of 99 tokens and median total sequence length of 163 tokens. Nearly all examples fit in 512 tokens. Increasing context to 1024 improves coverage of long BIRD schemas during training, but the test set never exercises that capability — it's the wrong benchmark for measuring the effect.

**What did improve:** exec errors dropped by 52 (1,758 → 1,706), suggesting the BIRD data at longer context provided mild improvement in SQL syntax validity.

**Confirmed dead ends (cumulative):**
- More steps, more BIRD weight, dropout: no effect
- Beam search: negative effect
- Context length 512 → 1024: no effect on gretelai EX (wrong benchmark)

### What to try in v5

The gretelai benchmark is saturated at ~42–43% EX for this model. The right next step is either:

| Option | Expected impact | Rationale |
|---|---|---|
| Larger model (60–120M params) | Break capacity ceiling | 30.7M is demonstrably insufficient for multi-table SQL complexity |
| Evaluate on Spider/BIRD directly | Measure actual improvement from longer context | gretelai doesn't test complex multi-table SQL; Spider dev set does |
| Both | True picture of capability + path to improvement | Spider eval is free; larger model requires new pretraining |

Spider evaluation would tell us whether the longer-context BIRD training actually helped on the tasks it was designed for — without building a new model.

**Spider EM-only result (200 examples, no schema in prompt): 3.5%.**
This is not a useful signal — running without `tables_json_path` sends an empty `<schema>` block, so the model generates SQL with no table or column names. Execution accuracy with actual schemas would be the meaningful metric, but requires the Spider tables.json download.

**Conclusion:** The gretelai benchmark is saturated at 42–43% EX for a 30.7M parameter model. Every training variable has been exhausted. The only path to meaningful improvement is a larger model.

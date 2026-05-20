# sql-lm: Building a Text-to-SQL Model from Scratch

**A 30.7M parameter transformer trained end-to-end in JAX/Flax — from raw corpus to SQL generation.**

---

## What I Built

A specialist language model that takes a database schema and a natural language question and generates the SQL query to answer it.

```
Input:  CREATE TABLE employees (id INT, name TEXT, salary FLOAT, dept TEXT)
        "What is the average salary by department?"

Output: SELECT dept, AVG(salary) FROM employees GROUP BY dept
```

Everything was built from scratch: the model architecture, the training data pipeline, the tokenizer, the training loop, and the evaluation framework. No pretrained base models were used.

---

## Model Architecture

**30.7M parameter decoder-only transformer** — a deep-narrow design based on MobileLLM research showing that at sub-100M scale, deep-narrow outperforms wide-shallow.

| Component | Value |
|---|---|
| Parameters | 30.7M |
| Layers | 16 |
| Hidden dimension | 384 |
| Attention heads | 6 |
| Context length | 1,024 tokens (final) |
| Positional encoding | RoPE (base 10,000) |
| Normalisation | RMSNorm (pre-norm) |
| MLP | SwiGLU |
| Weight tying | Yes (embedding = LM head) |
| Precision | bfloat16 |
| Framework | JAX + Flax |

Key techniques: rotary positional embeddings (RoPE), SwiGLU activations, RMSNorm, gradient checkpointing via `nn.remat`, layer scanning via `nn.scan` for memory efficiency.

---

## Training Pipeline

### Stage 1 — Pretraining

**Goal:** Learn SQL syntax, database concepts, and code structure from a large unlabelled corpus.

- **Data:** 2.51B tokens across 11 sources — Spider, BIRD, GitHub SQL files, sql_create_context, StackExchange, general code
- **Objective:** Standard next-token prediction
- **Hardware:** TPU v5e-1 (Google Colab)
- **Duration:** 76,708 steps over 2 epochs

| Hyperparameter | Value |
|---|---|
| Batch size | 128 sequences (65,536 tokens/step) |
| Peak LR | 5e-4 |
| LR schedule | WSD (warmup → stable → cosine decay) |
| Weight decay | 0.1 |
| Grad clip | 1.0 |

**Result:** Val loss 1.5618 at step 76,500 — used as the fine-tuning starting point.

---

### Stage 2 — Instruction Fine-Tuning

**Goal:** Teach the model to follow the schema + question → SQL format.

**Input format:**
```
<schema>CREATE TABLE ...</schema><question>natural language</question><sql>TARGET SQL</sql>
```

**Key design choice:** Masked cross-entropy — only tokens inside `<sql>...</sql>` contribute to the loss. The model learns to generate SQL, not to memorise the prompt.

**Training data mix (544k–591k examples depending on version):**

| Source | Weight | Why |
|---|---|---|
| BIRD (xu3kev/BIRD-SQL-data-train) | 5× | Most complex multi-table schemas |
| NSText2SQL (NumbersStation) | 2× | Good JOIN coverage |
| sql-create-context (b-mc2) | 1× | Clean schema-question-SQL triples |
| gretelai/synthetic_text_to_sql | 0.25× | Test split held out as benchmark |

---

## Results

**Benchmark:** gretelai/synthetic_text_to_sql test split — 5,851 examples, never seen during training.

**Metrics:**
- **Exact Match (EM):** string comparison after normalisation (lowercase, collapsed whitespace)
- **Execution Accuracy (EX):** run both predicted and gold SQL against an in-memory SQLite database, compare result sets as frozensets

| Metric | Result |
|---|---|
| Exact Match (EM) | 20.6% |
| Execution Accuracy (EX) | **42.7%** |
| Syntactically valid SQL | 70.3% |
| EX when SQL is valid | **60.7%** |

**EX by query complexity (sequence length):**

| Length bucket | Examples | EX |
|---|---|---|
| 0–128 tokens | 1,596 | 65.9% |
| 128–256 tokens | 3,488 | 38.7% |
| 256–384 tokens | 694 | 13.3% |
| 384–512 tokens | 70 | 2.9% |

Simple queries: strong. Complex multi-table queries: weak. The pattern is clear.

---

## The Four Fine-Tuning Versions

Four versions were trained and evaluated, testing every reasonable hyperparameter change.

| Version | Change | EX | Exec errors | Val loss |
|---|---|---|---|---|
| v1 | Baseline | 42.7% | 1,740 | 0.1355 |
| v2 | Dropout 0.1, 5k steps, BIRD 5× | 42.7% | 1,758 | 0.1078 |
| v3 | Dropout bug fixed | 42.7% | 1,758 | 0.1078 |
| v4 | Context 1,024, dataset rebuilt | 42.5% | 1,706 | 0.1595* |

*v4 val loss not comparable to v1–v3 — the v4 val set includes harder long BIRD examples.

**The benchmark never moved.** Every version produced 42–43% EX regardless of hyperparameters.

---

## What I Learned

### 1. Val loss and benchmark accuracy are completely decoupled

v2 achieved a 20% lower val loss than v1 (0.1078 vs 0.1355) with zero change in EX. The model became better at predicting gold SQL tokens but did not generalise to new examples. This is a well-known phenomenon in generation tasks and was confirmed rigorously here across four versions.

### 2. A subtle Flax bug caused dropout to silently not apply for two versions

`nn.scan` in Flax scans a module across layers. A natural-looking call like this:

```python
x, _ = ScanBlock(x, cos, sin, mask, deterministic=not train)
```

looks correct — but `nn.scan` silently drops `__call__` kwargs. The `deterministic` argument was ignored, and Flax printed a warning that was easy to miss. v2 and v3 were effectively trained with `dropout=0.0` despite setting `FT_DROPOUT=0.1`.

**Fix:** Pass `deterministic` as a constructor argument derived from the dropout rate, not as a call-time kwarg:

```python
nn.Dropout(rate=self.dropout_rate, deterministic=(self.dropout_rate == 0.0))
```

This is the correct pattern for `nn.scan` + dropout in Flax.

### 3. Beam search made SQL generation worse

Beam search is standard for machine translation, but for SQL it backfires. Exploring lower-probability token paths generates more syntactically invalid SQL. At beam size k=4:
- Greedy: 42.7% EX, 1,740 exec errors
- Beam search: 40.2% EX, 1,820 exec errors

Beam search is designed to find globally better sequences. For SQL, the grammar constraints are strict — diverging from the highest-probability path at early tokens tends to produce incomplete or invalid structure downstream.

### 4. BIRD was barely contributing — 73% of examples were silently skipped

BIRD is the highest-quality multi-table SQL dataset, so it was upsampled to 5× weight. But the data preparation pipeline silently skipped examples exceeding the 512-token context window. At 5× weight, BIRD contributed only ~10k effective examples out of 523k total (~1.9%).

Extending context to 1,024 tokens (v4) fixed this — the dataset grew from 523k to 591k examples. However, the gretelai benchmark wasn't sensitive enough to measure the difference (most gretelai examples are short). The right test would be BIRD or Spider dev set evaluation.

### 5. The ceiling is architectural, not a hyperparameter problem

After confirming that steps, data weights, dropout, context length, and decoding strategy all hit diminishing returns after v1, the conclusion is clear: 30.7M parameters is insufficient for the task. The primary failure mode is **schema grounding** — the model generates plausible-sounding but incorrect column names instead of reading them from the schema. This requires more capacity.

---

## Failure Analysis

**EM wrong / EX correct (23.3% of examples):** Not real failures. The model generates semantically equivalent SQL that differs in surface form — `JOIN` vs `INNER JOIN`, missing aliases like `AVG(x) AS avg_x`. These are indistinguishable in execution but fail exact match.

**Common EX failure patterns:**

| Pattern | Example | Root cause |
|---|---|---|
| Wrong column name | `donation_year` instead of `year` | Schema grounding — model invents names |
| Wrong aggregation | `COUNT(*)` instead of `SUM(amount)` | Semantic misread |
| Missing WHERE clause | Drops a filter condition | Multi-condition attention failure |
| Wrong JOIN logic | Incorrect ON clause in subqueries | Query complexity |
| Dynamic dates | Hardcodes `'2022-03-01'` instead of `DATEADD(...)` | No training exposure |

---

## What Would Actually Move the Needle

| Option | Expected impact | Why |
|---|---|---|
| Larger model (60–120M params) | +10–15pp EX estimate | Directly addresses schema grounding and complexity ceiling |
| Fine-tune a pretrained base (TinyLlama, GPT-2-medium) | +15–20pp EX estimate | Leverage pre-existing language understanding, not starting from scratch |
| Real SQL data (actual DB + query pairs) | Better grounding | Synthetic data has a quality ceiling |

The 30.7M parameter from-scratch model has been fully explored. The next meaningful experiment is a larger architecture.

---

## Tech Stack

| Component | Tool |
|---|---|
| Model | JAX + Flax (Linen) |
| Optimiser | Optax (AdamW + cosine schedule) |
| Checkpointing | Orbax |
| Tokenizer | HuggingFace tokenizers (BPE, 12,288 vocab) |
| Training data | HuggingFace datasets (streaming) |
| Hardware | Google Colab TPU v5e-1 |
| Evaluation | In-memory SQLite (Python stdlib) |

---

## Key Numbers at a Glance

- **30.7M** parameters trained from scratch
- **2.51B** tokens in the pretraining corpus
- **591k** instruction fine-tuning examples
- **42.7%** execution accuracy on a held-out benchmark
- **60.7%** EX when the generated SQL is syntactically valid
- **4** fine-tuning versions — same benchmark score across all of them
- **1** architectural conclusion: need a bigger model

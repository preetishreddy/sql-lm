# Pretraining Corpus — Design, Implementation & Results

This document covers the full pretraining data pipeline for the sql-lm 31.5M-parameter model:
why each source was chosen, how the pipeline works, what was built, and the exact numbers
from the completed run.

---

## 1. Goals

The model needs to be able to:
1. Read and write SQL — DDL (CREATE TABLE), DML (SELECT, INSERT, UPDATE), window functions, CTEs
2. Understand natural-language questions and map them to SQL (text-to-SQL)
3. Handle schema context — column names, types, foreign keys, constraints
4. Generalize beyond SQL into adjacent structured text (Markdown, Python data scripts, Ruby migrations)
5. Follow instructions in English prose (the "question" half of text-to-SQL pairs)

These goals drive the corpus composition: SQL-heavy but not SQL-only, with enough general
English to handle the instruction side of text-to-SQL without catastrophic forgetting.

---

## 2. Tokenizer (prerequisite)

| Property        | Value                                |
|-----------------|--------------------------------------|
| Algorithm       | BPE (Byte-Pair Encoding)             |
| Vocabulary size | 12,288 tokens                        |
| Special tokens  | IDs 0–14 (reserved at low indices)   |
| Training data   | SQL + code sampled from BigCode/Stack|

Special token assignments:

| ID | Token          | Use                              |
|----|----------------|----------------------------------|
| 0  | `<bos>`        | Beginning of sequence            |
| 1  | `<eos>`        | End of sequence                  |
| 2  | `<pad>`        | Padding (pretraining: never used)|
| 3  | `<unk>`        | Unknown token                    |
| 4  | `<schema>`     | Finetuning delimiter             |
| 5  | `<question>`   | Finetuning delimiter             |
| 6  | `<sql>`        | Finetuning delimiter             |
| 7  | `</schema>`    | Finetuning delimiter             |
| 8  | `</question>`  | Finetuning delimiter             |
| 9  | `</sql>`       | Finetuning delimiter             |
| 10–14 | `<reserved_N>` | Reserved for future use       |

Finetuning delimiters (IDs 4–9) appear at most as noise in the pretraining corpus
(literal `<schema>` strings in code comments/docs). Counts are <1 per million tokens.

---

## 3. Corpus Design

### 3.1 Token Budget Summary

Total pretraining corpus: **2,513,584,640 train tokens** (~2.51B).

| Source               | Category              | Token Budget | Actual Tokens  | % of Corpus | Sequences (train) |
|----------------------|-----------------------|--------------|----------------|-------------|-------------------|
| `stack_sql`          | SQL code (BigCode)    | 1,100M       | 1,089,357,824  | 43.3%       | 2,127,652         |
| `fineweb`            | English prose (web)   | 600M         |   594,905,600  | 23.7%       | 1,161,925         |
| `stack_python`       | Python code (BigCode) | 280M         |   277,317,632  | 11.0%       |   541,636         |
| `sqale`              | Text-to-SQL pairs     | 200M         |   198,047,744  |  7.9%       |   386,812         |
| `gretelai`           | Text-to-SQL + schema  | 100M         |    99,878,400  |  4.0%       |   195,075         |
| `stack_markdown`     | Markdown (BigCode)    |  80M         |    79,247,360  |  3.2%       |   154,780         |
| `nstext2sql`         | Text-to-SQL pairs     |  50M         |    49,695,744  |  2.0%       |    97,062         |
| `stack_ruby`         | Ruby migrations       |  50M         |    49,582,592  |  2.0%       |    96,841         |
| `bird`               | Hard text-to-SQL (BIRD)|  30M         |    29,744,640  |  1.2%       |    58,095         |
| `sql_create_context` | Text-to-SQL pairs     |  30M         |    30,559,232  |  1.2%       |    59,686         |
| `oasst`              | English Q&A (prompter)| 15M          |    15,247,872  |  0.6%       |    29,781         |
| **TOTAL**            |                       | **2,535M**   | **2,513,584,640** | **100%** | **4,948,564**  |

Validation split: 1% per source (seed=42, deterministic). Total val sequences: 48,582.

### 3.2 Why This Mix

**SQL-first (43%)**: The model's primary job is SQL. `stack_sql` is real-world SQL from
GitHub — DDL schemas, stored procedures, migrations, complex queries. This is the
hardest-to-fabricate signal: actual production code across many dialects and domains.

**English prose (24%)**: FineWeb-Edu supplies high-quality educational English. This keeps
the model from degenerating into a pure code model that can't parse questions or generate
coherent explanations. Without this, text-to-SQL performance collapses because the model
can't understand the instruction side.

**Python (11%)**: Python data scripts (pandas, SQLAlchemy, psycopg2) bridge natural language
thinking and SQL. They share the identifier-heavy structure of SQL while looking more like
English. Also prevents the model from becoming dialect-locked into SQL syntax.

**Text-to-SQL pairs (sqale + gretelai + nstext2sql + sql_create_context ≈ 15%)**: Supervised
examples that directly train the question-to-SQL mapping. Exposed during pretraining so the
model sees the task structure before finetuning.

**Ruby migrations (2%)**: Ruby on Rails schema files (`schema.rb`, `migrate/*.rb`) are a
qualitatively different view of SQL DDL — table definitions expressed in a fluent DSL.
This teaches the model that schema intent can be expressed multiple ways.

**BIRD (1.2%)**: BIRD is a hard text-to-SQL benchmark with complex multi-table queries,
nested aggregations, and domain-specific schemas. Including its training split exposes the
model to difficulty distributions beyond Spider/WikiSQL.

**OpenAssistant (0.6%)**: A tiny dose of English Q&A from human prompters. Adds
conversational phrasing ("Can you...", "What is...") that mirrors real text-to-SQL inputs.
Budget is small (15M) because the English prompter-only subset of oasst2 has only ~622K
unique tokens — repeating it more than ~24 times would cause memorization without gain.

### 3.3 Sources Excluded and Why

| Source            | Reason excluded                                                              |
|-------------------|------------------------------------------------------------------------------|
| Spider            | Largely covered by `sqale` and `nstext2sql` which include Spider subsets     |
| WikiSQL           | Same coverage issue; very simple single-table queries add little signal      |
| CoSQL             | Dialogue SQL — conversational turn structure not aligned with this model's use|
| GitHub raw SQL    | No filtering — too much noise, auto-generated migration garbage              |
| FineWeb (full)    | 600M tokens of FineWeb-Edu is sufficient; raw FineWeb is lower quality       |

---

## 4. Pipeline Implementation

### 4.1 File Layout

```
sql-lm/
  scripts/
    corpus_sources.py          # SOURCES dict: 11 source configs (loader, filter, budget)
    build_corpus_helpers.py    # pack_sequences, train_val_split, sha256_file, ...
    build_corpus.py            # CLI driver: --source <name>|all [--force]
    write_manifest.py          # Aggregate all sources into manifest.json
    verify_pipeline.py         # 10-check verification suite
    preprocess.py              # NFC normalization (single source of truth)
  data/
    tokenized/
      {source}_train.npy       # shape [N, 512], dtype int16
      {source}_val.npy         # shape [V, 512], dtype int16
      manifest.json            # token counts, sha256s, proportions
  tokenizer/
    tokenizer.json             # 12k-vocab BPE (already trained)
```

### 4.2 Sequence Format

Each document is encoded as:

```
[BOS=0] [token_1] [token_2] ... [token_N] [EOS=1]
```

Documents are concatenated into a flat token stream, then reshaped into `[N, 512]` int16
arrays. There is **no padding** — the trailing partial sequence is discarded. This is the
standard "packed pretraining" format used by GPT-style models.

At sequence boundaries within the packed array, BOS appears at the start of a new document
and EOS appears at the end of the previous one:

```
... [token_K] [EOS] [BOS] [token_1] ...
                ^doc boundary^
```

The causal attention mask sees across document boundaries — this is intentional for
pretraining efficiency (no wasted compute on padding). During finetuning, each sample is
treated as a single sequence within its own context window.

### 4.3 Key Implementation Decisions

**`encode_batch()` not a loop**: The tokenizer's `encode_batch(list_of_strings)` engages
its Rust threadpool, giving 10–15× throughput over a Python `for` loop calling `encode()`.
Documents are buffered in groups of 1,000 before each batch call.

**`array.array('h')` for accumulation**: Token IDs are accumulated into a Python
`array.array('h')` (signed int16), not a Python list. This uses ~2 bytes/token vs ~28
bytes/token for a list of Python ints, making 1.1B-token sources feasible in RAM without
chunked writes.

**Epoch cycling for small sources**: Datasets smaller than their token budget stream
through multiple epochs. The loader lambda is called fresh each epoch to get a new streaming
iterator. Sources affected: `sql_create_context` (~6 epochs), `bird` (~3 epochs),
`oasst` (~24 epochs).

**Deterministic train/val split**: After packing, rows are shuffled with
`np.random.default_rng(42).permutation(N)` and the first 1% becomes validation.
The seed is fixed — the same split is always produced from the same data.

**Line-buffered output**: `sys.stdout.reconfigure(line_buffering=True)` is set at startup
so progress lines are written immediately when the script runs as a background subprocess.
Without this, Python's default full-buffering suppresses all output until the process exits.

### 4.4 Source Configs (`corpus_sources.py`)

Each source in `SOURCES` has:

```python
{
    "name":           str,              # slug used for filenames
    "loader":         lambda -> Dataset,# returns a fresh streaming HF dataset
    "text_extractor": lambda row -> str,# extracts text from a row
    "filter":         lambda row, text -> bool,  # quality filter
    "token_budget":   int,              # stop streaming at this many tokens
    "gated":          bool,             # whether HF_TOKEN is required
}
```

Critical field-name notes (easy to get wrong):

| Source             | Text fields                                                  |
|--------------------|--------------------------------------------------------------|
| `sqale`            | `row["query"]` — NOT `row["sql"]`                           |
| `nstext2sql`       | `row["instruction"]` + `row["output"]`                      |
| `gretelai`         | `sql_context`, `sql_prompt`, `sql`, `sql_explanation`        |
| `oasst`            | `row["text"]`, filter: `role=="prompter" AND lang=="en"`     |
| `stack_ruby`       | `row["content"]`, filter: path contains `migration`/`schema`|
| `bird`             | `schema`, `question`, `evidence`, `SQL`                      |

### 4.5 Build Commands

```powershell
# Build one source
python -m scripts.build_corpus --source sqale

# Build all sources in recommended order (non-gated first)
python -m scripts.build_corpus --source all

# Force rebuild even if output files exist
python -m scripts.build_corpus --source stack_sql --force

# Write manifest after all sources are done
python -m scripts.write_manifest

# Verify the corpus (full)
python -m scripts.verify_pipeline

# Verify without decode/plausibility check (faster)
python -m scripts.verify_pipeline --skip-plausibility

# Verify one source only
python -m scripts.verify_pipeline --source sqale
```

Gated sources (`stack_sql`, `stack_markdown`, `stack_python`, `stack_ruby`) require a
HuggingFace token with BigCode access:

```bash
HF_TOKEN="hf_..." python -u -m scripts.build_corpus --source stack_sql
```

---

## 5. Verification Results

All 10 checks pass on the completed corpus.

| Check | Description | Result |
|-------|-------------|--------|
| 1. files_exist | All 22 .npy files (11 × train/val) present and loadable | PASS |
| 2. shapes | All files are shape `[N, 512]`, dtype `int16` | PASS |
| 3. id_range | All token IDs in `[0, 12287]` | PASS |
| 4. bos_at_start | BOS (ID 0) appears in sampled sequences at doc boundaries | PASS |
| 5. eos_before_bos | EOS immediately precedes every inner BOS (doc boundaries correct) | PASS |
| 6. no_pad | PAD (ID 2) rate < 1 per million tokens (no padding used) | PASS |
| 7. no_finetune_ids | Finetuning delimiter IDs 4–9 rate < 1 per million tokens | PASS |
| 8. proportions | Each source within ±5% of target proportion | PASS |
| 9. plausibility | 10 random sequences per source decode to readable text | PASS |
| 10. checksums | sha256 of every file matches manifest | PASS |

### Notes on checks 4, 6, 7

**Check 4 (BOS at start)**: In packed sequences, BOS only lands at position 0 of a 512-token
row when a document boundary aligns exactly to a 512-token multiple — probability ≈ 1/512.
The check verifies that BOS appears *somewhere* inside sampled sequences (confirming
document boundaries exist), not that it appears at the row start.

**Checks 6–7 (PAD and finetuning IDs)**: Source code (SQL schema files, Ruby migrations,
Markdown documentation) occasionally contains literal strings like `<schema>`, `<sql>`,
`<pad>` as XML/HTML tags. The tokenizer maps these to IDs 2 and 4–9. Counts are in the
range of 1–50 per source out of hundreds of millions of tokens. The threshold is 1 per
million tokens (any rate above this would suggest a bug, not source text noise).

---

## 6. Corpus File Sizes

| File                       | Size (MB) |
|----------------------------|-----------|
| stack_sql_train.npy        | 2,178.7   |
| fineweb_train.npy          | 1,189.8   |
| stack_python_train.npy     |   554.6   |
| sqale_train.npy            |   396.1   |
| gretelai_train.npy         |   199.8   |
| nstext2sql_train.npy       |    99.4   |
| stack_ruby_train.npy       |    99.2   |
| stack_markdown_train.npy   |   158.5   |
| sql_create_context_train.npy|   61.1   |
| bird_train.npy             |    59.5   |
| oasst_train.npy            |    30.5   |
| *(+ 11 val files)*         |    47.7   |
| **TOTAL**                  | **5.08 GB** |

Each token is stored as a signed int16 (2 bytes). The theoretical minimum for 2.51B tokens
is 2 × 2.51B = 5.03 GB; the actual 5.08 GB includes val files and npy array headers.

---

## 7. Manifest

`data/tokenized/manifest.json` records the complete corpus state:

```json
{
  "vocab_size": 12288,
  "sequence_length": 512,
  "total_train_tokens": 2513584640,
  "total_sources": 11,
  "created": "2026-05-18T20:10:32.080323+00:00",
  "sources": {
    "stack_sql": {
      "train": { "tokens": 1089357824, "sequences": 2127652, "sha256": "7c8529..." },
      "val":   { "tokens": 11003392,  "sequences": 21491,   "sha256": "b30e63..." },
      "token_budget": 1100000000,
      "target_proportion": 0.433925,
      "actual_proportion": 0.433388
    },
    ...
  }
}
```

The manifest is the ground truth for training: the data loader reads it to discover files,
verify sizes, and compute sampling weights for multi-source mixing.

---

## 8. Known Limitations

**oasst unique content is small**: The English prompter subset of oasst2 has ~8,400 rows
and ~622K unique tokens. The 15M budget repeats this content ~24 times. This is acceptable
for a small style-transfer signal but oasst contributes no unique information beyond what
those 622K tokens encode. If a larger Q&A corpus (ShareGPT, LMSYS-Chat-1M) were available,
it would be a better choice.

**No deduplication across sources**: sqale, nstext2sql, and sql_create_context all draw
from Spider and WikiSQL subsets. There is deliberate overlap — exposure to the same
schema+question pairs from multiple sources acts as data augmentation — but it also means
some queries appear many times. This is intentional, not an oversight.

**FineWeb is English-only**: The model will have weak multilingual SQL ability. This matches
the target use case (English text-to-SQL) and avoids diluting the token budget with content
the model won't need.

**Stack sources use file-level chunking**: BigCode Stack files can be very long
(10k–100k tokens). Within the packed sequences, most 512-token windows are mid-file.
This is fine for pretraining but means the model rarely sees a complete SQL file from
start to finish in a single context window.

---

## 9. Next Steps

With the corpus complete, the next work item is the training loop:

1. **`CorpusLoader`** — a PyTorch `Dataset` that reads the .npy files from the manifest,
   optionally shuffles sequences across sources using the target proportions as sampling
   weights, and yields batches of `[batch_size, 512]` int64 tensors (cast from int16).

2. **Model** — 31.5M parameter GPT-style transformer. Architecture defined separately.

3. **Training** — Colab TPU v5e-1, ~10,300 steps at batch size 256 × sequence length 512
   = ~131K tokens/step to consume the full 2.51B train tokens in one epoch.

4. **Finetuning** — After pretraining, finetune on structured `<schema>...<question>...<sql>...`
   examples using the delimiter tokens (IDs 4–9) that were reserved but not used in pretraining.

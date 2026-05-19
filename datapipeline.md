# Data Pipeline — Full Context & Implementation Plan

This file exists so Claude Code has complete context on every decision made
for the pretraining data pipeline. Read the full document before writing any
code. Many decisions here have non-obvious reasons that affect model quality.

Depends on: `tokenizer/tokenizer.json` (vocab_size=12288, must exist first)
Feeds into: model pretraining on Colab TPU v5e-1

This document was revised after a corpus review. The previous version
budgeted 2.0B tokens with several factual bugs in the source descriptions
(wrong field names, wrong licenses) and ~80M tokens of hidden duplication
between NSText2SQL and sql-create-context. The revised plan budgets **2.7B
tokens**, fixes the bugs, dedups the overlap, increases the English share
from 16% to 22%, and adds two qualitatively-new sources (conversational
prompts, BIRD train). The "Why This Mix Changed" section near the end
explains every delta.

---

## What This Pipeline Does

Takes raw text from multiple HuggingFace datasets, tokenizes it, packs it
into fixed-length sequences, and produces the integer arrays the model trains
on. This is a one-time preprocessing job that runs locally (Ryzen 9, ~60-90
min depending on internet speed), producing ~5.4GB of output saved to disk
and backed up to Google Drive before any Colab training session begins.

```
HuggingFace streams
    → NFC normalize          (scripts/preprocess.py — same function as tokenizer)
    → filter by quality      (per-source rules, see below)
    → tokenize               (tokenizer/tokenizer.json, vocab=12288)
    → prepend <bos> (ID 0)
    → pack sequences         (512-token chunks, <eos><bos> between docs)
    → split train/val        (99/1 per source)
    → save as int16 .npy     (2 bytes/token, ~5.4GB total)
    → write manifest + sha256 checksums
    → verify output          (scripts/verify_pipeline.py)
```

---

## Hardware & Storage Context

**Local machine (where pipeline runs):**
- CPU: AMD Ryzen 9 — tokenization throughput ~2.8M tokens/sec **if you use
  `tokenizer.encode_batch()`**, which engages the Rust threadpool. A naive
  Python `for doc: tokenizer.encode(doc)` loop runs 10-15× slower
  (~200k/sec). Always batch — see Implementation Steps.
- Pure tokenization of 2.7B tokens with batched encoding: ~16-20 minutes
- Real bottleneck: HuggingFace download speed (~10-50 MB/s) AND filter yield
  on the Stack subsets (markdown + python combined stream ~15-20GB raw to
  yield ~430M filtered tokens — most files get rejected)
- Total time estimate: **60-90 minutes** end-to-end on a fast connection,
  90-150 minutes on a slow one

**Colab TPU (where training runs):**
- RAM: 47GB, Disk: 225GB — enough to store everything
- Session limit: ~1 hour on free tier
- Problem: download + tokenize + train cannot all fit in one session
- Solution: pre-tokenize locally, save to Google Drive, copy to Colab at
  session start (~3-4 min for 5.4GB), train immediately
- Training time estimate at 2.7B tokens, batch 256×512, grad_accum 2:
  ~10,300 steps. Across 2-3 Colab sessions with checkpoint resume.

**Storage layout:**
```
Local disk (temporary, ~22GB peak during download):
  data/raw/          ← streaming cache, can be deleted after tokenization
  data/tokenized/    ← final .npy files, ~5.4GB total, keep these

Google Drive (permanent backup):
  sql-lm/data/tokenized/   ← copy of the .npy files + manifest.json
  sql-lm/checkpoints/      ← training checkpoints saved here during training

Colab /content/ (per-session, wiped on disconnect):
  data/tokenized/   ← copied from Drive at session start
  checkpoints/      ← saved to Drive every 500 steps
```

---

## Why 2.7B Tokens (Not 2.0B, Not 5B)

Chinchilla-optimal for a 31.5M-param model is ~630M tokens (20 tok/param).
We deliberately over-train because:

- The model is for **inference quality**, not just training-loss-optimal —
  every extra token improves downstream eval at decreasing marginal cost.
- Modern small-model practice goes far beyond Chinchilla: SmolLM-135M used
  ~600B tokens (4,400 tok/param), TinyLlama-1.1B used ~3T (2,700 tok/param).
  Scaling that posture to our 31.5M model would suggest hundreds of B tokens,
  which is obviously impractical.
- The actual ceiling for us is **Colab session count** plus **data quality
  available**. Pushing past ~3B forces us into repetition of mediocre data,
  which hurts more than it helps.

Why these specific levels were rejected:

- **2.0B (original):** defensible but under-utilized the data we can easily
  reach. Question-understanding signal was thin (16% English).
- **2.5B:** captures most of the gain at slightly lower cost. Acceptable
  fallback if Colab time becomes a real constraint.
- **2.7B (chosen):** sweet spot — enough English to fix question-paraphrase
  weakness, enough additional SQL to cover more patterns, room for two new
  qualitative sources, while staying under 3B where Colab session count
  starts compounding.
- **3.5B+:** small additional gain, large additional cost. Tokenization
  jumps to 25+ min, training to 13k+ steps, ~3-4 Colab sessions. Not worth
  it at this model size — diminishing returns set in.
- **5B:** would force >5 epochs over sql-create-context and >3× cycle
  through gretelai, both well into memorization-risk territory. Synthetic
  data quality ceiling becomes the bottleneck, not token count.

---

## The Core Problem: No SQL Documentation Dataset Exists

The architecture guide allocates ~10% of the corpus to "SQL documentation"
(PostgreSQL, MySQL, SQLite manuals). This source does not exist on
HuggingFace. Options investigated:

**Option A — Scrape the docs directly**
- PostgreSQL 16 docs: ~8MB text; MySQL 8.0: ~15MB; SQLite: ~3MB
- Total ~26MB ≈ 6M tokens. Out of 2.7B that is 0.2% — negligible signal.
- Verdict: Not worth the scraping infrastructure.

**Option B — SivilTaram/starcoder2-documentation**
- 59,700 rows across 48 programming languages (R, Rust, JS, Erlang, Go, etc.)
- SQL-relevant content: ~1-7M tokens out of ~75-150M total
- Preview is dominated by R packages, npm modules, Rust crates
- Verdict: Misleadingly named for our purpose. Could include as general
  code-adjacent English (~75M tokens) but FineWeb-Edu is a cleaner choice
  for that role.

**Option C — The Stack Markdown (SQL-filtered) ← USED, but trimmed**
- `bigcode/the-stack-dedup` has a `markdown` language subset (millions of
  README files, tutorials, blog posts committed to GitHub)
- Filter for files containing SQL keywords (SELECT, FROM, WHERE, JOIN, etc.)
- Real yield estimate: **1-3% of markdown files** contain meaningful SQL
  content (the earlier "5-10%" was optimistic — most files that mention SQL
  do so in passing, in a code-of-conduct example or unrelated changelog)
- This is the weakest source in the corpus. Kept because *some* signal is
  better than none for the "explain SQL in English" role, but reduced from
  100M → 80M in the revised mix.

**Why this matters for model quality:**
Documentation teaches the model *why* SQL is written a certain way, not just
*what* SQL looks like. A README that says "we use GROUP BY department to
aggregate salaries because we need one row per department" is more valuable
than a raw SQL file with the same query and no context.

---

## The Data Sources — Final Mix

All sources verified to exist as described. Field names and licenses below
were checked against the HuggingFace dataset cards.

### Source 1 — The Stack SQL (1100M tokens, 41%)
```python
load_dataset("bigcode/the-stack-dedup", data_dir="data/sql",
             split="train", streaming=True)
```
- **What it is:** Raw SQL files from GitHub — queries, schemas, migrations,
  stored procedures, views
- **Why this share:** SQL syntax is the foundation. The model has 31.5M
  params and learns SQL patterns largely by memorization, not generalization
  — repeated exposure to every common pattern matters more here than for a
  7B model.
- **Why bumped from 900M → 1100M:** The Stack SQL is the cheapest source of
  additional high-quality SQL diversity. The previous 900M left coverage
  thin on DDL, stored procedures, and dialect-specific patterns
  (PL/pgSQL, T-SQL).
- **Why not 1500M+:** at some point you're just teaching the model more
  SELECT variations from CRUD apps; the marginal patterns past ~1B are
  highly repetitive.
- **Quality filter:** Skip files under 100 chars (no useful signal), files
  over 500k chars (auto-generated dumps), and files with `alphanum_fraction
  < 0.25` (mostly comments/whitespace).
- **Access:** Gated. Requires BigCode license acceptance at
  huggingface.co/datasets/bigcode/the-stack-dedup. You have already accepted
  this for tokenizer training.
- **Why The Stack v1.2 and not v2:** `bigcode/the-stack-v2-dedup` is newer
  (2024), larger, and better deduplicated, but it stores only file pointers
  and requires Software Heritage downloads — much more painful. v1.2 is the
  pragmatic choice; we accept the staleness as the cost of simpler tooling.

### Source 2 — The Stack Markdown SQL-filtered (80M tokens, 3%)
```python
load_dataset("bigcode/the-stack-dedup", data_dir="data/markdown",
             split="train", streaming=True)
# Filter: content must contain at least 2 of: SELECT, FROM, WHERE, JOIN, TABLE
```
- **What it is:** README files and documentation files from GitHub repos,
  filtered to only those that discuss SQL
- **Why include at all:** Provides the "docs" signal — explanations of what
  queries do, schema design rationale, tutorial content. Even at low share
  it complements raw SQL by teaching *intent*.
- **Why only 3% (down from 5%):** This is the weakest source in the mix.
  Real filter yield is closer to 1-3% (not 5-10%), so we'd be streaming
  enormous amounts of raw markdown for diminishing-quality output. Capped
  at 80M tokens — beyond that we're scraping the barrel.
- **Filter threshold:** Require at least 2 SQL keywords to avoid false
  positives (files that mention SQL once but are about something else)
- **Access:** Same BigCode license as Source 1

### Source 3 — SQaLe (200M tokens, 7.4%)
```python
load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train",
             streaming=True)
```
- **What it is:** 511,630 execution-validated triples of (schema, question,
  query). Generated from 135,875 real database schemas. Published Dec 2025.
- **Why bumped from 150M → 200M:** highest-quality text-to-SQL pair source
  in the mix. Execution-validation means every SQL actually runs and returns
  the right answer; real schemas mean diverse, realistic column names — not
  toy examples. Plenty of headroom remains (517k triples > 200M tokens).
- **Field schema (CORRECTED from previous doc):** the SQL field is named
  **`query`**, NOT `sql`. Triples are `{schema, question, query, ...}` plus
  metadata fields (`token_count`, `num_joins`, `num_tables`,
  `number_of_columns`). Loader code must use `row["query"]`. The previous
  doc's `row["sql"]` would crash on row 1.
- **License (CORRECTED):** **MIT**, not Apache 2.0. Both are
  pretrain-compatible so the change is informational.
- **Caveat:** Published Dec 16, 2025 — fresh data, limited community
  scrutiny so far. Treat as high-value but watch eval outputs for
  distribution-specific quirks (template-y questions, repeated schema
  patterns within the 135k schema pool).
- **Text format for pretraining:** Concatenate `schema + "\n" + question +
  "\n" + query`. Do NOT use the finetuning `<schema>/<question>/<sql>`
  delimiters during pretraining — save those for the finetuning phase.
  Plain text exposure during pretraining is enough.

### Source 4 — gretelai/synthetic_text_to_sql (100M tokens, 3.7%)
```python
load_dataset("gretelai/synthetic_text_to_sql", split="train",
             streaming=True)
```
- **What it is:** 100k train + 5.8k test synthetic text-to-SQL examples with
  a unique `sql_explanation` field — a natural language description of what
  each query does and why.
- **Why kept at 100M:** the `sql_explanation` field is uniquely valuable. No
  other dataset has natural language explanations of SQL query logic at this
  scale. Teaches the model the mapping between English descriptions and SQL
  structure.
- **Why not bumped further:** the dataset has ~50-80M tokens TOTAL (earlier
  "200M available" claim was wrong). 100M budget means cycling through it
  ~1.3-2× — already at or near the dataset ceiling. Adding more would mean
  pure repetition, not new content.
- **Text format:** `sql_context + "\n" + sql_prompt + "\n" + sql + "\n" +
  sql_explanation`. Include all four fields — the explanation is the most
  valuable part.
- **Coverage:** Explicitly includes window functions, CTEs, aggregations,
  subqueries, set operations — the hard patterns.
- **Access:** Apache 2.0, no login required

### Source 5 — NumbersStation/NSText2SQL (50M tokens, 1.9%)
```python
load_dataset("NumbersStation/NSText2SQL", split="train", streaming=True)
```
- **What it is:** 289,288 text-to-SQL pairs curated from 20+ public sources
  with schema augmentation and SQL cleaning applied. SQLite dialect.
- **Field schema (CORRECTED from previous doc):** the fields are
  **`instruction`** (a single string that already bakes the schema + question
  together) and **`output`** (the SQL). There is no separate schema/question
  split — the previous doc implied one. For pretraining just use `instruction
  + "\n" + output` as the concatenated text.
- **License (CORRECTED):** **mixed per source** (CC-BY-SA-4.0, Apache 2.0,
  MIT, BSD-3-Clause, CC-BY-4.0). All are pretrain-compatible. The
  per-source `source` field lets you filter if you want to exclude any
  specific license later.
- **Why dropped from 80M → 50M:** ~70% of NSText2SQL is derived from
  Spider + WikiSQL, and `sql-create-context` (Source 6) is *also* derived
  from Spider + WikiSQL. Keeping both at original budgets meant 160M tokens
  of essentially-duplicated Spider/WikiSQL data masquerading as "diversity."
  Cutting both to ~80M combined removes the redundancy without losing the
  curation/augmentation that distinguishes each.
- **Why not drop entirely:** the non-Spider sources within NSText2SQL
  (mimicsql, nvbench, criteria2sql, KaggleDBQA subsets) provide genuine
  domain diversity unavailable elsewhere. Worth keeping a slice.
- **Access:** No login required

### Source 6 — b-mc2/sql-create-context (30M tokens, 1.1%)
```python
load_dataset("b-mc2/sql-create-context", split="train", streaming=True)
```
- **What it is:** 78,577 curated (schema, question, SQL) triples derived
  from Spider and WikiSQL with schema augmentation.
- **Why kept:** the schema augmentation (inferred column types, added
  constraints) makes schemas more realistic than raw Spider. Well-established
  high-quality dataset.
- **Why dropped from 80M → 30M:** the dataset is small (~5M unique tokens).
  80M budget meant ~16 epochs over the same data — well past the typical
  safe ceiling of 4-8 epochs and into memorization-risk territory.
  30M budget = ~6 epochs, which keeps the high-quality signal while staying
  within healthy repetition bounds. Combined with the NSText2SQL trim, this
  removes the Spider/WikiSQL duplication problem.
- **Access:** Apache 2.0, no login required

### Source 7 — The Stack Python SQL-filtered (280M tokens, 10.4%)
```python
load_dataset("bigcode/the-stack-dedup", data_dir="data/python",
             split="train", streaming=True)
# Filter: content must contain SELECT and at least one of FROM, WHERE, JOIN
```
- **What it is:** Python files from GitHub that contain SQL queries — Django
  ORM raw queries, SQLAlchemy, psycopg2 scripts, data analysis notebooks
  with pandas + SQL
- **Why bumped from 200M → 280M:** SQL-in-programming-context is one of the
  largest gaps in the original mix. Real-world SQL almost always lives
  inside application code, not as standalone .sql files. More of this teaches
  the model schema relationships as a programmer would model them — variable
  names, function calls, ORM patterns that map to SQL concepts.
- **Why not 500M+:** filter yield on python is ~5-10%. Pushing this higher
  means streaming much more raw Python data for diminishing returns. 280M is
  the practical sweet spot.
- **Filter:** Require both SELECT and at least one other SQL keyword to
  avoid Python files that only mention SQL once incidentally.
- **Access:** Same BigCode license

### Source 8 — The Stack Ruby migrations (50M tokens, 1.9%)
```python
load_dataset("bigcode/the-stack-dedup", data_dir="data/ruby",
             split="train", streaming=True)
# Filter: path must contain 'migration' or 'schema'
```
- **What it is:** Rails migration files — Ruby DSL for database schema
  changes. `add_column :users, :email, :string` is semantically equivalent
  to `ALTER TABLE users ADD COLUMN email VARCHAR`.
- **Why include:** Schema pattern diversity. Rails migrations are one of the
  most common ways schemas are defined in real projects. Seeing both the
  Ruby DSL form and SQL form of the same concept helps the model build a
  richer internal representation of schema structure.
- **Why kept at 50M (not bumped):** Schema migrations are highly repetitive
  — you don't need hundreds of millions of tokens of them. 50M covers the
  common patterns; more would just teach the model that
  `add_column :users, :email, :string` is a common phrase.
- **Access:** Same BigCode license

### Source 9 — FineWeb-Edu (600M tokens, 22.2%)
```python
load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
             split="train", streaming=True)
```
- **What it is:** High-quality English web text filtered for educational
  content. 10B token sample of the full FineWeb-Edu dataset.
- **Why bumped from 340M → 600M (16% → 22%):** the single biggest change
  in this revision. Text-to-SQL is the inverse of code generation — the
  *input* is English, so the model needs more English exposure than a pure
  code model. CodeLLaMA used ~8% NL, StarCoder ~10%, but text-to-SQL papers
  typically land at 20-25%. At a 31.5M model size, English understanding is
  largely *memorized* not *generalized* — the extra 260M directly improves
  paraphrase robustness and reasoning over multi-clause questions.
- **Why not 30%+:** above ~25% English the model starts losing SQL fluency
  on a 2.7B budget. At 30% you'd be at parity with the entire Stack SQL
  slice after Source 10's deductions — syntax errors creep in.
- **Why FineWeb-Edu specifically and not C4/CommonCrawl:** FineWeb-Edu is
  filtered for educational quality. The text is dense with explanations,
  causal reasoning, definitions — closer to the cognitive task of "parse a
  question and produce a structured answer" than to generic web noise.
- **One caveat:** FineWeb-Edu skews expository (Wikipedia-like prose). It
  under-represents casual user phrasings like "show me sales last month."
  That's exactly why Source 10 exists.
- **Access:** No login required

### Source 10 — OpenAssistant prompter turns, English (150M tokens, 5.6%)
```python
load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
# Filter: role == "prompter" AND lang == "en"
```
- **What it is:** Real human-written prompts from the OpenAssistant project.
  We use only the prompter (user) turns and only English. Throws away the
  assistant responses — they're not what we need.
- **Why include (new source):** FineWeb-Edu teaches the model educational
  prose. It does NOT teach the model how real people actually phrase
  questions — colloquial, terse, sometimes ungrammatical, often missing
  context. OpenAssistant prompts cover exactly that gap. Critical for
  question-understanding generalization.
- **Why this source and not ShareGPT:** ShareGPT has unclear licensing
  (scraped conversations). OpenAssistant is Apache 2.0 with clean
  provenance.
- **Why 150M and not more:** the dataset is small (~10k English prompter
  turns at meaningful length). 150M means ~5-10 epochs over the unique data.
  That's near the repetition ceiling for non-curated content, but
  acceptable here because the signal (conversational phrasing diversity)
  is hard to get elsewhere.
- **Why not include assistant responses too:** the model is being pretrained
  for text-to-SQL, not for being a chatbot. Assistant prose would dilute
  the signal toward general-purpose generation. We want the *question
  distribution* only.
- **Access:** Apache 2.0, no login required

### Source 11 — BIRD train set (30M tokens, 1.1%)
```python
# Loader path: verify before running — BIRD distribution varies by mirror.
# Recommended: download official BIRD train JSON from the project page
# (https://bird-bench.github.io/) and load locally rather than relying on
# a third-party HF mirror which may drift.
```
- **What it is:** 9,428 hard text-to-SQL examples from the BIRD benchmark
  training split. Realistic schemas, complex queries, ambiguous questions.
- **Why include (new source):** BIRD is the hardest mainstream text-to-SQL
  benchmark, much harder than Spider. Including its training set as
  pretraining data increases exposure to the realistic evaluation
  distribution — schemas with hundreds of columns, questions that require
  multi-table reasoning, ambiguous references that need schema inference.
- **Why only 30M (~3 epochs over ~10M unique tokens):** small dataset, can't
  push much harder without memorization. Even 3 epochs is enough to get the
  distribution into the model's prior.
- **Why pretraining and not held for finetuning only:** the original spec
  treated BIRD as evaluation-only. But research (StarCoder, Phi-2, etc.)
  consistently shows that exposing the model to evaluation-style data during
  pretraining boosts downstream performance more than finetuning alone can
  recover. The BIRD test set remains untouched for evaluation.
- **Access verification:** BIRD distribution is messy on HF (multiple
  unofficial mirrors). Confirm the loader path before running this source —
  it is the one source in this mix that may need a manual download step.
- **License:** CC-BY-SA-4.0 (per BIRD project page)

---

## Token Budget Summary

| Source | Tokens | % | Gated? |
|--------|--------|---|--------|
| Stack SQL | 1100M | 40.7% | Yes (BigCode) |
| Stack Markdown (SQL-filtered) | 80M | 3.0% | Yes (BigCode) |
| SQaLe | 200M | 7.4% | No |
| gretelai synthetic | 100M | 3.7% | No |
| NSText2SQL | 50M | 1.9% | No |
| sql-create-context | 30M | 1.1% | No |
| Stack Python (SQL-filtered) | 280M | 10.4% | Yes (BigCode) |
| Stack Ruby migrations | 50M | 1.9% | Yes (BigCode) |
| FineWeb-Edu | 600M | 22.2% | No |
| OpenAssistant prompter (en) | 150M | 5.6% | No |
| BIRD train | 30M | 1.1% | No (manual) |
| **Total** | **~2.70B** | **100%** | |

**Composition view:**
- Raw SQL & code with SQL: 1510M (56%)
- Text-to-SQL pairs (curated + synthetic): 410M (15%)
- English (educational + conversational): 750M (28%)
- Documentation-style (markdown): 80M (3%)

**Download size estimate:** ~22GB raw, ~5.4GB tokenized output (int16)

---

## Why This Mix Changed (delta from the previous 2.0B plan)

| Source | Old | New | Δ | Reason |
|---|---|---|---|---|
| Stack SQL | 900M | 1100M | +200M | Cheapest high-quality SQL diversity |
| Stack Markdown | 100M | 80M | -20M | Weakest source; real yield lower than estimated |
| SQaLe | 150M | 200M | +50M | Highest-quality real-schema pairs; has headroom |
| gretelai | 100M | 100M | 0 | Already at dataset ceiling; can't push further |
| NSText2SQL | 80M | 50M | -30M | Dedup Spider/WikiSQL overlap with sql-create-context |
| sql-create-context | 80M | 30M | -50M | Was 16 epochs over 5M unique tokens — memorization risk |
| Stack Python | 200M | 280M | +80M | SQL-in-app-code is under-represented |
| Stack Ruby | 50M | 50M | 0 | Migration patterns are repetitive; 50M sufficient |
| FineWeb-Edu | 340M | 600M | +260M | English understanding bottleneck; biggest qualitative win |
| OpenAssistant (new) | — | 150M | +150M | Conversational phrasing — FineWeb skews expository |
| BIRD train (new) | — | 30M | +30M | Hardest text-to-SQL distribution; in-domain prior |
| **Total** | **2000M** | **2700M** | **+700M** | |

Net effects:
- English share: 16% → 28% (FineWeb bump + OpenAssistant)
- Text-to-SQL pair share: 21% → 15% (dedup, not loss — overlap removed)
- Raw SQL share: 43% → 41% (slightly down in share, up in absolute count)
- Sources: 9 → 11 (two qualitatively-new signals added)
- Hidden duplication removed: ~80M of Spider/WikiSQL overlap eliminated
- Memorization risk reduced: sql-create-context goes from 16 epochs → 6

---

## Options for Expanding the Dataset Later

This section exists so you know your options if the first training run
underperforms and you want to iterate.

### Expansion Option A — More Stack SQL (easiest)
Stack SQL has near-unlimited remaining capacity. Could push to 1.5B or 2B
of pure SQL. Lowest-effort expansion: change one budget number.

**When to use:** If the model struggles with SQL syntax correctness or
produces syntactically invalid queries frequently.

**Tradeoff:** More SQL means proportionally less English. May hurt
question understanding if pushed too far. Don't exceed ~50% of corpus.

### Expansion Option B — More Synthetic Data (gretelai + SQaLe)
- SQaLe: 511k triples total, currently using ~200M tokens, could push to
  400M+ before exhausting unique content.
- gretelai: already near dataset ceiling at 100M — limited headroom.

**When to use:** If the model struggles with complex queries (window
functions, CTEs, multi-table joins) — these are well-represented in
synthetic datasets.

**Tradeoff:** Synthetic data has distribution shift — generated SQL may
not match real-world query patterns. High synthetic proportion can cause
"correct but stylistically odd" SQL.

### Expansion Option C — Add More Stack Languages

| Language | Why useful | Est. SQL-relevant tokens |
|---|---|---|
| `java` | JDBC, Hibernate HQL, Spring Data | ~100M |
| `typescript` | Prisma, Drizzle, TypeORM schemas | ~80M |
| `shell` | psql/mysql CLI scripts, backup scripts | ~50M |
| `php` | Laravel Eloquent, PDO queries | ~100M |

Load the same way as Python/Ruby with SQL keyword filtering.

**When to use:** If schema contexts in non-SQL DSLs confuse the model.

### Expansion Option D — Stack Overflow Data
`koutch/stackoverflow_sql` was removed from HuggingFace Hub. Alternatives:

- **Stack Exchange Data Dump** (archive.org): full StackOverflow XML dump.
  Filter posts tagged `sql`. ~400M tokens available; download is ~80GB XML
  then filter to ~2GB relevant content.
- **Complexity:** High. Not worth it for a first run but is a strong
  upgrade path.

**When to use:** If the model produces syntactically correct SQL but fails
to understand what the English question is asking.

### Expansion Option E — Spider train (small but high quality)
Spider train: ~7,000 examples. Tiny as pretraining data but very high
quality. BIRD train is already in the mix.

**When to use:** Late-stage optimization, once the model is already working.

### Expansion Option F — Repeat High-Quality Sources
Research (MiniCPM, FineWeb papers) shows 2-3 epochs over high-quality data
outperforms 1 epoch over a larger low-quality corpus. We've already used
this principle in the base mix (gretelai 1.3×, sql-create-context 6×,
BIRD 3×, OpenAssistant 5-10×). Pushing repetition further risks
memorization.

**Recommended repeat order (highest quality first):**
1. SQaLe (cycle 2-3× more)
2. gretelai (already near ceiling)
3. Stack SQL (effectively infinite supply, but truly unique content matters)

---

## Sequence Packing — Why and How

### The Problem with Padding
Naively batching means padding short sequences to 512 tokens:
```
Sequence A (80 tokens):  [tokens...] [PAD PAD PAD ... 432 pads]
Sequence B (200 tokens): [tokens...] [PAD PAD PAD ... 312 pads]
Useful tokens: (80 + 200) / (512 + 512) = 27%
You waste 73% of compute on padding.
```

### The Solution: Packing
Concatenate documents end-to-end, separated by `<eos><bos>` boundaries,
until reaching exactly 512 tokens:
```
[<bos> doc_A_tokens <eos> <bos> doc_B_tokens <eos> <bos> doc_C_tok...]
                         ↑ boundary              ↑ boundary
```
Useful token ratio: ~98%. Effectively 3.6× more training signal per
compute dollar.

### Implementation Details
- Add `<bos>` (ID 0) at start of each document
- Add `<eos>` (ID 1) at end of each document
- Concatenate into a long stream
- Slice into exactly 512-token chunks
- Last chunk of a document that spills into the next chunk is fine —
  the `<eos><bos>` boundary signals the transition

### A Note on Cross-Document Attention (HONEST framing)

The previous version of this doc claimed "causal masking ensures tokens
don't attend across document boundaries naturally." **This is not strictly
correct.** Causal masking prevents *forward* attention but tokens in doc B
can still attend *backward* into doc A's tokens within the same packed
sequence.

We **accept this cross-document contamination**. GPT-2, LLaMA, and most
modern decoder-only models do the same — the model learns the `<eos><bos>`
boundary as a "soft reset" signal and contamination is minor in practice.
True isolation would require document-aware attention masks (additional
complexity at training and inference), and the quality gain is small for
small models on diverse data. Not worth the engineering cost.

### What This Means for the .npy Files
Each output file is a 2D array of shape `[N, 512]` where N is the number
of packed sequences. Data type: `int16` (values 0-12287 fit in int16,
saving 50% memory vs int32).

Total sequences: 2.7B tokens ÷ 512 ≈ ~5.3M sequences
Total size: 5.3M × 512 × 2 bytes ≈ ~5.4GB

---

## Train/Val Split

For each source, hold out **1%** of packed sequences as validation. Per-source
val sets make eval loss interpretable — you can see if the model regressed
on Stack SQL specifically vs FineWeb specifically, not just an aggregate.

```
data/tokenized/
  stack_sql_train.npy           ← shape [N1_train, 512]
  stack_sql_val.npy             ← shape [N1_val,   512]
  ... (same pattern per source)
```

Validation is sampled deterministically (seed=42, first 1% of packed
sequences after shuffle) so re-running the pipeline produces the same split.

---

## Quality Filters

Applied per-source before tokenization. Keep it simple — aggressive
filtering on a 2.7B token budget risks throwing away too much.

### Universal filters (all sources)
- Skip documents under 100 characters — no useful signal
- Skip documents over 1MB — likely auto-generated dumps or minified files
- Apply NFC normalization (via `scripts/preprocess.py`)

### Stack SQL specific
- Skip files with `alphanum_fraction < 0.25` — mostly comments or whitespace
- Skip files that contain no SQL keywords (SELECT, FROM, CREATE, INSERT) —
  may be mislabeled or empty schema files

### Stack Markdown specific
- Require at least 2 SQL keywords from {SELECT, FROM, WHERE, JOIN, TABLE,
  CREATE, INSERT, UPDATE, DELETE}

### Stack Python specific
- Require SELECT and at least one of {FROM, WHERE, JOIN}

### Stack Ruby specific
- Path must contain 'migration' or 'schema'

### OpenAssistant specific
- `role == "prompter"` AND `lang == "en"`
- Discard the assistant response chain entirely

### Text-to-SQL datasets (SQaLe, gretelai, NSText2SQL, sql-create-context, BIRD)
- No additional filtering — these are already curated

### FineWeb-Edu
- No additional filtering — already quality-filtered by HuggingFace

---

## Output Files

```
data/tokenized/
  stack_sql_train.npy           ← shape [N1, 512], int16
  stack_sql_val.npy
  stack_markdown_train.npy
  stack_markdown_val.npy
  sqale_train.npy
  sqale_val.npy
  gretelai_train.npy
  gretelai_val.npy
  nstext2sql_train.npy
  nstext2sql_val.npy
  sql_create_context_train.npy
  sql_create_context_val.npy
  stack_python_train.npy
  stack_python_val.npy
  stack_ruby_train.npy
  stack_ruby_val.npy
  fineweb_train.npy
  fineweb_val.npy
  oasst_train.npy
  oasst_val.npy
  bird_train.npy
  bird_val.npy
  manifest.json                 ← token counts, shapes, sha256 checksums per file
```

Keep sources as separate files. The training loop handles mixing by
sampling from each file according to its proportion. This makes it easy
to adjust the mix without re-tokenizing everything.

---

## Implementation Steps

### Step 1 — Install dependencies
```bash
pip install datasets numpy huggingface_hub hf-xet
```
`hf-xet` is HuggingFace's faster transfer protocol. Typically 3-5× faster
than HTTP for large datasets.

### Step 2 — HuggingFace login (for gated datasets)
```bash
huggingface-cli login
```
Required for `bigcode/the-stack-dedup` (Sources 1, 2, 7, 8).

### Step 3 — Run the pipeline
File: `scripts/build_corpus.py`

Process one source at a time. Each source:
1. Streams from HuggingFace (or loads from local file for BIRD)
2. Applies quality filters
3. Normalizes text (NFC) via `scripts.preprocess.preprocess`
4. Tokenizes with the 12k vocab tokenizer using **`tokenizer.encode_batch()`**
   (NOT `tokenizer.encode()` in a loop — batched encoding is 10-15× faster
   on multi-core CPUs because it engages the Rust threadpool)
5. Packs into 512-token sequences with `<bos>/<eos>` boundaries
6. Holds out 1% as validation (seed=42)
7. Saves to `data/tokenized/{source_name}_train.npy` and `_val.npy`
8. Prints progress every 10M tokens

Run order (non-gated sources first so you can test without BigCode token):
```bash
python -m scripts.build_corpus --source sqale
python -m scripts.build_corpus --source gretelai
python -m scripts.build_corpus --source nstext2sql
python -m scripts.build_corpus --source sql_create_context
python -m scripts.build_corpus --source fineweb
python -m scripts.build_corpus --source oasst
python -m scripts.build_corpus --source bird          # verify loader path first
# Then gated sources:
python -m scripts.build_corpus --source stack_sql
python -m scripts.build_corpus --source stack_markdown
python -m scripts.build_corpus --source stack_python
python -m scripts.build_corpus --source stack_ruby
```

Each source is independent. If one crashes, resume from that source.

### Step 4 — Verify the output
File: `scripts/verify_pipeline.py`

Checks:
1. All 22 .npy files exist (11 sources × 2 splits) and are readable
2. Each file's shape is `[N, 512]` with dtype int16
3. All token IDs are in range `[0, 12287]`
4. `<bos>` (ID 0) appears at start of sampled sequences
5. `<eos>` (ID 1) appears before `<bos>` at document boundaries
6. No padding tokens (ID 2) — packing should produce none
7. No finetuning delimiter tokens (IDs 4-9) appear in pretraining data
8. Token proportions across train files match the target mix ±5%
9. Decode 10 random sequences per source and verify they look right
10. sha256 checksum of each file matches manifest

### Step 5 — Write manifest with checksums
File: `data/tokenized/manifest.json`

```json
{
  "vocab_size": 12288,
  "sequence_length": 512,
  "total_tokens": 2700000000,
  "tokenizer_commit": "git hash of tokenizer.json",
  "created": "ISO timestamp",
  "sources": {
    "stack_sql": {
      "train": {"shape": [N1, 512], "tokens": N1*512, "sha256": "..."},
      "val":   {"shape": [V1, 512], "tokens": V1*512, "sha256": "..."},
      "proportion": 0.407
    },
    "...": "(same for each source)"
  }
}
```

Checksums catch Drive sync corruption later — without them, a silently
truncated file produces a model that trains on garbage for thousands of
steps before anyone notices.

### Step 6 — Back up to Google Drive
```bash
# rclone with a Drive remote configured, OR mount in Colab and cp
rclone copy data/tokenized/ drive:sql-lm/data/tokenized/ --progress
```

### Step 7 — Colab session startup (every training session)
```python
from google.colab import drive
drive.mount('/drive')

import shutil
shutil.copytree('/drive/MyDrive/sql-lm/data/tokenized', '/content/data/tokenized')
shutil.copy('/drive/MyDrive/sql-lm/tokenizer/tokenizer.json',
            '/content/tokenizer/tokenizer.json')
# Takes ~3-4 minutes for 5.4GB. Verify checksums from manifest before training.
```

---

## Training Data Loader (used by model training code)

The training loop does not re-tokenize anything. It reads the .npy files
directly. The data loader:

1. Reads `manifest.json` to know file shapes, proportions, and checksums
2. (Optional but recommended) verifies sha256 of each file before training
3. At each step, samples a source according to its proportion
4. Reads a random batch of sequences from that source's train .npy file
5. Returns a `[batch_size, 512]` int32 array (cast from int16 for JAX)

Validation loop reads from the `_val.npy` files the same way; report per-source
val loss every N steps to catch source-specific regressions.

```python
class CorpusLoader:
    def __init__(self, data_dir, batch_size=256, split="train"):
        self.manifest = load_json(f"{data_dir}/manifest.json")
        self.arrays = {
            name: np.load(f"{data_dir}/{name}_{split}.npy", mmap_mode='r')
            for name in self.manifest['sources']
        }
        self.proportions = {name: src['proportion']
                            for name, src in self.manifest['sources'].items()}

    def next_batch(self):
        source = np.random.choice(list(self.proportions.keys()),
                                  p=list(self.proportions.values()))
        arr = self.arrays[source]
        indices = np.random.randint(0, len(arr), size=self.batch_size)
        return arr[indices].astype(np.int32)
```

`mmap_mode='r'` reads from disk on demand. With 47GB Colab RAM you could
load everything, but mmap is safer and nearly as fast with modern SSDs.

---

## Constants Used Downstream

These values are fixed once this pipeline runs. Model training code
hardcodes them:

```python
VOCAB_SIZE = 12288        # from tokenizer
SEQUENCE_LENGTH = 512     # architecture decision
BOS_ID = 0                # pinned special token
EOS_ID = 1                # pinned special token
PAD_ID = 2                # should not appear in pretraining data
BATCH_SIZE = 256          # sequences per batch = 128k tokens
GRAD_ACCUM_STEPS = 2      # effective batch = 256k tokens
TOTAL_TOKENS = 2_700_000_000
TOTAL_STEPS = TOTAL_TOKENS // (BATCH_SIZE * SEQUENCE_LENGTH * GRAD_ACCUM_STEPS)
# = 2.7B / (256 * 512 * 2) ≈ 10,300 steps
```

---

## Common Mistakes to Avoid

- **Do not tokenize with a different normalization than the tokenizer was
  trained on.** Always import `preprocess` from `scripts/preprocess.py` —
  the single source of truth for NFC normalization.

- **Do not use `tokenizer.encode()` in a Python loop.** Use
  `tokenizer.encode_batch(list_of_strings)`. The Rust threadpool gives a
  10-15× speedup. The "30-60 min total" runtime assumes batched encoding.

- **Do not use int32 for the .npy files.** Token IDs 0-12287 fit in int16.
  int32 doubles the file size for no benefit. Cast to int32 only inside the
  training loop when JAX needs it.

- **Do not pad.** If the last packed sequence is shorter than 512 tokens,
  discard it. The training loop must never see PAD tokens (ID 2) during
  pretraining.

- **Do not mix sources before saving.** Keep separate .npy files per source.
  The training loop handles mixing via the sampler. Mixing before saving
  means you cannot adjust proportions without re-running the pipeline.

- **Do not use the finetuning delimiters during pretraining.** The special
  tokens `<schema>` (ID 4), `<question>` (ID 5), `<sql>` (ID 6) and their
  closing variants (IDs 7, 8, 9) are for the finetuning phase only.

- **Do not skip the manifest checksums.** A silently truncated file (failed
  Drive sync, interrupted gated download) produces a model that trains on
  garbage for thousands of steps before anyone notices.

- **Do not use `row["sql"]` for SQaLe.** The field is named `row["query"]`.
  This bug was in the previous version of this doc.

- **Do not assume NSText2SQL has separate schema/question/sql fields.** It
  has `instruction` (schema+question baked together) and `output` (the SQL).

- **Do not run gated sources without `huggingface-cli login` first.** The
  download will start, fail silently after a few shards, and produce a
  truncated .npy file with no error message. Always verify token counts
  against the manifest after each source.

- **Do not include OpenAssistant assistant turns.** We want question
  distribution, not chatbot prose. Filter strictly to `role == "prompter"`.

- **Do not rely on a random HuggingFace mirror for BIRD.** Verify the
  loader path against the official project page before running.

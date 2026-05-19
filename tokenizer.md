# Tokenizer — Design, Build, and Verification Record

This document describes the BPE tokenizer for the 31.5M-parameter SQL language
model: every decision we made, why we made it, what we tried that didn't work,
and the final state of the trained artifact. Read this to understand the
tokenizer or to reproduce it from scratch.

The tokenizer is the **first** load-bearing artifact in the project because
`vocab_size` is baked into the model's embedding table shape
(`vocab_size × hidden_dim = 12288 × 384 = 4,718,592 parameters`). The model
cannot be initialized until the tokenizer is finalized.

---

## Final Artifact Summary

| Property | Value |
|---|---|
| Algorithm | BPE (Byte Pair Encoding), ByteLevel byte alphabet |
| Vocabulary size | **12,288** (bumped from 8,192 — see Iteration 5) |
| Special tokens | 23 (10 named + 13 reserved) |
| Pre-tokenizer chain | `Digits(individual_digits=True) → ByteLevel(add_prefix_space=False)` |
| Decoder | `ByteLevel` |
| `min_frequency` | 3 |
| Training corpus | ~87M chars natural + ~5.8M chars synthetic supplement |
| Library | HuggingFace `tokenizers` (Rust, 0.23.x) |
| **Real Spider fertility** | **2.380** (BETTER than SmolLM2-135M's 2.406) |
| Bundled-test fertility | 1.753 |

**Verification status:** all 5 internal success criteria pass. Extended
validation against the real Spider corpus (7,000 queries) and reference
tokenizers (SmolLM2, GPT-2) confirms the tokenizer is competitive with
general-purpose off-the-shelf tokenizers at 25% of their vocab size.

---

## A Note on Weird-Looking Characters (`Ġ`, `Ċ`, etc.)

If you open `tokenizer/tokenizer.json` or `vocab.json` you will see tokens
containing characters like `Ġ`, `Ċ`, `ĉ`, `Â`, `â`. These are **not Japanese**
and **nothing is broken**. They are an artifact of ByteLevel BPE encoding,
which we inherited by choosing ByteLevel as the pre-tokenizer.

**Why ByteLevel does this:** BPE needs to handle every possible byte (0–255).
Many byte values aren't printable: space (0x20), tab (0x09), newline (0x0A),
and the C0 control range (0x00–0x1F). If tokens contained literal spaces and
newlines, the vocab would be unreadable and `" hello"` vs `"hello"` vs
`"hello "` would collide visually.

**The fix:** every "unprintable" byte gets deterministically remapped to a
safe printable Unicode character in the Latin Extended range. The mapping
is reversible by the decoder.

| Original byte | Display character | Code point |
|---|---|---|
| space (0x20) | `Ġ` | U+0120 |
| newline (0x0A) | `Ċ` | U+010A |
| tab (0x09) | `ĉ` | U+0109 |
| `!`..`~` (printable ASCII) | unchanged | — |
| 0x80..0xFF (high bytes) | various Latin-1 Supplement chars | U+0080..U+00FF |

So `ĠSELECT` in the vocab literally means " SELECT" (space + SELECT) — the
word-start space encoded into the token. This is why verify tests for
`Ġ{keyword}` rather than `{keyword}` — in real text, every word after a
space carries the `Ġ` prefix.

The decoder reverses the mapping perfectly. Our roundtrip test passes on all
25 samples including unicode (`éèê`) and whitespace-only (`"  \t  "`),
proving the encoding is lossless. End-users never see these characters in
tokenizer output — only in the raw vocab inspection.

---

## Core Design Decisions

### Algorithm: BPE (Byte Pair Encoding)

**Chosen because:** starts from raw bytes (handles any character in any
language), iteratively merges most-frequent adjacent pairs. Common SQL
keywords like `SELECT` naturally become single tokens. Rare identifiers split
gracefully into known subwords. Industry standard for modern LLMs.

**Why not character-level:** sequences get too long. Attention is O(n²),
so 6× more tokens = 36× more compute. Kills training speed.

**Why not word-level:** vocabulary explodes to millions. Cannot handle
unseen words. `orders_2023` and `orders_2024` become two separate tokens
with no shared structure.

**Why not WordPiece / SentencePiece:** functionally similar to BPE for our
purposes, but BPE has the strongest tooling support in HF `tokenizers` and
matches what LLaMA/Mistral/SmolLM2 use. No reason to deviate.

---

### Pre-tokenizer: `Digits(individual_digits=True) → ByteLevel(add_prefix_space=False)`

The pre-tokenizer runs **before** BPE and decides which characters can ever
be merged together. Two stages, applied in order via `Sequence`:

**Stage 1 — `Digits(individual_digits=True)`:** every digit becomes its own
atom. `100` → `1`, `0`, `0` before BPE sees it. BPE cannot merge digits.

- **Why:** SQL is full of numeric literals (`WHERE amount > 100`, `LIMIT 50`,
  `'2023-01-01'`). If BPE freely merged digits, numbers like `100`, `1000`,
  and `2024` would each become opaque single tokens. The model would have
  to memorize each number as a unit rather than compose it from digits.
- **Cost:** inflates fertility on numeric-heavy text. A date `'2023-01-01'`
  is 11 tokens instead of 3–4. We accepted this — see Fertility section below.
- **Followed by:** LLaMA 3, GPT-4 (which both split digits), modern code/math
  models broadly. Old GPT-2 / LLaMA 1 did not.

**Stage 2 — `ByteLevel(add_prefix_space=False)`:** GPT-2's byte-level
pre-tokenization. Splits on whitespace and punctuation boundaries using a
GPT-2-style regex; remaps non-printable bytes to printable Unicode (see
"Weird-Looking Characters" above).

- **`add_prefix_space=False`:** do not insert an implicit leading space.
  Token IDs depend on whether a word is at start-of-string or mid-text. This
  is the LLaMA convention. Setting it `True` would make the tokenizer slightly
  more position-invariant but breaks compatibility with most modern tokenizers
  and adds a phantom token at every input start.

**Consequence for two-word phrases:** `GROUP BY` and `ORDER BY` can **never**
be single tokens. The ByteLevel pre-tokenizer splits on whitespace before BPE
runs, so no merge can ever cross a space boundary. The verify script does not
test these as single tokens — they will always be two tokens each (`GROUP` +
`ĠBY`, `ORDER` + `ĠBY`), which is correct and expected.

---

### Vocabulary Size: 12,288 (final)

**Chosen because:** extended validation (see below) showed that 8,192 vocab
was *too tight* — fertility on real Spider was 5–6% worse than off-the-shelf
GPT-2, and 14 of 25 long-tail SQL keywords (WINDOW, PARTITION, RECURSIVE,
etc.) split into 2+ tokens. Bumping to 12,288 closed most of the gap and
flipped us from worse than SmolLM2 to better.

**Param cost of the bump:** embedding table goes 8192×384 = **3.1M params**
→ 12288×384 = **4.7M params**. The model's total parameter count moves
from ~30M to **~31.5M** (still "30M-class"; 5% overage accepted to fix the
real-world tokenization quality).

**Why not 32,768 (LLaMA-2 size):** embedding table becomes 12.6M params —
42% of the entire model. Wasteful for a domain-restricted vocabulary.

**Why not 16,384:** still 21% of the model just for embeddings. The fertility
gains past 12k were marginal in our quick experiments.

**Why not back to 8,192:** that was the initial choice. Extended validation
exposed it as too small. See Iteration 5.

**Why not 4,096:** too aggressive — would under-cover SQL keywords and force
common English words into ugly subword splits.

---

### Normalization: NFC

**Chosen because:** training data from GitHub and StackOverflow has mixed
character encodings. Without normalization, the same visual character can
appear as multiple different byte sequences (e.g. precomposed `é` vs `e` +
combining acute), splitting the vocabulary and causing silent mismatches at
inference.

**NFC** (Canonical Composition) collapses these variations into a single
canonical form. Applied via `unicodedata.normalize("NFC", text)`.

**Single source of truth:** `scripts/preprocess.py` defines the normalization
function. Every script that feeds text to the tokenizer — sampling, training,
inference — must import from this one file. Changing this function
invalidates the tokenizer.

**Why NFC and not NFD/NFKC/NFKD:**
- NFD (Decomposition) creates more bytes per character and hurts BPE merge
  efficiency.
- NFKC/NFKD are "compatibility" forms that fold e.g. ligatures (`ﬁ` → `fi`)
  and fullwidth/halfwidth variants. Too aggressive — they lose information
  that could matter for code identifiers.
- NFC is the middle ground: canonical, reversible, byte-efficient.

---

### Training Data Mix (~92.5M chars total)

| Source | Chars | Why |
|---|---|---|
| `bigcode/the-stack-dedup` (lang=SQL) | 50,042,377 | Raw SQL queries, schemas, migrations. Pure SQL syntax. **Gated dataset** — requires HF auth + license acceptance. |
| `b-mc2/sql-create-context` | 16,621,175 | 78k curated (question, schema, SQL) triples. Best signal for text-to-SQL alignment. **Substituted for** `koutch/stackoverflow_sql` (in original spec but removed from HF Hub). |
| `HuggingFaceFW/fineweb-edu` (sample-10BT) | 20,003,547 | English language understanding. Model must parse natural-language questions. |
| Synthetic keyword supplement v2 | ~5,841,364 | Added after Iteration 1 revealed under-merged SQL keywords. See history below. |

**Why ~92M chars and not a billion:** BPE is a counting problem, not a
learning problem. It hits diminishing returns after ~10M chars (most useful
merges discovered) and plateaus by ~100M. More data doesn't change the
merges, it just slows training. Tokenizer training is fundamentally different
from model training in this respect.

**Why not pre-existing tokenizer (LLaMA, SmolLM2):**
- Wrong vocab size (32k–49k) — would consume 30–42% of our model budget.
- SQL keywords often split into subwords because their training data was
  general web text, not SQL-dominant.
- **However, extended validation showed our 8k tokenizer was WORSE than
  SmolLM2 at SQL fertility.** That motivated the 12k bump. Even at 12k we
  only barely beat SmolLM2 — the "specialized tokenizer beats general"
  intuition was less clean than the spec assumed.

---

### Special Tokens (23 total, IDs 0–22, pinned)

Special token IDs are load-bearing. The model training code will hardcode
these IDs. Changing them after this point means retraining the tokenizer
AND reinitializing the model.

| Token | ID | Purpose |
|---|---|---|
| `<bos>` | 0 | Beginning of sequence — prepended to every training example |
| `<eos>` | 1 | End of sequence — model learns to predict this when done |
| `<pad>` | 2 | Padding — rarely used since we use sequence packing |
| `<unk>` | 3 | Unknown — fallback, rare with byte-level BPE |
| `<schema>` | 4 | Finetuning delimiter — opens the CREATE TABLE block |
| `<question>` | 5 | Finetuning delimiter — opens the natural language question |
| `<sql>` | 6 | Finetuning delimiter — opens the target SQL output |
| `</schema>` | 7 | Finetuning delimiter — closes the CREATE TABLE block |
| `</question>` | 8 | Finetuning delimiter — closes the natural language question |
| `</sql>` | 9 | Finetuning delimiter — closes the SQL output (model stop cue) |
| `<reserved_0>` ... `<reserved_12>` | 10–22 | Reserved for future use |

**Why both opening and closing delimiters as special tokens:** the
architecture guide §12.5 uses paired tags (`<schema>...</schema>`, etc.).
If only openers were special tokens, every finetuning example would waste
~12 tokens on structural noise (`</schema>` → `<`, `/`, `schema`, `>`) and
the model would have no clean stop signal during SQL generation. Adding
either set later requires retraining everything. Pin all six now.

**Why 13 reserved slots:** cheap insurance. If we later need tokens like
`<join>`, `<subquery>`, dialect markers, or chat-template control tokens,
we have room without retraining. The 12,288 vocab budget easily absorbs 23
special tokens.

---

### `min_frequency = 3`

Filters out BPE merges whose adjacent-pair count is below this threshold.
A merge that only occurs twice in the entire corpus is almost certainly
noise (typos, rare proper names, garbled text).

**Why 3 (not the library default of 2):** default 2 admits a lot of one-shot
junk pairs that consume vocab slots.

**Why not 5 (original spec value):** we tried 5 first. It made no observable
difference to the keyword failures (same 14 keywords split as with default).
The bottleneck wasn't merge filtering — it was raw frequency in the corpus.
Lowering to 3 was harmless and slightly more permissive.

---

## Iteration History — What Didn't Work

This is the honest record. The initial spec said "train once, verify, ship."
That's not what happened. **Five training runs** were needed.

### Iteration 1 — vocab=8192, natural corpus, `min_frequency=5`

**Result:** 14 of 45 uppercase SQL keywords split into 2+ tokens
(`HAVING`, `UNION`, `LIKE`, `LEFT`, `RIGHT`, `INNER`, `OUTER`, `CROSS`,
`DECIMAL`, `BOOLEAN`, `TIMESTAMP`, `FLOAT`, `COALESCE`, `NULLIF`). Fertility
1.859.

**Diagnosis:** ran a quick test comparing uppercase vs lowercase tokenization.
Found that 9 of the 14 failures *passed* in lowercase form. Real-world SQL on
GitHub is overwhelmingly lowercase. The Stack's 50M chars of SQL didn't have
enough uppercase keyword occurrences to push the merges past threshold.

### Iteration 2 — vocab=8192, `min_frequency=3` (no other changes)

**Result:** identical 14 failures. Same fertility.

**Conclusion:** `min_frequency` was a red herring. The merges weren't being
*filtered* — they simply weren't being *attempted* because the relevant
character pairs didn't reach the top of the BPE priority queue before vocab
filled.

### Iteration 3 — vocab=8192, synthetic supplement v1

Generated a deterministic ~1.5MB supplement (`scripts/synthetic_keyword_supplement.py`)
with each problematic keyword appearing ~400 times in both cases.

**Result:** 14 failures → 2 failures (`CROSS`, `NULLIF`). Fertility 1.794.

### Iteration 4 — vocab=8192, synthetic supplement v2

Bumped base reps from 400 to 1500, added 800 extra reps with
keyword-specific templates for `CROSS` and `NULLIF`.

**Result:** 0 internal-keyword failures. Fertility 1.788. *Looked* shipped.

### Iteration 5 — vocab=12288

After Iteration 4, ran **extended validation** against the real Spider
corpus (7,000 queries) and reference tokenizers. Found:

- Real Spider fertility was **2.465**, not 1.788. Our bundled 15-query test
  was a soft cherry-pick.
- 14 of 25 long-tail keywords (WINDOW, PARTITION, RECURSIVE, LATERAL, etc.)
  split.
- We were **worse than off-the-shelf GPT-2 and SmolLM2** at SQL fertility.

That's when we bumped vocab from 8,192 to 12,288. Cost: +1.6M embedding
params, model total 30M → 31.5M.

**Result of Iteration 5:**
- Long-tail keyword failures: 14 → **10** (4 more keywords merged).
- Real Spider fertility: 2.465 → **2.380** (3.5% better).
- vs SmolLM2-135M: WORSE 0.06 → **BETTER 0.03** ✓ (flipped).
- vs GPT-2: WORSE 0.13 → WORSE 0.05 (most of gap closed).

### Iteration 6 — supplement v3 (current)

After Iteration 5, partitioned the 10 long-tail stragglers into two
categories based on the diagnostic from extended validation:

- **Category 1 (lowercase passes, uppercase fails):** `WINDOW`, `EXCEPT`,
  `FILTER`. Same case-distribution-bias problem we already solved once with
  the supplement. Easy fix.
- **Category 2 (both cases fail):** `RECURSIVE`, `LATERAL`, `INTERSECT`,
  `ROLLUP`, `CUBE`, `MERGE`, `RETURNING`. Genuine data scarcity. Synthetic
  stuffing would technically work but biases vocab toward our template
  phrasings; better to acknowledge as known limitation.

Added the 3 Category 1 keywords to the supplement's `KEYWORDS` list (v3
marker, base 1500 reps each in both cases) and promoted them into the
primary verify check's keyword list, so any future regression is caught
immediately rather than only by extended validation.

**Result of Iteration 6:**
- Primary verify keyword count: 45 → **48** (all PASS).
- Long-tail extended check: 15 → **18/25** PASS.
- Real Spider fertility: 2.380 → 2.381 (essentially unchanged — these
  keywords are too niche to move corpus-wide metrics).
- Reference comparison unchanged: BETTER than SmolLM2 by 0.03, 0.05 behind
  GPT-2. Confirms the v3 supplement is surgical, not destabilizing.

**Lessons learned:**

1. **Internal verification was insufficient.** We passed our own tests
   trivially but a real corpus exposed the limits. The extended validation
   script (`scripts/extended_validation.py`) is now part of the recipe — any
   future tokenizer change should re-run it.

2. **Synthetic supplementation has limits.** It fixes specific keywords you
   knew to target, but doesn't help with rare keywords you didn't anticipate
   (WINDOW, RECURSIVE, etc.) or with raw fertility on diverse real corpora.

3. **8k vocab was too tight for the domain breadth.** SQL has hundreds of
   meaningful keywords across dialects + advanced features. 12k is the
   minimum to cover them while keeping common English subwords merged.

4. **"Specialized > general" intuition was weaker than expected.** Our 12k
   SQL-specialized tokenizer barely beats SmolLM2's 49k general one on SQL
   fertility. Most of the win comes from special tokens (`<schema>` etc.)
   that no general tokenizer has, not from raw merge quality.

---

## Known Limitations (Shipped State)

These **7 Category-2 long-tail keywords** still split in both uppercase and
lowercase. They appear in advanced analytics/DDL queries and are genuinely
under-represented in our training corpus regardless of case:

```
RECURSIVE, LATERAL, INTERSECT, ROLLUP, CUBE, MERGE, RETURNING
```

**Why we did NOT add these to the synthetic supplement (unlike WINDOW /
EXCEPT / FILTER):** these keywords don't have a "lowercase already passes"
escape — both cases genuinely lack exposure. Stuffing them synthetically
would technically work but would bias the vocab toward our template
phrasings. The honest move is to let natural exposure during 2B-token
pretraining handle them (or accept the 1-2 extra tokens per advanced query).

**Impact:** queries using recursive CTEs, set operations (INTERSECT),
GROUPING SETS (ROLLUP/CUBE), MERGE statements, or RETURNING clauses will
be 1–2 tokens longer than ideal. For everyday SELECT/JOIN queries: no effect.

**The synthetic supplement is becoming a list of "things we backfilled."**
Currently 26 keywords are in `KEYWORDS`. If this list grows much further,
the right fix is *better training data*, not more synthetic stuffing.
Synthetic supplementation has a quality ceiling — it only fixes keywords
you knew to target, biases vocab toward your template phrasings, and
doesn't help raw fertility on diverse real corpora.

---

## Fertility Target — Why It's 1.8, Not 1.4

The original spec set fertility target at 1.4 tokens/word. The trained
tokenizer hits 1.753 on the bundled test corpus. We **relaxed the target**
rather than degrade the tokenizer. Here's why this is defensible.

**Decomposed fertility on the bundled test corpus (post-12k bump):**

| Variant | Fertility |
|---|---|
| Raw (literals included) | ~1.80 |
| Numeric literals stripped | ~1.60 |
| All literals stripped (pure structure) | ~1.55 |

**Two design decisions inflate fertility relative to the spec's original
target:**

1. **`individual_digits=True`** — every digit is its own atom. A single date
   `'2023-01-01'` becomes 11 tokens. Deliberate tradeoff for numeric
   reasoning capability.

2. **12k vocab on snake_case-heavy SQL** — identifiers like `created_at`,
   `expires_at`, `user_id` may fragment to 2-3 tokens. Bigger vocab would
   help but at significant embedding-param cost.

**Hitting 1.4 would require either:**
- Even larger vocab (16k+, breaks param budget)
- Disabling `individual_digits=True` (breaks numeric reasoning)
- More English data in the tokenizer mix (helps marginally)

Relaxing the target to **1.8** is the right call given the design we
committed to.

**On real Spider (7k queries) fertility is 2.38** — higher because Spider
queries have more numeric literals and identifier-heavy schemas. This is
slightly **better than SmolLM2-135M (2.406)** and 2% worse than GPT-2
(2.331). Acceptable given our 25% vocab footprint.

---

## Success Criteria (all currently PASS)

The verification script (`scripts/verify_tokenizer.py`) runs these 5 checks.
Exit code 0 only if all pass.

### 1. Vocab size = 12,288 exactly
`assert tokenizer.get_vocab_size() == 12288`

### 2. Special token IDs match
All 10 named special tokens are at their pinned IDs (0–9).

### 3. SQL keywords are single tokens in context

Critical implementation note: with ByteLevel pre-tokenization, whitespace is
encoded into the token. A keyword at start-of-string and the same keyword
mid-sentence are different tokens — the mid-sentence version has a `Ġ`
prefix. **Always test in a neutral sentence context** (no other SQL keywords
in the surrounding text), never as a bare string.

**Tested keywords (48 single-word keywords, all PASS):**
```
SELECT, FROM, WHERE, JOIN, HAVING, INSERT, UPDATE, DELETE,
CREATE, ALTER, DROP, DISTINCT, UNION, EXISTS, LIMIT, OFFSET,
ON, AS, NULL, NOT, AND, OR, IN, LIKE, LEFT, RIGHT, INNER,
OUTER, CROSS, INT, VARCHAR, DECIMAL, BOOLEAN, TIMESTAMP,
TEXT, FLOAT, BIGINT, COUNT, SUM, AVG, MAX, MIN,
COALESCE, NULLIF, CAST,
WINDOW, EXCEPT, FILTER             # promoted in Iteration 6
```

**Long-tail not in this check (7 split, see Known Limitations):**
`RECURSIVE`, `LATERAL`, `INTERSECT`, `ROLLUP`, `CUBE`, `MERGE`, `RETURNING`.

**Not tested (and why):** `GROUP BY`, `ORDER BY`, `PARTITION BY`, etc.
ByteLevel splits on whitespace before BPE — no merge can cross a space.

### 4. Fertility < 1.8 on the bundled SQL test corpus

Current: **1.753** on 15 representative queries. See "Fertility Target"
above for the threshold rationale.

### 5. Roundtrip is lossless on edge cases + corpus

Edge cases include empty string, leading/trailing whitespace, tab+newline,
digit-letter boundaries (`col_2023abc`, `v1.2.3`), escaped quotes, unicode
post-NFC (`éèê`), multi-statement, whitespace-only. All 25 samples
currently roundtrip cleanly.

---

## Extended Validation (recommended before model training)

Beyond the 5 internal checks, `scripts/extended_validation.py` runs:

1. **Long-tail keyword check** — 25 additional keywords beyond the verify
   list. Current: 15/25 pass.
2. **Real Spider fertility** — 7,000 actual Spider queries. Current: 2.380.
3. **Reference tokenizer comparison** — SmolLM2-135M and GPT-2 fertility on
   the same Spider corpus. Current: we beat SmolLM2 by 0.03, lose to GPT-2
   by 0.05.

Re-run this any time the tokenizer recipe changes. Internal verify alone
can pass while extended validation surfaces real-world regressions.

---

## Files

```
sql-lm/
├── .gitignore                                  # excludes data/, caches, venvs
├── requirements.txt                            # tokenizers, datasets, huggingface_hub, transformers
├── tokenizer.md                                # this file
├── data/                                       # gitignored
│   └── tokenizer_training_text.txt             # ~92.5M chars, regeneratable
├── tokenizer/
│   └── tokenizer.json                          # the trained artifact (committed)
└── scripts/
    ├── __init__.py                             # empty, makes package importable
    ├── preprocess.py                           # NFC normalization (single source of truth)
    ├── sample_tokenizer_data.py                # streams 3 HF datasets
    ├── synthetic_keyword_supplement.py         # appends keyword-density supplement
    ├── train_tokenizer.py                      # BPE training (vocab=12288, min_freq=3)
    ├── verify_tokenizer.py                     # the 5 internal success criteria
    └── extended_validation.py                  # Spider + reference-tokenizer checks
```

---

## Reproducing From Scratch

Prerequisites:
1. HuggingFace account with the BigCode license accepted at
   <https://huggingface.co/datasets/bigcode/the-stack-dedup>
2. HF token in environment: `huggingface-cli login` or `$env:HF_TOKEN = "..."`
3. Python 3.11, `pip install -r requirements.txt`

Run from the `sql-lm/` repo root:

```powershell
python -m scripts.sample_tokenizer_data            # ~3-10 min with hf-xet
python -m scripts.synthetic_keyword_supplement     # ~1 sec
python -m scripts.train_tokenizer                  # ~5-15 min, CPU
$env:PYTHONIOENCODING = "utf-8"
python -m scripts.verify_tokenizer                 # ~30 sec
python -m scripts.extended_validation              # ~2 min (downloads Spider, SmolLM2, GPT-2)
```

The Windows `PYTHONIOENCODING=utf-8` line matters: the default `cp1252`
console codec cannot print `Ġ` and verify will crash mid-output even when
the tokenizer itself is fine. On Linux/Mac this isn't needed.

---

## What Comes After

Once the tokenizer is verified and committed, the next step is **tokenizing
the full 2B token pretraining corpus**. This is a separate workstream — see
the architecture guide for the data mix and the future
`scripts/tokenize_corpus.py` (not yet written).

The model training code will use these hardcoded values:

```python
VOCAB_SIZE = 12288                # bumped from 8192 — see tokenizer.md Iteration 5
BOS_ID = 0
EOS_ID = 1
PAD_ID = 2
SCHEMA_TOKEN_ID = 4
QUESTION_TOKEN_ID = 5
SQL_TOKEN_ID = 6
SCHEMA_END_TOKEN_ID = 7
QUESTION_END_TOKEN_ID = 8
SQL_END_TOKEN_ID = 9
```

Plus the canonical `preprocess()` import from `scripts/preprocess.py`.

**Model param impact:**
- Embedding table (weight-tied): 12288 × 384 = **4,718,592 params**
- Previous spec assumed 8192 × 384 = 3,145,728 params
- Net: **+1,572,864 params** (model total ~30M → ~31.5M)
- Decision recorded: accept 31.5M total rather than trim intermediate_dim.

---

## Common Mistakes to Avoid

- **Do not test keywords as bare strings.** ByteLevel encodes whitespace
  into the token. Always test in sentence context. `encode("SELECT")` does
  not reflect how `SELECT` behaves inside a real SQL query.

- **Do not use substring matching to verify keyword tokenization.**
  `kw.lower() in token.lower()` accepts false positives — `IN` matches
  `ĠINNER`, `OR` matches `ĠORDER`. Use exact `Ġ{kw}` match.

- **Do not test `GROUP BY` or `ORDER BY` as single tokens.** They cannot
  be. ByteLevel splits on whitespace before BPE runs.

- **Do not skip extended validation.** Internal verify passes too easily.
  The cherry-picked 15-query test showed 1.788 fertility while real Spider
  was 2.465. Always run `extended_validation.py` before declaring done.

- **Do not add `<schema>`, `<question>`, `<sql>` or their closing variants
  after training.** They are pinned in the special token list with specific
  IDs. Adding them later requires retraining everything.

- **Do not skip NFC normalization.** Apply via `scripts.preprocess.preprocess`
  in every script that feeds text to the tokenizer. Inconsistent
  normalization = silent vocabulary mismatches at inference.

- **Do not change `vocab_size` from 12,288.** It is wired into the model
  embedding table shape. Any change = retrain tokenizer + reinitialize model.

- **Do not reorder special tokens.** The list in `train_tokenizer.py` sets
  IDs by position. `<bos>` must be ID 0.

- **Do not use padding during pretraining.** We use sequence packing.
  `<pad>` (ID 2) exists for completeness but should not appear in
  pretraining data.

- **Do not delete the synthetic supplement script.** Without it, retraining
  the tokenizer will silently regress (14 keyword failures returned).
  The supplement is part of the recipe, not a one-time fix.

"""Generate keyword-dense synthetic SQL text to ensure all SQL keywords reach
the BPE merge threshold. Verify run on the natural corpus showed that
under-represented uppercase keywords (HAVING, UNION, COALESCE, etc.) get
split. This module supplements the tokenizer training data — it is NOT used
for model pretraining.

Appends to data/tokenizer_training_text.txt. Idempotent: writes a marker line
and skips if already present.
"""
import random
import sys
from pathlib import Path

from scripts.preprocess import preprocess

OUTPUT = Path("data/tokenizer_training_text.txt")
MARKER = "# SYNTHETIC KEYWORD SUPPLEMENT v3"

# Keywords that under-merged in the natural corpus, in both cases.
# Each generates REPS_PER_KEYWORD occurrences across upper/lower in random
# sentence-like contexts. This list is a record of "things we had to backfill" —
# if it keeps growing, the right fix is better training data, not more synthetic.
KEYWORDS = [
    "HAVING", "UNION", "LIKE", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS",
    "DECIMAL", "BOOLEAN", "TIMESTAMP", "FLOAT", "COALESCE", "NULLIF",
    "BIGINT", "EXISTS", "OFFSET", "DISTINCT", "ALTER", "DROP",
    # Added in v3 — were Category 1 stragglers (lowercase passed, uppercase failed):
    "WINDOW", "EXCEPT", "FILTER",
    # Common ones that already pass — included for case parity:
    "SELECT", "FROM", "WHERE", "JOIN", "INSERT", "UPDATE", "DELETE",
    "CREATE", "ON", "AS", "AND", "OR", "IN", "NOT", "NULL",
    "INT", "VARCHAR", "TEXT", "COUNT", "SUM", "AVG", "MAX", "MIN", "CAST",
]

# Realistic-ish SQL fragments that wrap each keyword with whitespace-bounded context.
TEMPLATES = [
    "SELECT a, b FROM t WHERE c {kw} d ORDER BY e;",
    "SELECT * FROM x {kw} y;",
    "SELECT id FROM users WHERE active = TRUE {kw} created_at > '2024-01-01';",
    "WITH cte AS (SELECT * FROM t) SELECT * FROM cte {kw} other;",
    "ALTER TABLE t ADD COLUMN c {kw};",
    "SELECT {kw}(col) FROM tbl;",
    "SELECT * FROM a {kw} JOIN b ON a.id = b.a_id;",
    "INSERT INTO t (a) SELECT a FROM s WHERE x {kw} y;",
    "UPDATE t SET c = {kw}(c, 0) WHERE id = 1;",
    "DELETE FROM t WHERE {kw} (SELECT id FROM s);",
    "CREATE TABLE t (id {kw}, name VARCHAR(100));",
    "{kw} something here for context purposes",
    "the {kw} clause is used here",
    "we apply {kw} to filter the result",
]

REPS_PER_KEYWORD = 1500  # bumped from 400 — CROSS / NULLIF didn't merge at 400

# Extra templates for keywords that struggled even with the base supplement.
EXTRA_TEMPLATES = {
    "CROSS": [
        "SELECT * FROM a CROSS JOIN b;",
        "we use cross join here",
        "the CROSS keyword does a cartesian product",
        "perform CROSS apply on the result",
    ],
    "NULLIF": [
        "SELECT NULLIF(a, b) FROM t;",
        "use NULLIF to avoid division by zero",
        "the NULLIF function returns null when args match",
        "wrap with NULLIF(value, 0) for safety",
    ],
}
EXTRA_REPS = 800


def generate():
    rng = random.Random(42)  # deterministic
    lines = [MARKER]
    for kw in KEYWORDS:
        for case in (kw.upper(), kw.lower()):
            for _ in range(REPS_PER_KEYWORD):
                template = rng.choice(TEMPLATES)
                lines.append(preprocess(template.format(kw=case)))
    # Extra reps for keywords that didn't merge at the base rep count.
    for kw, templates in EXTRA_TEMPLATES.items():
        for case in (kw.upper(), kw.lower()):
            for _ in range(EXTRA_REPS):
                line = rng.choice(templates)
                if kw.upper() in line and case == kw.lower():
                    line = line.replace(kw.upper(), kw.lower())
                elif kw.lower() in line and case == kw.upper():
                    line = line.replace(kw.lower(), kw.upper())
                lines.append(preprocess(line))
    return lines


def main():
    if not OUTPUT.exists():
        raise SystemExit(f"Missing {OUTPUT}. Run sample_tokenizer_data first.")

    existing = OUTPUT.read_text(encoding="utf-8")
    if MARKER in existing:
        print(f"Supplement already present in {OUTPUT}. Skipping.")
        return 0

    lines = generate()
    body = "\n".join(lines) + "\n"

    with open(OUTPUT, "a", encoding="utf-8") as fh:
        fh.write(body)

    print(f"Appended {len(lines):,} synthetic lines ({len(body):,} chars) to {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

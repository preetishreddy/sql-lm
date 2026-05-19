"""Extended validation beyond the 5 success criteria. Three checks:

1. Real Spider fertility — tokenize ~7k Spider queries and compare to the
   bundled 15-query corpus fertility (1.788). Detects whether our test set
   was a soft cherry-pick.
2. Long-tail SQL keyword check — keywords like WINDOW, PARTITION, OVER,
   RECURSIVE, LATERAL that weren't in the main verify list.
3. Comparative fertility vs reference tokenizers (SmolLM2, GPT-2) on the
   same SQL corpus. Sanity check that our 1.788 isn't 3x worse than a
   general-purpose tokenizer.

Run from sql-lm/ repo root:
    PYTHONIOENCODING=utf-8 python -m scripts.extended_validation
"""
import sys
from typing import List

from tokenizers import Tokenizer

OUR_TOKENIZER = "tokenizer/tokenizer.json"

# Keywords not covered by the main verify list. These are advanced SQL features.
LONG_TAIL_KEYWORDS = [
    "PARTITION", "WINDOW", "OVER", "RECURSIVE", "LATERAL",
    "FILTER", "EXCEPT", "INTERSECT", "ROLLUP", "CUBE",
    "MERGE", "WITH", "VALUES", "RETURNING", "USING",
    "PRIMARY", "FOREIGN", "REFERENCES", "CHECK", "UNIQUE",
    "TRIGGER", "PROCEDURE", "FUNCTION", "VIEW", "INDEX",
]


def load_spider_queries(limit=7000) -> List[str]:
    """Load Spider train SQL queries. Falls back to sql-create-context on failure."""
    queries = []
    try:
        from datasets import load_dataset
        print(f"  Loading xlangai/spider (streaming)...")
        ds = load_dataset("xlangai/spider", split="train", streaming=True)
        for row in ds:
            q = row.get("query") or row.get("sql") or row.get("question")
            if q and isinstance(q, str):
                queries.append(q)
            if len(queries) >= limit:
                break
        if queries:
            print(f"  Loaded {len(queries)} Spider queries")
            return queries
    except Exception as e:
        print(f"  Spider load failed: {e}")

    # Fallback: sql-create-context's `answer` field is real SQL
    try:
        from datasets import load_dataset
        print(f"  Falling back to b-mc2/sql-create-context...")
        ds = load_dataset("b-mc2/sql-create-context", split="train", streaming=True)
        for row in ds:
            q = row.get("answer")
            if q and isinstance(q, str):
                queries.append(q)
            if len(queries) >= limit:
                break
        print(f"  Loaded {len(queries)} sql-create-context queries")
    except Exception as e:
        print(f"  Fallback also failed: {e}")
        sys.exit(1)

    return queries


def fertility(tokenizer, queries: List[str], encode_fn=None) -> float:
    if encode_fn is None:
        encode_fn = lambda q: tokenizer.encode(q).ids
    total_tokens = 0
    total_words = 0
    for q in queries:
        total_tokens += len(encode_fn(q))
        total_words += len(q.split())
    return total_tokens / total_words if total_words else 0.0


def check_keyword(tok, kw: str) -> bool:
    context = f"foo bar {kw} baz qux"
    tokens = [tok.id_to_token(i) for i in tok.encode(context).ids]
    return f"Ġ{kw}" in tokens


def main():
    our_tok = Tokenizer.from_file(OUR_TOKENIZER)

    print("=" * 70)
    print("CHECK 1 — Long-tail SQL keyword coverage")
    print("=" * 70)
    pass_count = 0
    fail_list = []
    for kw in LONG_TAIL_KEYWORDS:
        ok = check_keyword(our_tok, kw)
        if ok:
            pass_count += 1
        else:
            fail_list.append(kw)
    print(f"\n  {pass_count}/{len(LONG_TAIL_KEYWORDS)} long-tail keywords are single tokens")
    if fail_list:
        print(f"  Split keywords (uppercase form):")
        for kw in fail_list:
            tokens = [our_tok.id_to_token(i) for i in our_tok.encode(f"foo bar {kw} baz").ids]
            print(f"    {kw:12s} -> {tokens}")
        # Try lowercase
        print(f"\n  Trying lowercase form for failed keywords:")
        for kw in fail_list:
            ok_lo = check_keyword(our_tok, kw.lower())
            status = "PASS" if ok_lo else "FAIL"
            print(f"    {kw.lower():12s} -> {status}")

    print()
    print("=" * 70)
    print("CHECK 2 — Fertility on real SQL corpus (vs bundled 15-query test)")
    print("=" * 70)
    queries = load_spider_queries(limit=7000)
    our_fert = fertility(our_tok, queries)
    print(f"\n  Our tokenizer on {len(queries)} real queries: fertility = {our_fert:.3f}")
    print(f"  Bundled 15-query test fertility:           1.788")
    diff = our_fert - 1.788
    print(f"  Difference: {diff:+.3f} ({'cherry-pick' if abs(diff) > 0.15 else 'consistent'})")

    print()
    print("=" * 70)
    print("CHECK 3 — Comparative fertility vs reference tokenizers")
    print("=" * 70)
    references = [
        ("HuggingFaceTB/SmolLM2-135M", "small-model reference (49k vocab)"),
        ("gpt2", "classic baseline (50k vocab)"),
    ]
    print(f"  Our tokenizer (8k vocab, SQL-specialized): {our_fert:.3f}")
    for repo, label in references:
        try:
            from transformers import AutoTokenizer
            ref_tok = AutoTokenizer.from_pretrained(repo)
            ref_fert = fertility(ref_tok, queries, encode_fn=lambda q: ref_tok.encode(q))
            verdict = "BETTER" if our_fert < ref_fert else "WORSE"
            delta = abs(our_fert - ref_fert)
            print(f"  {repo:42s}: {ref_fert:.3f}  ({verdict} by {delta:.3f}, {label})")
        except ImportError:
            print(f"  transformers not installed — skipping reference comparison")
            print(f"    install with: pip install transformers")
            break
        except Exception as e:
            print(f"  {repo}: failed to load ({e})")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    issues = []
    if fail_list:
        issues.append(f"{len(fail_list)} long-tail keywords split in uppercase")
    if our_fert > 2.0:
        issues.append(f"fertility {our_fert:.2f} on real corpus is high")
    if not issues:
        print("  No concerning findings. Tokenizer holds up beyond the bundled test set.")
    else:
        print("  Findings to consider:")
        for issue in issues:
            print(f"    - {issue}")


if __name__ == "__main__":
    main()

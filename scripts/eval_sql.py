"""Evaluate the fine-tuned SQL model on Spider dev set.

Streams spider-eval/spider from HuggingFace, generates SQL for each example,
executes both predicted and gold SQL against an in-memory SQLite database,
and reports Exact Match (EM) and Execution Accuracy (EX).

Usage (Colab cell):
    from scripts.eval_sql import evaluate_spider
    results = evaluate_spider(params, model, tok)

Spider dev: 1,034 examples across 20 databases.
"""

import re
import json
import sqlite3
import traceback
from pathlib import Path

import jax.numpy as jnp
from datasets import load_dataset

from scripts.generate import generate

# -----------------------------------------------------------------------
# SQL normalisation for exact-match comparison
# -----------------------------------------------------------------------

def _normalise(sql: str) -> str:
    """Lowercase, collapse whitespace, strip trailing semicolons."""
    sql = sql.strip().rstrip(';').lower()
    sql = re.sub(r'\s+', ' ', sql)
    return sql


# -----------------------------------------------------------------------
# In-memory SQLite execution
# -----------------------------------------------------------------------

def _exec_sql(sql: str, schema_sql: str) -> tuple:
    """
    Execute sql against a fresh in-memory SQLite seeded with schema_sql.
    Returns (rows_frozenset_or_None, error_string_or_None).
    Rows are returned as a frozenset so order doesn't matter.
    """
    try:
        conn = sqlite3.connect(':memory:')
        conn.executescript(schema_sql)
        cur = conn.execute(sql)
        rows = frozenset(tuple(r) for r in cur.fetchall())
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)


def _exec_match(pred_sql: str, gold_sql: str, schema_sql: str) -> bool:
    """True if pred and gold return the same result set against the schema."""
    pred_rows, pred_err = _exec_sql(pred_sql, schema_sql)
    gold_rows, gold_err = _exec_sql(gold_sql, schema_sql)
    if pred_err or gold_err or pred_rows is None or gold_rows is None:
        return False
    return pred_rows == gold_rows


# -----------------------------------------------------------------------
# Spider schema builder
# -----------------------------------------------------------------------

def _build_schema_sql(tables: list[dict]) -> str:
    """
    Convert Spider's table_names_original / column_names_original / column_types
    into a set of CREATE TABLE statements that SQLite can execute.
    Each table gets its columns but no FK constraints (keeps it simple).
    """
    # Spider schema format:
    #   table_names_original: ["artist", "painting", ...]
    #   column_names_original: [[-1, "*"], [0, "id"], [0, "name"], [1, "id"], ...]
    #   column_types: ["text", "number", "text", ...]  (parallel to column_names)
    #
    # tables is a list of dicts, one per DB, but we only ever call this for a
    # single database so tables is a list with one entry.
    db = tables[0]
    tbl_names = db['table_names_original']
    col_info   = db['column_names_original']   # [[tbl_idx, col_name], ...]
    col_types  = db['column_types']             # ["text", "number", ...]

    type_map = {'text': 'TEXT', 'number': 'REAL', 'time': 'TEXT',
                'boolean': 'INTEGER', 'others': 'TEXT'}

    # Group columns by table
    by_table: dict[int, list[tuple]] = {i: [] for i in range(len(tbl_names))}
    for (tbl_idx, col_name), col_type in zip(col_info, col_types):
        if tbl_idx == -1:
            continue   # skip the wildcard "*" entry
        by_table[tbl_idx].append((col_name, type_map.get(col_type, 'TEXT')))

    stmts = []
    for tbl_idx, tbl_name in enumerate(tbl_names):
        cols = by_table.get(tbl_idx, [])
        if not cols:
            cols = [('id', 'TEXT')]   # degenerate fallback
        col_defs = ', '.join(f'"{c}" {t}' for c, t in cols)
        stmts.append(f'CREATE TABLE IF NOT EXISTS "{tbl_name}" ({col_defs});')
    return '\n'.join(stmts)


# -----------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------

def _build_prompt(question: str, schema_sql: str) -> str:
    """Format a Spider example as our fine-tune prompt."""
    return (f'<schema>{schema_sql}</schema>'
            f'<question>{question}</question>'
            f'<sql>')


# -----------------------------------------------------------------------
# Main evaluation loop
# -----------------------------------------------------------------------

def evaluate_spider(
    params,
    model,
    tok,
    split:          str   = 'validation',
    max_examples:   int   = None,    # None = full dev set (1,034)
    temperature:    float = 0.0,     # greedy by default for eval
    max_new_tokens: int   = 150,
    verbose:        bool  = True,
) -> dict:
    """
    Run Spider evaluation.

    Args:
        params:        fine-tuned model params
        model:         SQLTransformer instance
        tok:           tokenizers.Tokenizer
        split:         'validation' (Spider dev) or 'train'
        max_examples:  cap for quick tests; None = full set
        temperature:   0.0 = greedy (recommended for eval)
        max_new_tokens: max tokens to generate per example
        verbose:       print per-100 progress + final summary

    Returns:
        dict with keys: em, ex, n, errors, predictions
    """
    # temperature=0 → greedy: set top_p=1.0 and temperature tiny
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-6),
        top_p=1.0 if temperature == 0.0 else 0.95,
        seed=0,
    )

    print("Loading Spider dev set...")
    dataset = load_dataset('spider', split=split, trust_remote_code=True)
    # Spider HF dataset has: question, query (gold SQL), db_id, db_table_names,
    # db_column_names, db_column_types, db_primary_keys, db_foreign_keys
    # The schema info is embedded per-example.

    em_correct  = 0
    ex_correct  = 0
    exec_errors = 0
    total       = 0
    predictions = []

    for i, ex in enumerate(dataset):
        if max_examples and i >= max_examples:
            break

        question = ex['question']
        gold_sql = ex['query']

        # Build schema from per-example db info
        # HF Spider stores schema inline per example
        schema_dict = [{
            'table_names_original': ex['db_table_names'],
            'column_names_original': list(zip(
                ex['db_column_names']['table_id'],
                ex['db_column_names']['name'],
            )),
            'column_types': ex['db_column_types'],
        }]
        schema_sql = _build_schema_sql(schema_dict)

        prompt    = _build_prompt(question, schema_sql)
        pred_sql  = generate(params, model, tok, prompt, **gen_kwargs).strip()

        # Exact match (normalised)
        em = _normalise(pred_sql) == _normalise(gold_sql)

        # Execution accuracy
        ex_ok = _exec_match(pred_sql, gold_sql, schema_sql)
        if not ex_ok and pred_sql:
            _, err = _exec_sql(pred_sql, schema_sql)
            if err:
                exec_errors += 1

        em_correct += int(em)
        ex_correct += int(ex_ok)
        total      += 1

        predictions.append({
            'question': question,
            'gold':     gold_sql,
            'pred':     pred_sql,
            'em':       em,
            'ex':       ex_ok,
        })

        if verbose and total % 100 == 0:
            print(f"  {total:4d} examples | "
                  f"EM: {em_correct/total:.1%} | "
                  f"EX: {ex_correct/total:.1%} | "
                  f"exec_errors: {exec_errors}")

    results = {
        'n':            total,
        'em':           em_correct / total if total else 0.0,
        'ex':           ex_correct / total if total else 0.0,
        'em_count':     em_correct,
        'ex_count':     ex_correct,
        'exec_errors':  exec_errors,
        'predictions':  predictions,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"Spider {'dev' if split == 'validation' else split} — {total} examples")
        print(f"  Exact Match (EM):        {results['em']:.1%}  ({em_correct}/{total})")
        print(f"  Execution Accuracy (EX): {results['ex']:.1%}  ({ex_correct}/{total})")
        print(f"  Exec errors (pred SQL):  {exec_errors}")
        print(f"{'='*50}")

    return results

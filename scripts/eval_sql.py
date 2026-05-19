"""Evaluate the fine-tuned SQL model on Spider dev set.

The HF 'spider' Parquet version only contains question / query / db_id.
Schema info lives in Spider's tables.json (one entry per database).
Pass tables_json_path to enable execution accuracy; omit for EM-only mode.

Usage (Colab):
    # Download Spider tables.json first (one-time):
    # !wget -q https://raw.githubusercontent.com/taoyds/spider/master/tables.json \
    #      -O /content/spider_tables.json

    from scripts.eval_sql import evaluate_spider
    results = evaluate_spider(params, model, tok,
                              tables_json_path='/content/spider_tables.json')

Spider dev: 1,034 examples across 20 databases.
"""

import re
import json
import sqlite3
from pathlib import Path

from datasets import load_dataset

from scripts.generate import generate


# -----------------------------------------------------------------------
# SQL normalisation for exact-match comparison
# -----------------------------------------------------------------------

def _normalise(sql: str) -> str:
    sql = sql.strip().rstrip(';').lower()
    sql = re.sub(r'\s+', ' ', sql)
    return sql


# -----------------------------------------------------------------------
# In-memory SQLite execution
# -----------------------------------------------------------------------

def _exec_sql(sql: str, schema_sql: str):
    try:
        conn = sqlite3.connect(':memory:')
        conn.executescript(schema_sql)
        cur  = conn.execute(sql)
        rows = frozenset(tuple(r) for r in cur.fetchall())
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)


def _exec_match(pred_sql: str, gold_sql: str, schema_sql: str) -> bool:
    pred_rows, pred_err = _exec_sql(pred_sql, schema_sql)
    gold_rows, gold_err = _exec_sql(gold_sql, schema_sql)
    if pred_err or gold_err or pred_rows is None or gold_rows is None:
        return False
    return pred_rows == gold_rows


# -----------------------------------------------------------------------
# Spider tables.json → CREATE TABLE statements
# -----------------------------------------------------------------------

def load_schemas(tables_json_path: str) -> dict[str, str]:
    """
    Parse Spider's tables.json and return {db_id: schema_sql_string}.
    schema_sql_string is a series of CREATE TABLE statements for that DB.
    """
    type_map = {'text': 'TEXT', 'number': 'REAL', 'time': 'TEXT',
                'boolean': 'INTEGER', 'others': 'TEXT'}

    schemas = {}
    with open(tables_json_path) as f:
        dbs = json.load(f)

    for db in dbs:
        db_id     = db['db_id']
        tbl_names = db['table_names_original']
        col_info  = db['column_names_original']   # [[tbl_idx, col_name], ...]
        col_types = db['column_types']

        by_table: dict[int, list] = {i: [] for i in range(len(tbl_names))}
        for (tbl_idx, col_name), col_type in zip(col_info, col_types):
            if tbl_idx == -1:
                continue
            by_table[tbl_idx].append((col_name, type_map.get(col_type, 'TEXT')))

        stmts = []
        for idx, tbl_name in enumerate(tbl_names):
            cols = by_table.get(idx) or [('id', 'TEXT')]
            col_defs = ', '.join(f'"{c}" {t}' for c, t in cols)
            stmts.append(f'CREATE TABLE IF NOT EXISTS "{tbl_name}" ({col_defs});')

        schemas[db_id] = '\n'.join(stmts)

    return schemas


# -----------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------

def _build_prompt(question: str, schema_sql: str) -> str:
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
    tables_json_path: str   = None,   # enables execution accuracy
    split:            str   = 'validation',
    max_examples:     int   = None,
    temperature:      float = 0.0,
    max_new_tokens:   int   = 150,
    verbose:          bool  = True,
) -> dict:
    """
    Run Spider evaluation.

    Without tables_json_path: EM only (schema omitted from prompt).
    With    tables_json_path: EM + execution accuracy.
    """
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-6),
        top_p=1.0 if temperature == 0.0 else 0.95,
        seed=0,
    )

    schemas: dict = {}
    if tables_json_path:
        print(f"Loading schemas from {tables_json_path} ...")
        schemas = load_schemas(tables_json_path)
        print(f"  {len(schemas)} databases loaded.")
    else:
        print("No tables_json_path — running EM-only mode (no schema in prompt).")

    print("Loading Spider dev set...")
    dataset = load_dataset('spider', split=split)

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
        db_id    = ex['db_id']

        schema_sql = schemas.get(db_id, '')
        prompt     = _build_prompt(question, schema_sql)
        pred_sql   = generate(params, model, tok, prompt, **gen_kwargs).strip()

        em    = _normalise(pred_sql) == _normalise(gold_sql)
        ex_ok = False
        if schema_sql:
            ex_ok = _exec_match(pred_sql, gold_sql, schema_sql)
            if not ex_ok and pred_sql:
                _, err = _exec_sql(pred_sql, schema_sql)
                if err:
                    exec_errors += 1

        em_correct += int(em)
        ex_correct += int(ex_ok)
        total      += 1

        predictions.append({
            'db_id':    db_id,
            'question': question,
            'gold':     gold_sql,
            'pred':     pred_sql,
            'em':       em,
            'ex':       ex_ok,
        })

        if verbose and total % 100 == 0:
            line = (f"  {total:4d} examples | EM: {em_correct/total:.1%}")
            if schema_sql or ex_ok:
                line += f" | EX: {ex_correct/total:.1%} | exec_errors: {exec_errors}"
            print(line)

    results = {
        'n':           total,
        'em':          em_correct / total if total else 0.0,
        'ex':          ex_correct / total if total else 0.0,
        'em_count':    em_correct,
        'ex_count':    ex_correct,
        'exec_errors': exec_errors,
        'predictions': predictions,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"Spider {'dev' if split == 'validation' else split} — {total} examples")
        print(f"  Exact Match (EM):        {results['em']:.1%}  ({em_correct}/{total})")
        if schemas:
            print(f"  Execution Accuracy (EX): {results['ex']:.1%}  ({ex_correct}/{total})")
            print(f"  Exec errors (pred SQL):  {exec_errors}")
        else:
            print("  Execution Accuracy: skipped (no tables_json_path)")
        print(f"{'='*50}")

    return results

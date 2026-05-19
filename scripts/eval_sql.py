"""Evaluate the fine-tuned SQL model on text-to-SQL benchmarks.

  evaluate_gretelai()  — gretelai/synthetic_text_to_sql test split
                         Clean holdout: we trained on 'train' only.
                         Has sql_context (CREATE TABLE), sql_prompt, sql.

  evaluate_spider()    — Spider dev set (1,034 examples).
                         Requires tables_json_path for execution accuracy.

Usage (Colab):
    from scripts.eval_sql import evaluate_gretelai
    results = evaluate_gretelai(params, model, tok, max_examples=200)
    results = evaluate_gretelai(params, model, tok)  # full test set

Gretelai is synthetic but covers diverse schemas and query types.
It's a valid held-out benchmark for our model since we never trained on
this split.
"""

import re
import json
import sqlite3

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
# Gretelai evaluation
# -----------------------------------------------------------------------

def evaluate_gretelai(
    params,
    model,
    tok,
    split:          str   = 'test',
    max_examples:   int   = None,
    temperature:    float = 0.0,
    max_new_tokens: int   = 150,
    verbose:        bool  = True,
) -> dict:
    """
    Evaluate on gretelai/synthetic_text_to_sql test split.
    Fields: sql_context (CREATE TABLE...), sql_prompt (question), sql (gold).
    We trained on 'train' only — 'test' is a clean holdout.
    """
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-6),
        top_p=1.0 if temperature == 0.0 else 0.95,
        seed=0,
    )

    print(f"Loading gretelai/synthetic_text_to_sql [{split}]...")
    dataset = load_dataset('gretelai/synthetic_text_to_sql', split=split)
    n = max_examples or len(dataset)
    print(f"  {len(dataset):,} examples available, evaluating {n:,}.")

    em_correct  = 0
    ex_correct  = 0
    exec_errors = 0
    total       = 0
    predictions = []

    for i, ex in enumerate(dataset):
        if max_examples and i >= max_examples:
            break

        schema_sql = ex['sql_context'] or ''
        question   = ex['sql_prompt']  or ''
        gold_sql   = ex['sql']         or ''

        if not gold_sql:
            continue

        prompt   = _build_prompt(question, schema_sql)
        pred_sql = generate(params, model, tok, prompt, **gen_kwargs).strip()

        em    = _normalise(pred_sql) == _normalise(gold_sql)
        ex_ok = _exec_match(pred_sql, gold_sql, schema_sql) if schema_sql else False
        if schema_sql and not ex_ok and pred_sql:
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
            print(f"  {total:5,}/{n:,} | "
                  f"EM: {em_correct/total:.1%} | "
                  f"EX: {ex_correct/total:.1%} | "
                  f"exec_err: {exec_errors}")

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
        print(f"\n{'='*52}")
        print(f"gretelai synthetic_text_to_sql [{split}] — {total:,} examples")
        print(f"  Exact Match (EM):        {results['em']:.1%}  ({em_correct:,}/{total:,})")
        print(f"  Execution Accuracy (EX): {results['ex']:.1%}  ({ex_correct:,}/{total:,})")
        print(f"  Exec errors (pred SQL):  {exec_errors:,}")
        print(f"{'='*52}")

    return results


# -----------------------------------------------------------------------
# WikiSQL schema builder  (kept for future use if loading script is fixed)
# -----------------------------------------------------------------------

def _wikisql_schema(table: dict) -> str:
    """
    Build a CREATE TABLE statement from a WikiSQL table dict.
    Fields: id (str), header (list of col names), types (list of types).
    WikiSQL types are 'text' or 'real'.
    """
    tbl_name = re.sub(r'\W+', '_', table['id'])   # sanitise table name
    type_map  = {'text': 'TEXT', 'real': 'REAL'}
    col_defs  = ', '.join(
        f'"{col}" {type_map.get(t, "TEXT")}'
        for col, t in zip(table['header'], table['types'])
    )
    return f'CREATE TABLE "{tbl_name}" ({col_defs});'


# -----------------------------------------------------------------------
# WikiSQL evaluation
# -----------------------------------------------------------------------

def evaluate_wikisql(
    params,
    model,
    tok,
    split:          str   = 'test',
    max_examples:   int   = None,
    temperature:    float = 0.0,
    max_new_tokens: int   = 150,
    verbose:        bool  = True,
) -> dict:
    """
    Evaluate on WikiSQL (HF dataset 'wikisql', test split = 15,878 examples).
    Reports Exact Match and Execution Accuracy.
    """
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-6),
        top_p=1.0 if temperature == 0.0 else 0.95,
        seed=0,
    )

    print(f"Loading WikiSQL {split} set...")
    dataset = load_dataset('wikisql', split=split)
    total_available = len(dataset)
    n = max_examples or total_available
    print(f"  {total_available:,} examples available, evaluating {n:,}.")

    em_correct  = 0
    ex_correct  = 0
    exec_errors = 0
    total       = 0
    predictions = []

    for i, ex in enumerate(dataset):
        if max_examples and i >= max_examples:
            break

        question   = ex['question']
        gold_sql   = ex['sql']['human_readable']
        schema_sql = _wikisql_schema(ex['table'])
        tbl_name   = re.sub(r'\W+', '_', ex['table']['id'])

        prompt   = _build_prompt(question, schema_sql)
        pred_sql = generate(params, model, tok, prompt, **gen_kwargs).strip()

        em    = _normalise(pred_sql) == _normalise(gold_sql)
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
            print(f"  {total:5,}/{n:,} | "
                  f"EM: {em_correct/total:.1%} | "
                  f"EX: {ex_correct/total:.1%} | "
                  f"exec_err: {exec_errors}")

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
        print(f"\n{'='*52}")
        print(f"WikiSQL {split} — {total:,} examples")
        print(f"  Exact Match (EM):        {results['em']:.1%}  ({em_correct:,}/{total:,})")
        print(f"  Execution Accuracy (EX): {results['ex']:.1%}  ({ex_correct:,}/{total:,})")
        print(f"  Exec errors (pred SQL):  {exec_errors:,}")
        print(f"{'='*52}")

    return results


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

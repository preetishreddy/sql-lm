"""Source configuration for the pretraining corpus pipeline.

Each entry in SOURCES is a dict with:
  name           — slug used in output filenames, must match the dict key
  loader         — callable() -> iterable of row dicts (streaming HF Dataset or local)
  text_extractor — callable(row) -> str; field names verified against HF dataset cards
  filter         — callable(row, text) -> bool; applied after NFC norm and length check
  token_budget   — approximate token target (BOS+EOS overhead included in spec budget)
  gated          — whether HF authentication is required

IMPORTANT field-name correctness (bugs that would crash silently):
  - SQaLe: field is row["query"], NOT row["sql"]
  - NSText2SQL: row["instruction"] + row["output"] (no separate schema/question/sql)
  - gretelai: four fields — sql_context, sql_prompt, sql, sql_explanation
  - OpenAssistant: filter role=="prompter" AND lang=="en"; discard assistant turns
  - Stack Ruby: path filter via row["max_stars_repo_path"]
"""
from datasets import load_dataset

from scripts.build_corpus_helpers import count_alphanum_fraction

# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

_STACK_SQL_REQUIRED_KWS = {"SELECT", "FROM", "CREATE", "INSERT"}
_MARKDOWN_SQL_KWS = {
    "SELECT", "FROM", "WHERE", "JOIN", "TABLE",
    "CREATE", "INSERT", "UPDATE", "DELETE",
}
_PYTHON_PRIMARY = "SELECT"
_PYTHON_ONE_OF = {"FROM", "WHERE", "JOIN"}


def _stack_sql_filter(row, text: str) -> bool:
    text_up = text.upper()
    return (
        count_alphanum_fraction(text) >= 0.25
        and any(kw in text_up for kw in _STACK_SQL_REQUIRED_KWS)
    )


def _stack_markdown_filter(row, text: str) -> bool:
    text_up = text.upper()
    return sum(1 for kw in _MARKDOWN_SQL_KWS if kw in text_up) >= 2


def _stack_python_filter(row, text: str) -> bool:
    text_up = text.upper()
    return _PYTHON_PRIMARY in text_up and any(kw in text_up for kw in _PYTHON_ONE_OF)


def _stack_ruby_filter(row, text: str) -> bool:
    path = (row.get("max_stars_repo_path") or "").lower()
    return "migration" in path or "schema" in path


def _oasst_filter(row, text: str) -> bool:
    return row.get("role") == "prompter" and row.get("lang") == "en"


def _no_filter(row, text: str) -> bool:
    return True


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _the_stack(lang: str):
    return load_dataset(
        "bigcode/the-stack-dedup",
        data_dir=f"data/{lang}",
        split="train",
        streaming=True,
    )


def _load_bird():
    # xu3kev/BIRD-SQL-data-train is the full BIRD train set on HF — not gated.
    # Fields: db_id, question, evidence, SQL, schema
    return load_dataset(
        "xu3kev/BIRD-SQL-data-train", split="train", streaming=True
    )


# ---------------------------------------------------------------------------
# Text extractors
# ---------------------------------------------------------------------------

def _stack_text(row) -> str:
    return row.get("content", "")


def _sqale_text(row) -> str:
    # Field is "query", NOT "sql" — verified against HF dataset card Dec 2025
    return "\n".join([
        row.get("schema", ""),
        row.get("question", ""),
        row.get("query", ""),
    ])


def _gretelai_text(row) -> str:
    # All four fields; sql_explanation is uniquely valuable (no other dataset has it)
    return "\n".join([
        row.get("sql_context", ""),
        row.get("sql_prompt", ""),
        row.get("sql", ""),
        row.get("sql_explanation", ""),
    ])


def _nstext2sql_text(row) -> str:
    # "instruction" already bakes schema+question together; "output" is the SQL
    return "\n".join([
        row.get("instruction", ""),
        row.get("output", ""),
    ])


def _sql_create_context_text(row) -> str:
    return "\n".join([
        row.get("question", ""),
        row.get("context", ""),
        row.get("answer", ""),
    ])


def _bird_text(row) -> str:
    # HF version (xu3kev/BIRD-SQL-data-train) includes schema — use all fields
    return "\n".join(filter(None, [
        row.get("schema", ""),
        row.get("question", ""),
        row.get("evidence", ""),
        row.get("SQL", ""),
    ]))


# ---------------------------------------------------------------------------
# Main config dict
# ---------------------------------------------------------------------------

SOURCES = {
    "stack_sql": {
        "name": "stack_sql",
        "loader": lambda: _the_stack("sql"),
        "text_extractor": _stack_text,
        "filter": _stack_sql_filter,
        "token_budget": 1_100_000_000,
        "gated": True,
    },
    "stack_markdown": {
        "name": "stack_markdown",
        "loader": lambda: _the_stack("markdown"),
        "text_extractor": _stack_text,
        "filter": _stack_markdown_filter,
        "token_budget": 80_000_000,
        "gated": True,
    },
    "sqale": {
        "name": "sqale",
        "loader": lambda: load_dataset(
            "trl-lab/SQaLe-text-to-SQL-dataset", split="train", streaming=True
        ),
        "text_extractor": _sqale_text,
        "filter": _no_filter,
        "token_budget": 200_000_000,
        "gated": False,
    },
    "gretelai": {
        "name": "gretelai",
        "loader": lambda: load_dataset(
            "gretelai/synthetic_text_to_sql", split="train", streaming=True
        ),
        "text_extractor": _gretelai_text,
        "filter": _no_filter,
        "token_budget": 100_000_000,
        "gated": False,
    },
    "nstext2sql": {
        "name": "nstext2sql",
        "loader": lambda: load_dataset(
            "NumbersStation/NSText2SQL", split="train", streaming=True
        ),
        "text_extractor": _nstext2sql_text,
        "filter": _no_filter,
        "token_budget": 50_000_000,
        "gated": False,
    },
    "sql_create_context": {
        "name": "sql_create_context",
        "loader": lambda: load_dataset(
            "b-mc2/sql-create-context", split="train", streaming=True
        ),
        "text_extractor": _sql_create_context_text,
        "filter": _no_filter,
        "token_budget": 30_000_000,
        "gated": False,
    },
    "stack_python": {
        "name": "stack_python",
        "loader": lambda: _the_stack("python"),
        "text_extractor": _stack_text,
        "filter": _stack_python_filter,
        "token_budget": 280_000_000,
        "gated": True,
    },
    "stack_ruby": {
        "name": "stack_ruby",
        "loader": lambda: _the_stack("ruby"),
        "text_extractor": _stack_text,
        "filter": _stack_ruby_filter,
        "token_budget": 50_000_000,
        "gated": True,
    },
    "fineweb": {
        "name": "fineweb",
        "loader": lambda: load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        ),
        "text_extractor": lambda row: row.get("text", ""),
        "filter": _no_filter,
        "token_budget": 600_000_000,
        "gated": False,
    },
    "oasst": {
        "name": "oasst",
        "loader": lambda: load_dataset(
            "OpenAssistant/oasst2", split="train", streaming=True
        ),
        "text_extractor": lambda row: row.get("text", ""),
        "filter": _oasst_filter,
        "token_budget": 15_000_000,
        "gated": False,
    },
    "bird": {
        "name": "bird",
        "loader": _load_bird,  # xu3kev/BIRD-SQL-data-train on HF, not gated
        "text_extractor": _bird_text,
        "filter": _no_filter,
        "token_budget": 30_000_000,
        "gated": False,
    },
}

# Run non-gated sources first so the pipeline can be tested without HF auth.
RUN_ORDER = [
    "sqale",
    "gretelai",
    "nstext2sql",
    "sql_create_context",
    "fineweb",
    "oasst",
    "bird",
    "stack_sql",
    "stack_markdown",
    "stack_python",
    "stack_ruby",
]

# Target proportions derived from token budgets (sum = 2535M tokens after oasst reduction)
TARGET_PROPORTIONS = {
    "stack_sql":          1_100 / 2_535,
    "stack_markdown":        80 / 2_535,
    "sqale":                200 / 2_535,
    "gretelai":             100 / 2_535,
    "nstext2sql":            50 / 2_535,
    "sql_create_context":    30 / 2_535,
    "stack_python":         280 / 2_535,
    "stack_ruby":            50 / 2_535,
    "fineweb":              600 / 2_535,
    "oasst":                 15 / 2_535,
    "bird":                  30 / 2_535,
}

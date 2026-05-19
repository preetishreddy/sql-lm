"""Build the instruction-tuning dataset for fine-tuning.

Streams BIRD, sql_create_context, nstext2sql, and gretelai, formats each row
as <schema>...</schema><question>...</question><sql>...</sql>, tokenizes,
computes per-token loss masks (1 on the SQL response, 0 on prompt + padding),
applies upsampling weights, and writes train/val .npy files.

Output: /content/data/finetune/{train,val}_{tokens,mask}.npy + manifest.json

Special token IDs (pinned in tokenizer.md):
    BOS=0  EOS=1  PAD=2
    <schema>=4   </schema>=7
    <question>=5 </question>=8
    <sql>=6      </sql>=9
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

BOS, EOS, PAD = 0, 1, 2
SCHEMA_OPEN,   SCHEMA_CLOSE   = 4, 7
QUESTION_OPEN, QUESTION_CLOSE = 5, 8
SQL_OPEN,      SQL_CLOSE      = 6, 9

MAX_LEN = 512

# Upsampling weights — repeat each example this many times in the final mix.
# Based on review: BIRD 3x, nstext2sql 2x (Spider stand-in), sql_create_context 1x,
# gretelai 0.5x (only the SQL targets, not the explanations).
WEIGHTS = {
    'bird':              3.0,
    'nstext2sql':        2.0,
    'sql_create_context': 1.0,
    'gretelai':          0.5,
}

VAL_FRACTION = 0.01  # 1% of each source held out


# --------------------------------------------------------------------------
# Per-source row extractors → (schema, question, sql)
# --------------------------------------------------------------------------

def _bird_row(row):
    schema  = row.get('schema', '') or ''
    evid    = row.get('evidence', '') or ''
    # BIRD's evidence column is a hint for the question — append to it
    question = (row.get('question', '') or '').strip()
    if evid.strip():
        question = f"{question}\nHint: {evid.strip()}"
    sql = row.get('SQL', '') or ''
    return schema, question, sql


def _sql_create_context_row(row):
    return (row.get('context', '') or '',
            row.get('question', '') or '',
            row.get('answer', '') or '')


def _nstext2sql_row(row):
    # instruction field bakes schema+question; output is the SQL.
    # Heuristic: split instruction on "Question:" or fall back to using all of
    # it as the question with an empty schema. nstext2sql formats vary.
    instr  = row.get('instruction', '') or ''
    sql    = row.get('output', '') or ''
    schema = ''
    question = instr
    for marker in ['Question:', 'Q:', 'question:']:
        if marker in instr:
            parts = instr.split(marker, 1)
            schema, question = parts[0].strip(), parts[1].strip()
            break
    return schema, question, sql


def _gretelai_row(row):
    return (row.get('sql_context', '') or '',
            row.get('sql_prompt', '')  or '',
            row.get('sql', '')         or '')


SOURCES = {
    'bird': {
        'loader':    lambda: load_dataset('xu3kev/BIRD-SQL-data-train',
                                          split='train', streaming=True),
        'extractor': _bird_row,
    },
    'sql_create_context': {
        'loader':    lambda: load_dataset('b-mc2/sql-create-context',
                                          split='train', streaming=True),
        'extractor': _sql_create_context_row,
    },
    'nstext2sql': {
        'loader':    lambda: load_dataset('NumbersStation/NSText2SQL',
                                          split='train', streaming=True),
        'extractor': _nstext2sql_row,
    },
    'gretelai': {
        'loader':    lambda: load_dataset('gretelai/synthetic_text_to_sql',
                                          split='train', streaming=True),
        'extractor': _gretelai_row,
    },
}


# --------------------------------------------------------------------------
# Tokenize one (schema, question, sql) → token ids + loss mask
# --------------------------------------------------------------------------

def encode_example(tok: Tokenizer, schema: str, question: str, sql: str):
    """
    Returns (tokens, mask) both length MAX_LEN, or None if the example
    doesn't fit (no truncation — fine-tuning examples that don't fit are
    skipped rather than corrupted).

    Layout:
      [BOS][<schema>] schema [</schema>][<question>] question [</question>]
      [<sql>] sql [</sql>][EOS][PAD...]

    Loss mask is 1 on positions whose TARGET is in the SQL region
    (the first SQL content token through </sql> through EOS), 0 elsewhere.
    Mask is shifted-by-one for next-token prediction.
    """
    s_ids = tok.encode(schema).ids   if schema   else []
    q_ids = tok.encode(question).ids if question else []
    a_ids = tok.encode(sql).ids      if sql      else []

    if not a_ids:
        return None  # no target = no supervision signal

    seq = (
        [BOS, SCHEMA_OPEN] + s_ids + [SCHEMA_CLOSE]
        + [QUESTION_OPEN] + q_ids + [QUESTION_CLOSE]
        + [SQL_OPEN] + a_ids + [SQL_CLOSE, EOS]
    )

    if len(seq) > MAX_LEN:
        return None  # skip oversize examples

    # Position of <sql> in seq — everything strictly AFTER this is the response
    sql_open_pos = (
        2 + len(s_ids) + 1   # BOS, <schema>, schema..., </schema>
        + 1 + len(q_ids) + 1 # <question>, question..., </question>
    )
    # First response token's INPUT position = sql_open_pos (the <sql> token).
    # Its TARGET position (in shifted-mask space) = sql_open_pos.
    # Mask covers targets[sql_open_pos] ... targets[len(seq)-2] inclusive
    # (last target is EOS at position len(seq)-1 in seq, which is index
    # len(seq)-2 in the [:-1] targets array).
    mask = np.zeros(MAX_LEN, dtype=np.uint8)
    mask[sql_open_pos:len(seq) - 1] = 1   # 1 across SQL response + </sql> + EOS

    tokens = np.full(MAX_LEN, PAD, dtype=np.int16)
    tokens[:len(seq)] = seq

    return tokens, mask


# --------------------------------------------------------------------------
# Build the dataset
# --------------------------------------------------------------------------

def build(tokenizer_path: str, output_dir: str, max_per_source: int = None,
          verbose: bool = True):
    tok = Tokenizer.from_file(tokenizer_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tokens, train_masks = [], []
    val_tokens,   val_masks   = [], []
    stats = {}

    for name, cfg in SOURCES.items():
        if verbose: print(f"\n=== {name} (weight={WEIGHTS[name]}x) ===")
        kept, skipped = 0, 0
        try:
            stream = cfg['loader']()
        except Exception as e:
            print(f"  [skip] could not load: {e}")
            continue

        rng = np.random.default_rng(42)
        for i, row in enumerate(stream):
            if max_per_source and kept >= max_per_source:
                break
            schema, question, sql = cfg['extractor'](row)
            out = encode_example(tok, schema, question, sql)
            if out is None:
                skipped += 1
                continue
            tokens, mask = out

            target = val_tokens if rng.random() < VAL_FRACTION else train_tokens
            target_mask = val_masks if target is val_tokens else train_masks

            # Apply upsampling: integer copies + probabilistic remainder
            w = WEIGHTS[name]
            n_copies = int(w) + (1 if rng.random() < (w - int(w)) else 0)
            for _ in range(n_copies):
                target.append(tokens)
                target_mask.append(mask)

            kept += 1
            if verbose and kept % 5000 == 0:
                print(f"  kept={kept:>7,}  skipped={skipped:>6,}")

        stats[name] = {'kept': kept, 'skipped': skipped}
        if verbose: print(f"  TOTAL kept={kept:,}  skipped={skipped:,}")

    train_tokens = np.stack(train_tokens) if train_tokens else np.zeros((0, MAX_LEN), dtype=np.int16)
    train_masks  = np.stack(train_masks)  if train_masks  else np.zeros((0, MAX_LEN), dtype=np.uint8)
    val_tokens   = np.stack(val_tokens)   if val_tokens   else np.zeros((0, MAX_LEN), dtype=np.int16)
    val_masks    = np.stack(val_masks)    if val_masks    else np.zeros((0, MAX_LEN), dtype=np.uint8)

    # Shuffle train
    perm = np.random.default_rng(0).permutation(len(train_tokens))
    train_tokens, train_masks = train_tokens[perm], train_masks[perm]

    np.save(output_dir / 'train_tokens.npy', train_tokens)
    np.save(output_dir / 'train_mask.npy',   train_masks)
    np.save(output_dir / 'val_tokens.npy',   val_tokens)
    np.save(output_dir / 'val_mask.npy',     val_masks)

    manifest = {
        'train_examples': int(len(train_tokens)),
        'val_examples':   int(len(val_tokens)),
        'max_len':        MAX_LEN,
        'weights':        WEIGHTS,
        'sources':        stats,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {len(train_tokens):,} train + {len(val_tokens):,} val "
          f"examples to {output_dir}")
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', default='/content/tokenizer/tokenizer.json')
    p.add_argument('--output',    default='/content/data/finetune')
    p.add_argument('--max_per_source', type=int, default=None,
                   help='Cap examples per source (debug)')
    args = p.parse_args()
    build(args.tokenizer, args.output, args.max_per_source)

# retry_truncated.py
#
# Workflow
# --------
#   1. Read the output parquet from the initial pass.
#   2. Extract rows where valid_output=False.
#   3. Split each row's note text into the minimum number of chunks needed
#      so that each chunk fits within the token budget.
#   4. Process every chunk through vLLM (same async pool as initial pass).
#   5. Merge chunk outputs back into one output per original row.
#   6. Write a separate retry parquet — original is never touched.
#
# Output columns
# --------------
#   overall_output        str   – the best available output for every row:
#                                   · valid_output=True  → copied from `concepts`
#                                   · valid_output=False → retry_output (even if invalid)
#   overall_valid         bool  – True if overall_output is valid JSON
#
#   retry_output          str   – merged JSON array of all chunk outputs
#   retry_valid           bool  – True if merged result is valid JSON
#   retry_chunks          int   – how many chunks the note was split into
#   retry_chunk_outputs   str   – JSON list of per-chunk raw outputs (debugging)
#   retry_exhausted       bool  – True if still invalid after retry
#
# How splitting works
# -------------------
#   Estimate prompt tokens as len(text) // CHARS_PER_TOKEN.
#   Start from n_chunks=1 and increment until each chunk fits inside
#   MAX_PROMPT_TOKENS.  Cuts are made at the last \n\n, then \n before
#   the target character boundary.  Never cuts mid-word.
#
# Dead-server handling
# --------------------
#   On APIConnectionError the offending server URL is evicted from the pool
#   and the request is immediately retried on the next live server (up to 3
#   attempts).  A warning is printed showing how many servers remain live.
#
# Resume behaviour
# ----------------
#   If the retry parquet already exists, rows where retry_valid=True are
#   skipped.  Just re-run the script to continue from where it left off.
#   overall_output / overall_valid are always recomputed at save time.

import asyncio
import json
import traceback
import argparse
import itertools
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional

import pandas as pd
from openai import AsyncOpenAI, APIConnectionError
from tqdm.asyncio import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN   = 2        # conservative: 1 token ≈ 2 chars
MAX_PROMPT_TOKENS = 2000   # leaves room for output tokens inside a 16k context
MAX_CHUNKS        = 16        # hard ceiling on splits per note

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def split_at_newline(text: str, target_char: int) -> tuple[str, str]:
    """
    Split *text* into (head, tail) trying to cut at the last \n\n before
    *target_char*, then the last \n.  If no newline exists before
    *target_char*, returns (text, "") — caller must use a larger chunk size.
    """
    if target_char >= len(text):
        return text, ""

    cut = text.rfind('\n\n', 0, target_char)
    if cut == -1:
        cut = text.rfind('\n', 0, target_char)
    if cut == -1:
        return text, ""

    return text[:cut], text[cut:].lstrip('\n')


def minimum_chunks(text: str,
                   max_prompt_tokens: int = MAX_PROMPT_TOKENS,
                   chars_per_token: int   = CHARS_PER_TOKEN,
                   max_chunks: int        = MAX_CHUNKS) -> List[str]:
    """
    Return the fewest chunks such that each chunk's estimated token count
    fits within *max_prompt_tokens*.
    """
    max_chars = max_prompt_tokens * chars_per_token
    if len(text) <= max_chars:
        return [text]

    for n in range(2, max_chunks + 1):
        chunk_size = len(text) // n
        if chunk_size <= max_chars:
            chunks, remaining = [], text
            for _ in range(n - 1):
                head, remaining = split_at_newline(remaining, chunk_size)
                chunks.append(head)
                if not remaining:
                    break
            if remaining:
                chunks.append(remaining)
            return [c for c in chunks if c.strip()]

    # Fallback: force max_chunks pieces
    chunk_size = len(text) // max_chunks
    chunks, remaining = [], text
    for _ in range(max_chunks - 1):
        head, remaining = split_at_newline(remaining, chunk_size)
        chunks.append(head)
        if not remaining:
            break
    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c.strip()]


def build_prompt(note_date: str, note_text: str,
                 chunk_idx: int, total_chunks: int) -> str:
    instruction = """Extract concepts from the following clinical note. Determine event time from context. 
If date (year-mon-day) is incomplete, use only the known information, use 0000 or 00 for the rest parts.
When date is not available, determine if it is a current event or historical event. If current event, use the note date, otherwise use "unknown".\n\n"""
    date_line   = f'Note date (use this as the current event date): {note_date}.\n\n '
    return instruction + date_line + note_text


def check_output(text: Optional[str]) -> bool:
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def merge_chunk_outputs(outputs: List[str]) -> tuple[str, bool]:
    """
    Flatten multiple JSON outputs into a single JSON array.
    If any chunk is invalid JSON, returns (raw_bad_output, False) immediately
    so the caller treats the whole row as invalid.
    Returns (merged_json, True) only when every chunk is valid.
    """
    merged = []
    for raw in outputs:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                merged.extend(parsed)
            else:
                merged.append(parsed)
        except Exception:
            return raw, False   # short-circuit: any bad chunk -> whole row invalid
    return json.dumps(merged, ensure_ascii=False), True


def compute_overall(df: pd.DataFrame,
                    initial_output_col: str,
                    valid_col: str) -> pd.DataFrame:
    """
    Populate overall_output and overall_valid:
      - valid_output=True  → copy from initial_output_col (already valid)
      - valid_output=False → copy from retry_output (may still be invalid)
    overall_valid reflects whether overall_output is valid JSON.
    """
    df["overall_output"] = None
    df["overall_valid"]  = False

    initial_ok = df[valid_col] == True
    df.loc[initial_ok, "overall_output"] = df.loc[initial_ok, initial_output_col]
    df.loc[initial_ok, "overall_valid"]  = True

    retried = df[valid_col] == False
    df.loc[retried, "overall_output"] = df.loc[retried, "retry_output"]
    df.loc[retried, "overall_valid"]  = df.loc[retried, "retry_valid"]

    return df


# ---------------------------------------------------------------------------
# vLLM client pool with dead-server eviction
# ---------------------------------------------------------------------------

class VLLMPool:
    def __init__(self, num_servers: int = 8, base_port: int = 8000,
                 max_concurrent: int = 256, host: str = "127.0.0.1"):
        ports           = [1]
        self._all_urls  = [f"http://{host}:{base_port + i}/v1" for i in ports]
        self._dead      : set[str] = set()
        self._lock      = asyncio.Lock()
        self._clients   = {url: AsyncOpenAI(base_url=url, api_key="EMPTY")
                           for url in self._all_urls}
        self._cycle     = itertools.cycle(self._all_urls)
        self.semaphore  = asyncio.Semaphore(max_concurrent)

    def _next_live(self) -> tuple[str, AsyncOpenAI]:
        for _ in range(len(self._all_urls)):
            url = next(self._cycle)
            if url not in self._dead:
                return url, self._clients[url]
        raise RuntimeError("All vLLM servers are unreachable.")

    async def _evict(self, url: str):
        async with self._lock:
            if url not in self._dead:
                self._dead.add(url)
                live = len(self._all_urls) - len(self._dead)
                print(f"\n⚠  Server {url} evicted after connection error. "
                      f"Live servers remaining: {live}")

    async def call(self, model: str, prompt: str, max_tokens: int = 2048) -> Dict:
        """
        Call the LLM.  On APIConnectionError: evict the server, retry on
        the next live one (up to 3 attempts total).
        """
        async with self.semaphore:
            last_err = None
            for _ in range(3):
                try:
                    url, client = self._next_live()
                except RuntimeError as e:
                    return {"output": None, "error": str(e)}
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                    )
                    return {"output": resp.choices[0].message.content, "error": None}
                except APIConnectionError as e:
                    last_err = e
                    await self._evict(url)
                except Exception as e:
                    traceback.print_exc()
                    return {"output": None, "error": str(e)}

            return {"output": None,
                    "error": f"All retry attempts failed: {last_err}"}


# ---------------------------------------------------------------------------
# Row-level retry
# ---------------------------------------------------------------------------

async def retry_row(pool: VLLMPool, model: str, row: pd.Series,
                    max_tokens: int = 2048) -> Dict:
    note_date = str(row.get('CHARTDATE', ''))
    note_text = str(row.get('TEXT', ''))

    chunks  = minimum_chunks(note_text)
    n       = len(chunks)
    prompts = [build_prompt(note_date, chunk, i, n) for i, chunk in enumerate(chunks)]
    results       = await asyncio.gather(*[pool.call(model, p, max_tokens) for p in prompts])
    chunk_outputs = [r["output"] or "" for r in results]
    if n == 1:
        merged, valid = chunk_outputs[0], check_output(chunk_outputs[0])
    else:
        merged, valid = merge_chunk_outputs(chunk_outputs)

    return {
        "retry_output":        merged,
        "retry_valid":         valid,
        "retry_chunks":        n,
        "retry_chunk_outputs": json.dumps(chunk_outputs, ensure_ascii=False),
        "retry_prompt":        json.dumps(prompts, ensure_ascii=False),
        "retry_exhausted":     not valid,
    }


# ---------------------------------------------------------------------------
# File-level orchestration
# ---------------------------------------------------------------------------

async def process_file(pool: VLLMPool, model: str,
                       input_file: Path, output_file: Path,
                       valid_col: str          = "valid_output",
                       initial_output_col: str = "concepts",
                       max_tokens: int         = 2048):
    print(f"\n{'='*60}")
    print(f"Input  : {input_file}")
    print(f"Output : {output_file}")
    print(f"{'='*60}")

    df = pd.read_parquet(input_file)

    for col, default in [("retry_output", None), ("retry_valid", False),
                         ("retry_chunks", 0), ("retry_chunk_outputs", None),
                         ("retry_prompt", None), ("retry_exhausted", False)]:
        if col not in df.columns:
            df[col] = default

    # Resume
    if output_file.exists():
        print("Resuming from existing retry parquet...")
        prev      = pd.read_parquet(output_file)
        done_mask = prev["retry_valid"] == True
        for col in ["retry_output", "retry_valid", "retry_chunks",
                    "retry_chunk_outputs", "retry_prompt", "retry_exhausted"]:
            if col in prev.columns:
                df.loc[done_mask, col] = prev.loc[done_mask, col]
        print(f"  Already resolved: {done_mask.sum()} rows")

    todo_idx = df[(df[valid_col] == False) & (df["retry_valid"] == False)].index.tolist()

    if not todo_idx:
        print("Nothing to retry.")
        _save_and_report(df, output_file, valid_col, initial_output_col)
        return

    print(f"Rows to retry : {len(todo_idx)} "
          f"(of {(df[valid_col] == False).sum()} failed in input)")

    results = await tqdm.gather(
        *[retry_row(pool, model, df.loc[idx], max_tokens) for idx in todo_idx],
        desc="Retrying rows"
    )

    for idx, res in zip(todo_idx, results):
        for col, val in res.items():
            df.at[idx, col] = val

    _save_and_report(df, output_file, valid_col, initial_output_col)


def _save_and_report(df: pd.DataFrame, output_file: Path,
                     valid_col: str, initial_output_col: str):
    df = compute_overall(df, initial_output_col, valid_col)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file)

    n_total        = len(df)
    n_initial_ok   = (df[valid_col] == True).sum()
    n_failed       = (df[valid_col] == False).sum()
    n_retry_ok     = (df["retry_valid"] == True).sum()
    n_exhausted    = (df["retry_exhausted"] == True).sum()
    n_overall_ok   = (df["overall_valid"] == True).sum()
    dist           = (df.loc[df[valid_col] == False, "retry_chunks"]
                      .value_counts().sort_index())

    print(f"\n── Results ──────────────────────────────────────────────")
    print(f"  Total rows               : {n_total}")
    print(f"  Initial pass valid       : {n_initial_ok}")
    print(f"  Initial pass failed      : {n_failed}")
    print(f"    Resolved by retry      : {n_retry_ok}")
    print(f"    Still exhausted        : {n_exhausted}")
    print(f"  Overall valid            : {n_overall_ok} / {n_total}")
    print(f"\n  Chunk distribution (retried rows):")
    for n, count in dist.items():
        label = "1 chunk (no split)" if n == 1 else f"{n} chunks"
        print(f"    {label}: {count} rows")
    print(f"\n  Saved to: {output_file}")


# ---------------------------------------------------------------------------
# Multi-file wrapper
# ---------------------------------------------------------------------------

async def process_multiple_files(pool: VLLMPool, model: str,
                                  input_pattern: str, output_dir: Path,
                                  valid_col: str          = "valid_output",
                                  initial_output_col: str = "concepts",
                                  max_tokens: int         = 2048,
                                  max_parallel_files: int = 4,
                                  preserve_structure: bool = True):
    input_files = [Path(f) for f in glob(input_pattern, recursive=True)]
    if not input_files:
        print(f"No parquet files found: {input_pattern}")
        return

    print(f"Found {len(input_files)} file(s) to retry")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_sem = asyncio.Semaphore(max_parallel_files)

    async def run(f: Path):
        async with file_sem:
            out = (output_dir / f.parent.name / f.name) if preserve_structure \
                  else (output_dir / f.name)
            await process_file(pool, model, f, out,
                               valid_col, initial_output_col, max_tokens)

    await asyncio.gather(*[run(f) for f in input_files])
    print(f"\nAll retry outputs saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Retry valid_output=False rows with note splitting."
    )
    parser.add_argument('--model',                 type=str, required=True)
    parser.add_argument('--input-pattern',          type=str, required=True)
    parser.add_argument('--output-dir',             type=str, required=True)
    parser.add_argument('--valid-column',           type=str, default='valid_output')
    parser.add_argument('--initial-output-column',  type=str, default='concepts',
                        help='Column in the initial-pass parquet holding the LLM output.')
    parser.add_argument('--max-tokens',             type=int, default=2048)
    parser.add_argument('--num-servers',            type=int, default=8)
    parser.add_argument('--max-concurrent',         type=int, default=256)
    parser.add_argument('--max-parallel-files',     type=int, default=4)
    parser.add_argument('--preserve-structure',        action='store_true')
    parser.add_argument('--host',                   type=str, default='127.0.0.1')

    args = parser.parse_args()

    pool = VLLMPool(
        num_servers=args.num_servers,
        max_concurrent=args.max_concurrent,
        host=args.host,
    )

    asyncio.run(process_multiple_files(
        pool=pool,
        model=args.model,
        input_pattern=args.input_pattern,
        output_dir=Path(args.output_dir),
        valid_col=args.valid_column,
        initial_output_col=args.initial_output_column,
        max_tokens=args.max_tokens,
        max_parallel_files=args.max_parallel_files,
        preserve_structure=args.preserve_structure,
    ))


if __name__ == "__main__":
    main()
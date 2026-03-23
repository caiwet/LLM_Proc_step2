import pandas as pd
import json
import re
from tqdm import tqdm


# ─────────────────────────────── helpers ─────────────────────────────────────

def is_valid_id(id_str: str) -> bool:
    """True only when id_str matches the strict pattern  r<int>_<int>."""
    return bool(re.fullmatch(r'r\d+_\d+', str(id_str)))


def parse_id(id_str: str):
    """Return (first_num, second_num) for a valid id, else None."""
    m = re.fullmatch(r'r(\d+)_(\d+)', str(id_str))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _patch_reference(ref: str, id_map: dict) -> str:
    """
    Update the ID portion of a reference string.
    e.g. 'Encounter/r138_0'  →  'Encounter/r0_0'
    """
    for old, new in id_map.items():
        ref = ref.replace(f'/{old}', f'/{new}')
    return ref


def _apply_id_map(obj, id_map: dict):
    """
    Recursively walk a parsed-JSON structure and apply *id_map* to:
      - every "id" field value
      - every "reference" field value (ResourceType/<id>)
    """
    if isinstance(obj, dict):
        return {
            k: (id_map.get(v, v)                # plain "id" field
                if k == 'id' and isinstance(v, str)
                else _patch_reference(v, id_map) # "reference" field
                if k == 'reference' and isinstance(v, str)
                else _apply_id_map(v, id_map))   # everything else
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_apply_id_map(item, id_map) for item in obj]
    return obj


# ─────────────────────────────── main ────────────────────────────────────────

def process_resources_column(df: pd.DataFrame, col: str):
    """
    Validate and (where needed) renumber resource IDs stored as JSON
    strings in *col*.

    Logic
    -----
    1. Parse the JSON string into a list of resource dicts.
    2. If any 'id' does NOT match r<int>_<int>  →  row goes to df_invalid.
    3. If the minimum leading number across all IDs > 0  →  renumber so that
       sorted unique leading numbers map to 0, 1, 2, …
       References  (ResourceType/<id>)  are updated as well.

    Returns
    -------
    df_clean   : rows with valid, zero-based IDs
    df_invalid : rows with bad IDs + an 'invalid_reason' column
    """
    df = df.copy()
    bad_idx    = []
    bad_reason = {}

    for idx in tqdm(df.index, desc=f"Processing {col}", total=len(df.index)):
        val = df.at[idx, col]

        # ── 1. parse ──────────────────────────────────────────────────────────
        try:
            resources = json.loads(val) if isinstance(val, str) else list(val)
        except Exception as exc:
            bad_idx.append(idx)
            bad_reason[idx] = f"JSON parse error: {exc}"
            continue

        if not isinstance(resources, list):
            bad_idx.append(idx)
            bad_reason[idx] = "top-level value is not a JSON array"
            continue

        # ── 2. detect invalid IDs ─────────────────────────────────────────────
        bad_ids = [
            r['id']
            for r in resources
            if isinstance(r, dict) and 'id' in r and not is_valid_id(r['id'])
        ]
        if bad_ids:
            bad_idx.append(idx)
            bad_reason[idx] = f"invalid id(s): {bad_ids}"
            continue

        # ── 3. decide if renumbering is needed ────────────────────────────────
        all_ids  = [r['id'] for r in resources if isinstance(r, dict) and 'id' in r]
        parsed   = [p for p in map(parse_id, all_ids) if p is not None]

        if not parsed:
            continue                         # no IDs at all → nothing to do

        unique_fn = sorted({fn for fn, _ in parsed})   # unique leading numbers

        if unique_fn[0] == 0:
            continue                         # already starts at r0_x → skip

        # ── 4. build renaming map and apply it ────────────────────────────────
        fn_remap = {old: new for new, old in enumerate(unique_fn)}
        id_map   = {
            f"r{fn}_{sn}": f"r{fn_remap[fn]}_{sn}"
            for fn, sn in parsed
        }

        df.at[idx, col] = json.dumps(_apply_id_map(resources, id_map))

    # ── assemble output DataFrames ────────────────────────────────────────────
    df_invalid = df.loc[bad_idx].copy()
    df_invalid.insert(0, 'invalid_reason', df_invalid.index.map(bad_reason))

    df_clean = df.drop(index=bad_idx).reset_index(drop=True)

    return df_clean, df_invalid

if __name__ == "__main__":
    df = pd.read_parquet("shard_0001/final_full_merged.parquet")
    df_clean, df_invalid = process_resources_column(df, 'resources')
    breakpoint()
    assert len(df_invalid) == 0, f"Found {len(df_invalid)} invalid rows:\n{df_invalid[['invalid_reason']].head()}"

    # df_clean.to_parquet("step2_gemini_output/final_full_merged_remapped.parquet", index=False)

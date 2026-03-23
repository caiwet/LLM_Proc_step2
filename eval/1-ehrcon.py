import pickle
import pandas as pd
import os
from pathlib import Path

base = '/data/bwh-comppath-img2/MGH_CID/public_dataset/physionet.org/files/ehrcon-consistency-of-notes/1.0.1'
os.chdir(base)
KEEP_FIELDS = [
    "amount", "amountuom",
    "charttime", "chartdate",
    "dose_val_rx", "dose_unit_rx",
    "drug", "form_unit_disp", 
    "org_name", "originalroute", 
    "rate", "rateuom", "route", 
    "spec_type_desc",
    "value", "valuenum", "valueuom"
]

IGNORE_KEYS = {
    "label", "errors",
    "d_items", "d_labitems", "d_icd_diagnoses", "d_icd_procedures"
}

def clean(x):
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"nan", "none", ""}:
        return None
    return x

def uniq_preserve_order(vals):
    seen = set()
    out = []
    for v in vals:
        key = repr(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out

def flatten_note_only(pkl_file, list_sep=" | "):
    with open(pkl_file, "rb") as f:
        obj = pickle.load(f)
    
    rows = []

    for row_id, entity_list in obj.items():
        for item in entity_list:
            if not isinstance(item, dict) or len(item) != 1:
                continue

            entity_text, meta = next(iter(item.items()))
            position = meta.get("position")
            entity_type = meta.get("entity_type")

            data_entries = meta.get("data", [])

            # If no extracted fields, still keep the entity mention
            if not data_entries:
                row = {
                    "ROW_ID": row_id,
                    "position": position,
                    "entity_text": entity_text,
                    "entity_type": entity_type,
                }
                for f in KEEP_FIELDS:
                    row[f] = None
                rows.append(row)
                continue

            field_values = {f: [] for f in KEEP_FIELDS}
            for entry in data_entries:
                for k, v in entry.items():
                    if k in IGNORE_KEYS:
                        continue
                    if isinstance(v, dict):
                        for subk, subv in v.items():
                            if subk in KEEP_FIELDS:
                                subv = clean(subv)
                                if subv is not None:
                                    field_values[subk].append(subv)
               

            row = {
                "ROW_ID": row_id,
                "position": position,
                "entity_text": entity_text,
                "entity_type": entity_type,
            }

            for f in KEEP_FIELDS:
                uniq_vals = uniq_preserve_order(field_values[f])
                row[f] = list_sep.join(map(str, uniq_vals)) if uniq_vals else None

            rows.append(row)

    df = pd.DataFrame(rows)

    # optional de-dup
    dedup_cols = ["ROW_ID", "position", "entity_text", "entity_type"] + KEEP_FIELDS
    df = df.drop_duplicates(subset=dedup_cols)

    return df

base_dir = Path("/data/bwh-comppath-img2/MGH_CID/public_dataset/physionet.org/files/ehrcon-consistency-of-notes/1.0.1/original")  
pkl_paths = list(base_dir.rglob("*.pkl"))

# dfs = []
# for pkl_file in pkl_paths:
    
#     cat = pkl_file.parent.parent.name
#     df = flatten_note_only(pkl_file)
#     df['category'] = cat
#     dfs.append(df)

# dfs = pd.concat(dfs, ignore_index=True)
# dfs.to_parquet("/data/bwh-comppath-img2/MGH_CID/LLM_Proc/benchmark/ehrcon_all_notes_flattened.parquet", index=False)


notes_paths = list(base_dir.rglob("*.csv"))
note_dfs = []
for notes_path in notes_paths:
    df = pd.read_csv(notes_path)
    cat = notes_path.parent.name
    df['category'] = cat
    note_dfs.append(df)
notes_df = pd.concat(note_dfs, ignore_index=True)
breakpoint()
notes_df.to_parquet("/data/bwh-comppath-img2/MGH_CID/LLM_Proc/benchmark/ehrcon_all_notes.parquet", index=False)
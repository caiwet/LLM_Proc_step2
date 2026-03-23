import pickle
from pathlib import Path
from collections import defaultdict

IGNORE_KEYS = {"label", "errors"}

def scan_fields(pkl_paths):
    table_to_fields = defaultdict(set)
    all_fields = set()
    tables = set()

    for pkl_file in pkl_paths:
        with open(pkl_file, "rb") as f:
            obj = pickle.load(f)

        for note_id, entity_list in obj.items():
            for item in entity_list:
                if not isinstance(item, dict) or len(item) != 1:
                    continue

                entity_text, meta = next(iter(item.items()))
                data_entries = meta.get("data", [])

                for entry in data_entries:
                    for k, v in entry.items():
                        if k in IGNORE_KEYS:
                            continue
                        if not isinstance(v, dict):
                            continue

                        tables.add(k)

                        for subk in v.keys():
                            if subk == 'label':
                                breakpoint()
                            table_to_fields[k].add(subk)
                            all_fields.add(subk)

    return tables, table_to_fields, all_fields


# ---- run on your dataset ----

base_dir = Path("/data/bwh-comppath-img2/MGH_CID/public_dataset/physionet.org/files/ehrcon-consistency-of-notes/1.0.1/original")  
pkl_paths = list(base_dir.rglob("*.pkl"))

tables, table_to_fields, all_fields = scan_fields(pkl_paths)

print("\n=== Tables found ===")
for t in sorted(tables):
    print(t)

print("\n=== Fields per table ===")
for t in sorted(table_to_fields):
    print(f"\n{t}:")
    for f in sorted(table_to_fields[t]):
        print(f"  - {f}")

print("\n=== ALL UNIQUE FIELDS ===")
for f in sorted(all_fields):
    print(f)



import os
import math
from collections import Counter

import pandas as pd


# -----------------------------
# Config
# -----------------------------
anno_path = "/data/bwh-comppath-img2/MGH_CID/LLM_Proc/benchmark/ehrcon_all_notes_flattened.parquet"
pred_dir = "/data/bwh-comppath-img2/MGH_CID/LLM_Proc/benchmark/qwen3-4b-instruct_bs8_lr3e-5/reformat"

output_dir = os.path.join(os.path.dirname(pred_dir), "metrics")
os.makedirs(output_dir, exist_ok=True)
row_metrics_out = os.path.join(output_dir, "recall_metrics_per_row.csv")
summary_out = os.path.join(output_dir, "recall_metrics_summary.csv")
overall_out = os.path.join(output_dir, "overall_merged.parquet")

# -----------------------------
# Normalization helpers
# -----------------------------
def normalize_text(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    return x if x != "" else None


def normalize_value(x):
    """
    Conservative normalization:
    - keep missing as None
    - strip/lower strings
    - normalize numeric-looking values so 1, 1.0, 1.00 match
    """
    if pd.isna(x):
        return None

    # Already numeric
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if isinstance(x, float) and math.isnan(x):
            return None
        if float(x).is_integer():
            return str(int(float(x)))
        return str(float(x)).rstrip("0").rstrip(".")

    s = str(x).strip().lower()
    if s == "":
        return None

    # Try numeric normalization
    try:
        f = float(s)
        if math.isnan(f):
            return None
        if f.is_integer():
            return str(int(f))
        return str(f).rstrip("0").rstrip(".")
    except Exception:
        return s


# -----------------------------
# Load gold annotations
# -----------------------------
anno_df = pd.read_parquet(anno_path).copy()
anno_df["values"] = anno_df["value"].fillna(anno_df["valuenum"])
anno_df["entity_text"] = anno_df["entity_text"].map(normalize_text)
anno_df["values"] = anno_df["values"].map(normalize_value)

# keep position in case you want to inspect duplicates later
anno_df = anno_df[["ROW_ID", "position", "entity_text", "entity_type", "values"]]
anno_df = anno_df[anno_df["entity_text"].notna()].copy()

# -----------------------------
# Metric computation
# -----------------------------
rows = []

global_total_gold = 0
global_entity_matched = 0
global_value_matched = 0
num_missing_pred_files = 0

overall_merged_df = []

for row_id in sorted(anno_df["ROW_ID"].unique()):
    pred_path = os.path.join(pred_dir, f"{row_id}_codes.parquet")
    label_df = anno_df[anno_df["ROW_ID"] == row_id].copy()

    total_gold = len(label_df)

    if not os.path.exists(pred_path):
        print(f"Missing prediction for ROW_ID {row_id}")
        rows.append(
            {
                "ROW_ID": row_id,
                "total_gold": total_gold,
                "total_pred": 0,
                "entity_matched": 0,
                "value_matched": 0,
                "entity_recall": 0.0 if total_gold > 0 else None,
                "value_accuracy_given_entity_match": None,
                "complete_fact_recall": 0.0 if total_gold > 0 else None,
                "missing_prediction_file": True,
            }
        )
        global_total_gold += total_gold
        num_missing_pred_files += 1
        continue

    pred_df = pd.read_parquet(pred_path).copy()
    pred_df["description"] = pred_df["description"].map(normalize_text)
    pred_df["values"] = pred_df["valueQuantity_value"].fillna(pred_df["valueString"])
    pred_df["values"] = pred_df["values"].map(normalize_value)
    pred_df = pred_df[["description", "values"]]
    pred_df = pred_df[pred_df["description"].notna()].copy()

    total_pred = len(pred_df)

    merged_df = label_df.merge(pred_df, left_on="entity_text", right_on="description", how="left", suffixes=("_gold", "_pred"))
    overall_merged_df.append(merged_df)

    # Count gold and predicted mentions per entity_text
    gold_entity_counts = label_df["entity_text"].value_counts().to_dict()
    pred_entity_counts = pred_df["description"].value_counts().to_dict()

    entity_matched = 0
    value_matched = 0

    # Evaluate only against gold entities (recall-only setting)
    for entity_text, gold_count in gold_entity_counts.items():
        pred_count = pred_entity_counts.get(entity_text, 0)

        # Entity recall at mention count level:
        # a gold mention is recovered only if there is an available predicted mention
        entity_matched += min(gold_count, pred_count)

        # Value matching:
        # compare multisets of normalized values for this entity_text
        gold_vals = label_df.loc[label_df["entity_text"] == entity_text, "values"].tolist()
        pred_vals = pred_df.loc[pred_df["description"] == entity_text, "values"].tolist()

        gold_counter = Counter(gold_vals)
        pred_counter = Counter(pred_vals)

        # Maximum number of exact value matches for this entity
        for val, gold_val_count in gold_counter.items():
            value_matched += min(gold_val_count, pred_counter.get(val, 0))

    entity_recall = entity_matched / total_gold if total_gold > 0 else None
    value_acc_given_entity = value_matched / entity_matched if entity_matched > 0 else 0
    complete_fact_recall = value_matched / total_gold if total_gold > 0 else None

    print(
        f"ROW_ID {row_id}: "
        f"entity_recall={entity_matched}/{total_gold}={entity_recall:.4f} | "
        f"value_acc_given_entity={value_matched}/{entity_matched}={value_acc_given_entity:.4f} | "
        f"complete_fact_recall={value_matched}/{total_gold}={complete_fact_recall:.4f}"
    )

    rows.append(
        {
            "ROW_ID": row_id,
            "total_gold": total_gold,
            "total_pred": total_pred,
            "entity_matched": entity_matched,
            "value_matched": value_matched,
            "entity_recall": entity_recall,
            "value_accuracy_given_entity_match": value_acc_given_entity,
            "complete_fact_recall": complete_fact_recall,
            "missing_prediction_file": False,
        }
    )

    global_total_gold += total_gold
    global_entity_matched += entity_matched
    global_value_matched += value_matched

overall_merged_df = pd.concat(overall_merged_df, ignore_index=True)
overall_merged_df.to_parquet(overall_out, index=False)

# -----------------------------
# Aggregate metrics
# -----------------------------
metrics_df = pd.DataFrame(rows)

micro_entity_recall = (
    global_entity_matched / global_total_gold if global_total_gold > 0 else None
)
micro_value_acc_given_entity = (
    global_value_matched / global_entity_matched if global_entity_matched > 0 else None
)
micro_complete_fact_recall = (
    global_value_matched / global_total_gold if global_total_gold > 0 else None
)

macro_entity_recall = metrics_df["entity_recall"].dropna().mean()
macro_value_acc_given_entity = metrics_df["value_accuracy_given_entity_match"].dropna().mean()
macro_complete_fact_recall = metrics_df["complete_fact_recall"].dropna().mean()

summary_df = pd.DataFrame(
    [
        {
            "n_rows": len(metrics_df),
            "n_rows_missing_prediction_file": num_missing_pred_files,
            "global_total_gold": global_total_gold,
            "global_entity_matched": global_entity_matched,
            "global_value_matched": global_value_matched,
            "micro_entity_recall": micro_entity_recall,
            "micro_value_accuracy_given_entity_match": micro_value_acc_given_entity,
            "micro_complete_fact_recall": micro_complete_fact_recall,
            "macro_entity_recall": macro_entity_recall,
            "macro_value_accuracy_given_entity_match": macro_value_acc_given_entity,
            "macro_complete_fact_recall": macro_complete_fact_recall,
        }
    ]
)

metrics_df.to_csv(row_metrics_out, index=False)
summary_df.to_csv(summary_out, index=False)

print("\n=== Overall Summary ===")
print(summary_df.to_string(index=False))

print(f"\nSaved per-row metrics to: {row_metrics_out}")
print(f"Saved summary metrics to: {summary_out}")
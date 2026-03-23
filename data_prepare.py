import pandas as pd
import json
from sklearn.model_selection import GroupShuffleSplit

shard0 = pd.read_parquet('/data/bwh-comppath-img2/MGH_CID/LLM_Proc/for_caiwei/step2_shard_0000/final_full_merged.parquet')
shard1 = pd.read_parquet('/data/bwh-comppath-img2/MGH_CID/LLM_Proc/for_caiwei/step2_shard_0001/final_full_merged.parquet')
shard0 = shard0[['patient_id_x', 'note_time_use', 'text', 'resources']]
shard1 = shard1[['patient_id_x', 'note_time_use', 'text', 'resources']]

df = pd.concat([shard0, shard1], ignore_index=True)  # Combine the two shards
df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)  # Remove duplicates based on 'text' column

# Assuming your dataframe is called 'df'
# Split by patient_id (80/20 split)
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, eval_idx = next(gss.split(df, groups=df['patient_id_x']))

# Create train and eval dataframes
train_df = df.iloc[train_idx]
eval_df = df.iloc[eval_idx]

# Verify no patient_id overlap (optional check)
print(f"Train patients: {train_df['patient_id_x'].nunique()}")
print(f"Eval patients: {eval_df['patient_id_x'].nunique()}")
print(f"Overlap: {set(train_df['patient_id_x']) & set(eval_df['patient_id_x'])}")

# Function to save as JSONL
def save_jsonl(dataframe, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in dataframe.iterrows():
            instruction = """Extract concepts from the following clinical note. Determine event time from note using this priority order:
  1. Explicit date in note → use exact date
  2. Calculable relative date → calculate from reference date
  3. Current observation/symptom → note date
  4. Historical event, date unclear → "unknown"
  5. Future plan → when documented (e.g. note date)
If date (year-mon-day) is incomplete, use only the known information, use 0000 or 00 for the rest parts.
When date is not available, determine if it is a current event or historical event. If current event, use the note date, otherwise use "unknown".\n\n"""
            
            note_date   = f'Note date (use this as the current event date): {row['note_time_use']}.\n\n '
            note_text = row['text']
            json_obj = {
                "input": instruction + note_date + note_text,
                "output": row['resources']
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

# Save to JSONL files
save_jsonl(train_df, 'step2_gemini_output/train.jsonl')
save_jsonl(eval_df, 'step2_gemini_output/eval.jsonl')

print(f"Train set: {len(train_df)} rows")
print(f"Eval set: {len(eval_df)} rows")
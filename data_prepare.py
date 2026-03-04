import pandas as pd
import json
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_parquet('step2_gemini_output/final.parquet')

# Assuming your dataframe is called 'df'
# Split by patient_id (80/20 split)
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, eval_idx = next(gss.split(df, groups=df['patient_id']))

# Create train and eval dataframes
train_df = df.iloc[train_idx]
eval_df = df.iloc[eval_idx]

# Verify no patient_id overlap (optional check)
print(f"Train patients: {train_df['patient_id'].nunique()}")
print(f"Eval patients: {eval_df['patient_id'].nunique()}")
print(f"Overlap: {set(train_df['patient_id']) & set(eval_df['patient_id'])}")

# Function to save as JSONL
def save_jsonl(dataframe, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in dataframe.iterrows():
            json_obj = {
                "input": row['prompt'],
                "output": row['validated_output']
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

# Save to JSONL files
save_jsonl(train_df, 'step2_gemini_output/train.jsonl')
save_jsonl(eval_df, 'step2_gemini_output/eval.jsonl')

print(f"Train set: {len(train_df)} rows")
print(f"Eval set: {len(eval_df)} rows")
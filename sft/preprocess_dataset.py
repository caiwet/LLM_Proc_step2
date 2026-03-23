"""
Pre-tokenize dataset and save with input_ids and labels.
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import sys
# sys.path.append('/data/bwh-comppath-img2/MGH_CID/6-llm/2-multi-stage-pipeline')  # Add project root to path
# from step2_prompt import SYSTEM_PROMPT 

def preprocess_dataset(
    input_path: str,
    output_path: str,
    model_name: str,
    max_seq_length: int,
    # system_prompt: str,
    input_field: str = "input",
    output_field: str = "output",
):
    """Pre-tokenize dataset with instruction masking."""
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading dataset: {input_path}")
    if Path(input_path).exists():
        dataset = load_dataset("json", data_files=input_path, split="train")
    else:
        dataset = load_dataset(input_path, split="train")
    
    print(f"Original dataset size: {len(dataset)}")
    
    # Templates for Qwen
    if "qwen" in model_name.lower() or "lfm" in model_name.lower():
        response_template = "<|im_start|>assistant\n"
    elif "gemma" in model_name.lower():
        response_template = "<start_of_turn>model\n"
    else:
        raise NotImplementedError(f"Instruction masking template not defined for model: {model_name}")
    
    def tokenize_function(item):
        """Format, tokenize, and create labels with instruction masking."""
        input_text = item[input_field]
        output_text = item[output_field]
        
        # Format with chat template
        if tokenizer.chat_template is not None:
            if "qwen" in model_name.lower() or "lfm" in model_name.lower():
                messages = [
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": output_text}
                ]
            elif "gemma" in model_name.lower():
                messages = [
                    # {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "text", "text": input_text}]},
                    {"role": "assistant", "content": [{"type": "text", "text": output_text}]}
                ]
            formatted_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            NotImplementedError("Chat template is required for instruction masking")
        
        # Tokenize without truncation, filter after
        encoding = tokenizer(
            formatted_text,
            truncation=False,
            padding=False,  # Don't pad during preprocessing
        )
        
        input_ids = encoding["input_ids"]
        
        # Create labels with instruction masking
        # Find where the response starts
        response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        
        labels = input_ids.copy()
        
        # Mask everything before the response (set to -100)
        for i in range(len(input_ids) - len(response_token_ids)):
            if input_ids[i:i+len(response_token_ids)] == response_token_ids:
                # Mask everything before and including the response template
                for j in range(i + len(response_token_ids)):
                    labels[j] = -100
                break
        
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
            "length": len(input_ids),
        }
    
    # Tokenize with progress bar
    print("Tokenizing dataset...")
    dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=16
    )
    
    # Filter out sequences that are too long
    original_size = len(dataset)
    dataset = dataset.filter(
        lambda x: x["length"] <= max_seq_length,
        desc="Filtering long examples",
        num_proc=16
    )
    filtered_count = original_size - len(dataset)
    print(f"Filtered {filtered_count} examples ({filtered_count/original_size*100:.1f}%) exceeding {max_seq_length} tokens")
    
    print(f"Final dataset size: {len(dataset)}")
    
    # Print token length stats
    lengths = dataset["length"]
    print(f"\nToken length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f}")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]}")
    
    # Remove the length column before saving
    dataset = dataset.remove_columns(["length"])
    
    # Save as Arrow format (much faster to load than JSON)
    print(f"\nSaving to: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as Arrow dataset or parquet
    if output_path.endswith('.arrow'):
        dataset.save_to_disk(output_path)
        print(f"Saved as Arrow dataset")
    elif output_path.endswith('.parquet'):
        dataset.to_parquet(output_path)
        print(f"Saved as Parquet file")
    else:
        # Default to Arrow format
        dataset.save_to_disk(output_path)
        print(f"Saved as Arrow dataset")
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, help="Input dataset path")
    parser.add_argument("--output-path", required=True, help="Output dataset path (.arrow or .parquet)")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name for tokenizer")
    parser.add_argument("--max-seq-length", type=int, default=16384, help="Maximum sequence length")
    parser.add_argument("--input-field", default="input", help="Input field name")
    parser.add_argument("--output-field", default="output", help="Output field name")
    
    args = parser.parse_args()
    
    preprocess_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        input_field=args.input_field,
        output_field=args.output_field,
        # system_prompt=SYSTEM_PROMPT
    )
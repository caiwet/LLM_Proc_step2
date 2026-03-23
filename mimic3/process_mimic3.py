# process_parquet_vllm.py

import asyncio
import pandas as pd
from openai import AsyncOpenAI
from pathlib import Path
import itertools
from tqdm.asyncio import tqdm
import json
from typing import Dict
import argparse
from glob import glob
from datetime import datetime, timedelta
import traceback

def build_prompt(row: pd.Series) -> str:
    instruction = """Extract concepts from the following clinical note. Determine event time from context. 
        If date (year-mon-day) is incomplete, use only the known information, use 0000 or 00 for the rest parts.
        When date is not available, determine if it is a current event or historical event. If current event, use the note date, otherwise use "unknown".\n\n"""
    note_date   = f'Note date (use this as the current event date): {row['CHARTDATE']}.\n'
    note_text = row['TEXT']
    prompt = instruction + note_date + note_text
    return prompt

def check_output(concepts):
    try:
        json.loads(concepts)
        return True
    except:
        return False

class VLLMParquetProcessor:
    def __init__(self, num_servers=8, base_port=8000, max_concurrent=256, host="0.0.0.0"):
        """Initialize processor with 8 vLLM servers"""
        self.servers = [f"http://{host}:{base_port + i}/v1" for i in range(num_servers)]
        # self.servers = [f"http://{host}:{base_port + i}/v1" for i in [0, 1, 4, 6, 7]]
        self.clients = [AsyncOpenAI(base_url=s, api_key="EMPTY") for s in self.servers]
        self.client_cycle = itertools.cycle(self.clients)
        self.max_concurrent = max_concurrent
        
    async def process_row(self, model, prompt: str, last_max_tokens: int, row_idx: int, semaphore: asyncio.Semaphore) -> Dict:
        """Process a single row"""
        async with semaphore:
            client = next(self.client_cycle)
            
            try:
                if last_max_tokens > 0:
                    context_limit = 16384
                    estimated_prompt_tokens = len(prompt) // 4
                    safety_margin = 256
                    max_tokens = min(int(last_max_tokens) * 2, context_limit - estimated_prompt_tokens - safety_margin)
                    max_tokens = max(512, max_tokens)  # Ensure at least 512 tokens for retry
                else:
                    max_tokens = 2048

                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                )
                
                return {
                    'row_idx': row_idx,
                    'output': response.choices[0].message.content,
                    'status': 'success',
                    'error': None,
                    'max_tokens': max_tokens
                }
            except Exception as e:
                print(f"Error processing row {row_idx}: {e}")
                print(f"Error processing row {row_idx}: {type(e).__name__}: {e}")
                traceback.print_exc()
                return {
                    'row_idx': row_idx,
                    'output': None,
                    'status': 'error',
                    'error': str(e),
                    'max_tokens': max_tokens
                }
    
    async def process_file(self, 
                      model: str, 
                      input_file: Path, 
                      output_file: Path,
                      prompt_column: str = 'prompt',
                      output_column: str = 'model_output',
                      valid_column: str = 'valid_output',
                      resume: bool = True,
                      unique_notes_only: bool = False) -> Dict:
        """Process a Parquet file and add results as a new column"""
        
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print(f"{'='*60}")
        # breakpoint()
        
        # Load input file
        df = pd.read_parquet(input_file)
        
        # **************************** randomly sample 1k rows for testing ****************************
        # df = df.sample(n=min(100, len(df)), random_state=42).reset_index(drop=True)

        total_rows = len(df)
        print(f"Total rows: {total_rows}")
        # if unique_notes_only:
        #     df = df[df["is_unique"]]
        #     total_rows = len(df)
        #     print(f"Unique note rows: {total_rows}")
        
        if prompt_column not in df.columns:
            print("Building prompt column...")
            # Shift dates
            df['CHARTDATE'] = pd.to_datetime(df['CHARTDATE'], errors='raise')
            df[prompt_column] = df.apply(build_prompt, axis=1)
        
        # Initialize output column if it doesn't exist
        if output_column not in df.columns:
            df[output_column] = None
            df[f'{output_column}_status'] = None
            df[f'{output_column}_error'] = None
            df[valid_column] = False
            df[f'{output_column}_max_tokens'] = 0
        
        # Check for existing checkpoint and load previous results
        if resume and output_file.exists():
            print(f"Resuming: loading existing results from {output_file}")
            existing_df = pd.read_parquet(output_file)
            
            # Update current df with existing results
            # Only update rows that have been processed
            processed_mask = existing_df[valid_column]
            for col in [output_column, f'{output_column}_status', f'{output_column}_error', valid_column]:
                if col in existing_df.columns:
                    df.loc[processed_mask, col] = existing_df.loc[processed_mask, col]
            df[f'{output_column}_max_tokens'] = existing_df[f'{output_column}_max_tokens']
            
            already_processed = processed_mask.sum()
            print(f"  Found {already_processed} already processed rows")
        
        # Find rows that need processing
        mask = ~df[valid_column]
        rows_to_process = df[mask].index.tolist()
        
        if not rows_to_process:
            print("✓ All rows already processed!")
            success_count = len(df[df[f'{output_column}_status'] == 'success'])
            error_count = len(df[df[f'{output_column}_status'] == 'error'])
            return {
                'status': 'complete', 
                'total': total_rows, 
                'processed': total_rows, 
                'success': success_count, 
                'errors': error_count
            }
        
        print(f"Processing {len(rows_to_process)} remaining rows...")
        print(f"Saving to {output_file}")
        
        # Process rows with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)        
        tasks = [
            self.process_row(model, df.loc[idx, prompt_column], df.loc[idx, f"{output_column}_max_tokens"], idx, semaphore)
            for idx in rows_to_process
        ]
        
        results = await tqdm.gather(*tasks, desc=f"Processing {input_file.name}")
        
        # Update dataframe with results
        for result in results:
            idx = result['row_idx']
            df.at[idx, output_column] = result['output']
            df.at[idx, f'{output_column}_status'] = result['status']
            df.at[idx, f'{output_column}_error'] = result['error']
            df.at[idx, f'{output_column}_max_tokens'] = result['max_tokens']
        
        # Save results to parquet
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df[valid_column] = df[output_column].apply(check_output) # Check if JSON valid, if not, mark for rerun next time
        df.to_parquet(output_file)
        
        # Calculate stats
        success_count = len(df[df[f'{output_column}_status'] == 'success'])
        error_count = len(df[df[f'{output_column}_status'] == 'error'])
        
        print(f"\n✓ Completed: {input_file.name}")
        print(f"  Success: {success_count}/{total_rows}")
        print(f"  Errors: {error_count}/{total_rows}")
        
        return {
            'status': 'complete',
            'file': str(input_file),
            'total': total_rows,
            'success': success_count,
            'errors': error_count
        }
    
    async def process_multiple_files(self, 
                                    model: str,
                                    input_pattern: str,
                                    output_dir: Path,
                                    prompt_column: str = 'prompt',
                                    output_column: str = 'model_output',
                                    valid_column: str = 'valid_output',
                                    max_parallel_files: int = 4,
                                    resume: bool = True,
                                    preserve_structure: bool = True,
                                    unique_notes_only: bool = False):
        """Process multiple Parquet files matching a glob pattern"""
        
        # # Find all parquet files matching pattern
        # base_path = Path(input_pattern.split('**')[0]) if '**' in input_pattern else Path(input_pattern).parent
        # if '**' in input_pattern:
        #     pattern = '/'.join(input_pattern.split('/')[len(base_path.parts):])
        #     input_files = list(base_path.glob(pattern, recursive=True))
        # else:
        #     input_files = list(Path(input_pattern).parent.glob(Path(input_pattern).name))
        base_path = Path(input_pattern.split('**')[0]) if '**' in input_pattern else Path(input_pattern).parent
        input_files = [Path(f) for f in glob(input_pattern, recursive=True)]
        
        if not input_files:
            print(f"No parquet files found matching pattern: {input_pattern}")
            return
        
        print(f"\nFound {len(input_files)} files to process")
        print(f"Processing up to {max_parallel_files} files in parallel")
        print(f"Prompt column: '{prompt_column}'")
        print(f"Output column: '{output_column}'")
        print(f"Valid column: '{valid_column}'")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create processing tasks
        async def process(model, input_file, unique_notes_only):
            if preserve_structure:
                output_file = output_dir / input_file.parent.name / input_file.name
            else:
                output_file = output_dir / input_file.name
            
            return await self.process_file(
                model, input_file, output_file, prompt_column, output_column, valid_column, resume, unique_notes_only
            )
        
        # Process files with controlled parallelism
        file_semaphore = asyncio.Semaphore(max_parallel_files)
        
        async def process_file_limited(model, input_file, unique_notes_only):
            async with file_semaphore:
                return await process(model, input_file, unique_notes_only)
        
        tasks = [process_file_limited(model, f, unique_notes_only) for f in input_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        
        successful_results = [r for r in results if isinstance(r, dict)]
        failed_results = [r for r in results if not isinstance(r, dict)]
        
        if successful_results:
            total_success = sum(r['success'] for r in successful_results)
            total_errors = sum(r['errors'] for r in successful_results)
            total_rows = sum(r['total'] for r in successful_results)
            
            print(f"Successfully processed files: {len(successful_results)}")
            print(f"Failed files: {len(failed_results)}")
            print(f"Total rows: {total_rows}")
            print(f"Successful rows: {total_success}")
            print(f"Error rows: {total_errors}")
        
        if failed_results:
            print(f"\n⚠ Failed files:")
            for i, error in enumerate(failed_results[:5]):
                print(f"  {i+1}. {error}")
        
        print(f"\nOutputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Process Parquet files with vLLM')
    parser.add_argument('--model', type=str, required=True, help='sft_out_full/lfm2.5-1.2b-instruct_bs8_lr3e-5')
    parser.add_argument('--input-pattern', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--prompt-column', type=str, default='prompt')
    parser.add_argument('--output-column', type=str, default='concepts')
    parser.add_argument('--valid-column', type=str, default='valid_output')
    parser.add_argument('--num-servers', type=int, default=8)
    parser.add_argument('--max-concurrent', type=int, default=256)
    parser.add_argument('--max-parallel-files', type=int, default=4)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--preserve-structure', action='store_true')
    parser.add_argument('--unique-notes-only', action='store_true')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    
    args = parser.parse_args()
    
    processor = VLLMParquetProcessor(
        num_servers=args.num_servers,
        max_concurrent=args.max_concurrent,
        host=args.host
    )
    
    asyncio.run(processor.process_multiple_files(
        model=args.model,
        input_pattern=args.input_pattern,
        output_dir=Path(args.output_dir),
        prompt_column=args.prompt_column,
        output_column=args.output_column,
        valid_column=args.valid_column,
        max_parallel_files=args.max_parallel_files,
        resume=not args.no_resume,
        preserve_structure=args.preserve_structure,
        unique_notes_only=args.unique_notes_only
    ))


if __name__ == "__main__":
    main()
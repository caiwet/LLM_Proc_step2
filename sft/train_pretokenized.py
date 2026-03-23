"""
Supervised finetuning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import load_from_disk, load_dataset
from datasets import Features, Sequence, Value

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

@dataclass
class DataCollatorForCausalLMWithPrecomputedLabels:
    tokenizer: any
    label_pad_token_id: int = -100

    def __call__(self, features):
        # split labels from other fields
        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        # pad model inputs
        batch = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]

        # pad labels to max_len
        padded_labels = []
        for lab in labels:
            padded_labels.append(lab + [self.label_pad_token_id] * (max_len - len(lab)))

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


@dataclass
class DataConfig:
    """Dataset and model configuration."""

    model_name: str = field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        metadata={"help": "Model name or path"},
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-tokenized training dataset (.arrow or .parquet)"},
    )
    eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-tokenized validation dataset (.arrow or .parquet)"},
    )


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    output_dir: str = field(
        default="./outputs",
        metadata={"help": "Output directory for model and checkpoints"},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint to resume from (e.g., './outputs/checkpoint-1000')"},
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device"},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Gradient accumulation steps"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate"},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={"help": "Learning rate scheduler type"},
    )
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Warmup steps"},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging interval"},
    )
    save_steps: int = field(
        default=10000,
        metadata={"help": "Save checkpoint interval"},
    )
    eval_steps: int = field(
        default=5000,
        metadata={"help": "Evaluation interval"},
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep"},
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimizer"},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing"},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 training"},
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "Report to (none, tensorboard, wandb)"},
    )


def train(data_config: DataConfig, training_config: TrainingConfig) -> None:
    """Run supervised finetuning with pre-tokenized data."""
    
    # Determine if we're resuming from a checkpoint
    resume_from_checkpoint = training_config.resume_from_checkpoint
    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {resume_from_checkpoint}")
        if not (checkpoint_path / "trainer_state.json").exists():
            raise ValueError(f"Invalid checkpoint: missing trainer_state.json in {resume_from_checkpoint}")
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {resume_from_checkpoint}")
        print(f"{'='*60}\n")
    
    print(f"Loading model: {data_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        data_config.model_name,
        torch_dtype=torch.bfloat16 if training_config.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
        # attn_implementation="sdpa",
        trust_remote_code=True,
        # use_cache=False,
    )

    print(f"Loading tokenizer: {data_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        data_config.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if training_config.gradient_checkpointing:
        print("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Load pre-tokenized datasets
    print(f"Loading pre-tokenized training dataset: {data_config.dataset_path}")
    if data_config.dataset_path.endswith('.arrow') or Path(data_config.dataset_path).is_dir():
        train_dataset = load_from_disk(data_config.dataset_path)
    else:
        # train_dataset = load_dataset("parquet", data_files=data_config.dataset_path, split="train")
        features = Features({
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "labels": Sequence(Value("int64")),  # matches your metadata
        })

        train_dataset = load_dataset(
            "parquet",
            data_files=data_config.dataset_path,
            split="train",
            features=features,
        )

    
    print(f"Loaded {len(train_dataset)} training examples")
    print(f"Dataset columns: {train_dataset.column_names}")

    eval_dataset = None
    if data_config.eval_dataset_path:
        print(f"Loading pre-tokenized eval dataset: {data_config.eval_dataset_path}")
        if data_config.eval_dataset_path.endswith('.arrow') or Path(data_config.eval_dataset_path).is_dir():
            eval_dataset = load_from_disk(data_config.eval_dataset_path)
        else:
            features = Features({
                "input_ids": Sequence(Value("int32")),
                "attention_mask": Sequence(Value("int8")),
                "labels": Sequence(Value("int64")),  # matches your metadata
            })

            eval_dataset = load_dataset(
                "parquet",
                data_files=data_config.eval_dataset_path,
                split="train",
                features=features,
            )
            # eval_dataset = load_dataset("parquet", data_files=data_config.eval_dataset_path, split="train")
        print(f"Loaded {len(eval_dataset)} eval examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        lr_scheduler_type=training_config.lr_scheduler_type,
        warmup_steps=training_config.warmup_steps,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=training_config.eval_steps if eval_dataset else None,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        optim=training_config.optim,
        gradient_checkpointing=training_config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=training_config.bf16,
        report_to=training_config.report_to,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    # Data collator (handles padding only, labels already created)
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,
    # )
    
    data_collator = DataCollatorForCausalLMWithPrecomputedLabels(tokenizer)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\n" + "="*60)
    print("Starting training...")
    print(f"  Num examples: {len(train_dataset)}")
    print(f"  Num epochs: {training_config.num_train_epochs}")
    print(f"  Batch size per device: {training_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {training_config.gradient_accumulation_steps}")
    print(f"  Total batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps * torch.cuda.device_count()}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  LR scheduler: {training_config.lr_scheduler_type}")
    print(f"  Warmup steps: {training_config.warmup_steps}")
    if resume_from_checkpoint:
        print(f"  Resuming from: {resume_from_checkpoint}")
    print("="*60 + "\n")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"\nSaving model to {training_config.output_dir}")
    trainer.save_model(training_config.output_dir)
    tokenizer.save_pretrained(training_config.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    parser = HfArgumentParser((DataConfig, TrainingConfig))
    data_config, training_config = parser.parse_args_into_dataclasses()

    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    print("\nData Configuration:")
    print(f"  model_name: {data_config.model_name}")
    print(f"  dataset_path: {data_config.dataset_path}")
    print(f"  eval_dataset_path: {data_config.eval_dataset_path}")
    
    print("\nTraining Configuration:")
    print(f"  output_dir: {training_config.output_dir}")
    print(f"  resume_from_checkpoint: {training_config.resume_from_checkpoint}")
    print(f"  num_train_epochs: {training_config.num_train_epochs}")
    print(f"  per_device_train_batch_size: {training_config.per_device_train_batch_size}")
    print(f"  gradient_accumulation_steps: {training_config.gradient_accumulation_steps}")
    print(f"  learning_rate: {training_config.learning_rate}")
    print(f"  lr_scheduler_type: {training_config.lr_scheduler_type}")
    print(f"  warmup_steps: {training_config.warmup_steps}")
    print(f"  gradient_checkpointing: {training_config.gradient_checkpointing}")
    print(f"  bf16: {training_config.bf16}")
    print(f"  save_steps: {training_config.save_steps}")
    print(f"  eval_steps: {training_config.eval_steps if data_config.eval_dataset_path else 'N/A'}")
    print(f"  report_to: {training_config.report_to}")
    print("="*60 + "\n")

    train(data_config, training_config)
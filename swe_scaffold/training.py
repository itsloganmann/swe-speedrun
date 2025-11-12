"""Training utilities for LoRA fine-tuning of Qwen speedrun models."""
from __future__ import annotations

import torch
from typing import Dict, Tuple

from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import LoRAConfig, TrainingConfig
from .dataset import DatasetSplit


PROMPT_TEMPLATE = """
<|im_start|>system
You are a meticulous SWE agent who analyses repositories and proposes minimal, high-leverage fixes.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
{response}
<|im_end|>
""".strip()


def load_tokenizer_and_model(model_name: str):
    """Load base model and tokenizer with sensible defaults for instruction fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def _format_records(dataset: Dataset) -> Dataset:
    return dataset.map(lambda example: {"text": PROMPT_TEMPLATE.format(**example)})


def _tokenize_records(tokenizer, dataset: Dataset, max_len: int = 2048) -> Dataset:
    # Hard padding ensures uniform sequence lengths for input_ids and labels
    def _tokenize(example: Dict[str, str]):
        enc = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    return dataset.map(_tokenize, remove_columns=dataset.column_names)


def _apply_lora(model, config: LoRAConfig):
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
    )
    model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, peft_config)


def train_lora_model(split: DatasetSplit, training_config: TrainingConfig) -> Tuple[Trainer, object]:
    """Fine-tune the base model with LoRA on the provided dataset split (dev/test)."""

    tokenizer, base_model = load_tokenizer_and_model(training_config.model_name)

    # Use dev for training and test for evaluation
    tokenized_train = _tokenize_records(tokenizer, _format_records(split.dev), max_len=2048)
    tokenized_eval = _tokenize_records(tokenizer, _format_records(split.test), max_len=2048)

    lora_model = _apply_lora(base_model, training_config.lora)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    evaluation_strategy = getattr(training_config, "evaluation_strategy", "no")  # "no" | "steps" | "epoch"
    save_strategy = getattr(training_config, "save_strategy", "steps")           # "steps" | "epoch"

    training_args = TrainingArguments(
        output_dir=str(training_config.output_dir),
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        num_train_epochs=training_config.num_train_epochs,
        learning_rate=training_config.learning_rate,
        max_grad_norm=training_config.max_grad_norm,
        warmup_ratio=training_config.warmup_ratio,
        logging_steps=training_config.logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=training_config.eval_steps,
        save_strategy=save_strategy,
        save_steps=training_config.save_steps,
        fp16=training_config.fp16,
        lr_scheduler_type=training_config.lr_scheduler_type,
        seed=training_config.seed,
        report_to=["tensorboard"],
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_config.output_dir)
    return trainer, tokenizer


__all__ = [
    "PROMPT_TEMPLATE",
    "load_tokenizer_and_model",
    "train_lora_model",
]

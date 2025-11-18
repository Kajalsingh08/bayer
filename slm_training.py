
# Fine-tune Qwen2.5-0.5B-Instruct on a graph corpus 
# Optimized for small parameter models (<1B) with efficient multi-GPU settings.

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset


def main():
    # -----------------------------
    # Config
    # -----------------------------
    model_name = "Qwen2.5-0.5B-Instruct"
    corpus_path = "graph_corpus.txt"  # change if needed
    output_dir = "models/qwen_graph_pretrained"
    max_length = 2048  # long context helps encode relationships
    seed = 42

    # -----------------------------
    # Load tokenizer and model
    # -----------------------------
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,   # fp16 for small models
        device_map="auto"
    )

    # Optional: gradient checkpointing (usually not needed for 0.5B, but harmless)
    # model.gradient_checkpointing_enable()
    # -----------------------------
    # Load and tokenize dataset
    # -----------------------------
    print(f"Loading corpus from: {corpus_path}")
    dataset = load_dataset('text', data_files={'train': corpus_path})

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # -----------------------------
    # Training arguments 
    # -----------------------------
    # Adjust LR slightly higher for small model and larger batch, monitor loss.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,   # small model → larger batch per GPU
        gradient_accumulation_steps=1,    # more GPUs → less accumulation
        learning_rate=5e-5,               # higher LR works for small models; lower if unstable
        warmup_steps=800,                 # longer warmup for stability at larger batch
        logging_steps=20,
        save_steps=2000,
        save_total_limit=5,
        fp16=True,                        # use fp16 for speed/memory on small models
        bf16=False,                       # bf16 not necessary unless on A100/H100 and you prefer it
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=8,
        report_to=["tensorboard"],
        evaluation_strategy="no",         # set "steps" and add eval_dataset if you have one
        remove_unused_columns=False,
        seed=seed
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=data_collator
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("\n============================================")
    print("Starting continued pre-training (schema-aware)")
    print("============================================\n")
    trainer.train()

    # -----------------------------
    # Save
    # -----------------------------
    final_dir = os.path.join(output_dir, "final")
    print(f"\nSaving model and tokenizer to: {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("✓ Training complete")


if __name__ == "__main__":
    main()

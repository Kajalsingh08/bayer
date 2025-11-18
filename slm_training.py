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
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    corpus_path = "graph_corpus_v1.txt"   # <-- your corpus
    output_dir = "models/qwen_graph_pretrained"
    max_length = 2048
    seed = 42

    # HF token (optional if model is gated/private)
    hf_token = os.environ.get("HF_TOKEN", None)

    # -----------------------------
    # Load tokenizer & model
    # -----------------------------
    print(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        use_auth_token=hf_token         # FIXED
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=hf_token         # FIXED
    )

    # -----------------------------
    # Load and tokenize dataset
    # -----------------------------
    print(f"Loading corpus: {corpus_path}")
    dataset = load_dataset("text", data_files={"train": corpus_path})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # -----------------------------
    # Training arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_steps=800,
        logging_steps=20,
        save_steps=2000,
        save_total_limit=5,
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        seed=seed
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("\n============================================")
    print("Starting continued pre-training on graph corpus")
    print("============================================\n")
    trainer.train()
   

    # -----------------------------
    # Save final model
    final_dir = os.path.join(output_dir, "final")
    print(f"Saving model to: {final_dir}")

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("✓ Training complete")

if __name__ == "__main__":
    main()

# ======================================================================================
# Instruction Fine-Tuning Script with LoRA and 4-bit Quantization
#
# Description:
# This script performs Parameter-Efficient Fine-Tuning (PEFT) on a base language model
# using the LoRA (Low-Rank Adaptation) technique. It is designed to teach the model
# a specific skill—in this case, reasoning about a question and generating a structured
# JSON output containing an explanation and a Cube query.
#
# Key Technologies Used:
# - `transformers`: For loading the base model and tokenizer.
# - `bitsandbytes`: For 4-bit quantization (QLoRA), drastically reducing memory usage.
# - `peft`: For applying the LoRA configuration to the model.
# - `trl`: For using the `SFTTrainer` (Supervised Fine-tuning Trainer), which simplifies
#   the training process for instruction-based datasets.
# - `datasets`: For loading and processing the training data.
#
# Workflow:
# 1. Load Model & Tokenizer: The base model is loaded in 4-bit precision.
# 2. Prepare for PEFT: The quantized model is wrapped with LoRA adapters.
# 3. Load & Format Dataset: The `instructions.jsonl` file is loaded and each entry
#    is formatted into a conversational prompt.
# 4. Train: The `SFTTrainer` handles the training loop, updating only the LoRA weights.
# 5. Save: The trained LoRA adapters are saved to the output directory.
# ======================================================================================

import argparse
import json
import os
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

class LoRAFinetuner:
    """
    A class to handle the fine-tuning of a language model using LoRA (Low-Rank Adaptation).
    """

    def __init__(self, config):
        """
        Initializes the Finetuner with a configuration object.
        Args:
            config: An argparse.Namespace containing all the training parameters.
        """
        self.config = config
        self.model = None
        self.tokenizer = None

    def _load_model_and_tokenizer(self):
        """
        Loads the base model and tokenizer.
        It applies 4-bit quantization using bitsandbytes to reduce memory footprint,
        allowing fine-tuning of large models on consumer-grade GPUs.
        """
        print(f"Loading base model from: {self.config.base_model_path}")

        # --- Quantization Configuration ---
        # This configures the model to be loaded in 4-bit precision.
        # `load_in_4bit`: Activates 4-bit quantization.
        # `bnb_4bit_quant_type="nf4"`: Uses the "Normal Float 4" data type, which is
        #   optimized for normally distributed weights, common in LLMs.
        # `bnb_4bit_compute_dtype=torch.bfloat16`: Specifies that computations
        #   (like matrix multiplications) should be done in bfloat16 for speed,
        #   while weights are stored in 4-bit.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # --- Load Model ---
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            quantization_config=quantization_config,
            device_map="auto",  # Automatically distributes the model across available devices (GPU/CPU)
            trust_remote_code=True,
        )
        self.model.config.use_cache = False # Disable caching for training
        self.model.config.pretraining_tp = 1 # Set tensor parallelism to 1

        # --- Load Tokenizer ---
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Tokenizer `pad_token` was not set. Setting it to `eos_token`.")

    def _prepare_peft_model(self):
        """
        Prepares the quantized model for Parameter-Efficient Fine-Tuning (PEFT)
        by attaching LoRA adapters to it.
        """
        print("Preparing model for PEFT training...")
        # This prepares the quantized model for training by ensuring gradients can flow correctly.
        self.model = prepare_model_for_kbit_training(self.model)

        # --- LoRA Configuration ---
        # This defines how the LoRA adapters are applied.
        # `r` (rank): The dimension of the low-rank matrices. A key hyperparameter to tune.
        #   - Higher `r` -> more trainable parameters, potentially better performance, but more memory.
        #   - Common values: 8, 16, 32, 64.
        # `lora_alpha`: The scaling factor for the LoRA weights. `alpha / r` is the scaling.
        #   - Higher `alpha` gives more weight to the LoRA activations.
        #   - Common practice is to set `lora_alpha` to be 2x `r`.
        # `lora_dropout`: Dropout probability for LoRA layers to prevent overfitting.
        # `target_modules`: The specific layers of the model to apply LoRA to.
        #   - For modern transformers, targeting all linear layers (`"all-linear"`) is a robust default.
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )

        # Wrap the base model with the PEFT model configuration
        self.model = get_peft_model(self.model, lora_config)
        print("PEFT model prepared. Trainable parameters:")
        self.model.print_trainable_parameters() # Shows the dramatic reduction in trainable params

    def _create_dataset(self):
        """
        Loads the instruction dataset from a .jsonl file and formats it into a
        conversational prompt structure that the SFTTrainer can use.
        """
        print(f"Loading and formatting dataset from: {self.config.dataset_path}")

        def format_instruction(sample):
            # This function is crucial. It defines the exact format of the prompt
            # that the model will be trained on. The model learns to generate text
            # that follows the `<|assistant|>` token.
            return f"<|user|>\n{sample['question']}<|end|>\n<|assistant|>\n{json.dumps({'explanation': sample['explanation'], 'cube_query': sample['cube_query']}, indent=2)}<|end|>"

        dataset = Dataset.from_json(self.config.dataset_path)
        # The SFTTrainer expects a 'text' column containing the fully formatted prompt.
        dataset = dataset.map(lambda sample: {"text": format_instruction(sample)})

        print(f"Dataset loaded and formatted. Number of examples: {len(dataset)}")
        print("\n--- Example of Formatted Text ---")
        print(dataset[0]['text'])
        print("---------------------------------\n")

        return dataset

    def train(self):
        """
        Executes the full training process from loading to saving.
        """
        self._load_model_and_tokenizer()
        self._prepare_peft_model()
        dataset = self._create_dataset()

        # --- Training Arguments ---
        # These arguments control the training loop.
        # `learning_rate`: A critical hyperparameter. 2e-4 is a common starting point for LoRA.
        # `per_device_train_batch_size`: How many samples to process per GPU at once.
        #   - Tune this based on your VRAM. Decrease if you get "Out of Memory" errors.
        # `gradient_accumulation_steps`: Simulates a larger batch size. `batch_size * accumulation_steps`
        #   is the effective batch size.
        # `optim`: "paged_adamw_8bit" is an optimizer that works well with quantization.
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=2,
            optim="paged_adamw_8bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=10,
            fp16=True, # Use mixed-precision training for speed
            push_to_hub=False,
            report_to="tensorboard",
            logging_dir=f"{self.config.output_dir}/logs",
        )

        # --- Initialize the Trainer ---
        # The SFTTrainer from TRL simplifies the training process significantly.
        # The model is already PEFT-prepared, so we don't pass peft_config.
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            args=training_args,
        )

        print("Starting fine-tuning...")
        trainer.train()

        print("Training complete. Saving final LoRA adapters...")
        final_path = os.path.join(self.config.output_dir, "final_model")
        trainer.save_model(final_path)
        print(f"LoRA adapters saved successfully to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA (QLoRA).")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base pre-trained model directory.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the .jsonl instruction dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained LoRA adapters.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size per device. Lower this if you run out of VRAM.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (dimension). A key hyperparameter. Try 8, 16, 32.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling parameter. Often set to 2 * lora_r.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers.")

    args = parser.parse_args()

    finetuner = LoRAFinetuner(args)
    finetuner.train()
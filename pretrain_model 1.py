"""
Continued Pre-Training on Graph Corpus
Embeds graph knowledge into a base model's weights.

This script performs the first crucial step of our two-phase training strategy:
1.  **Continued Pre-Training (This Script):** We take a general-purpose base model
    and continue its training on our specific `schema_corpus.txt`. This forces
    the model to learn the vocabulary, structure, and relationships of our
    unique data schema, embedding that knowledge directly into its weights.
2.  **Instruction Fine-Tuning (Next Step):** After this, we will teach the
    now schema-aware model how to answer questions and follow instructions.
"""

# --- Core Libraries ---
import torch  # The fundamental library for machine learning with tensors and GPU support.
from transformers import (
    AutoModelForCausalLM,  # Loads any Causal Language Model (e.g., GPT, Phi-3).
    AutoTokenizer,         # Loads the tokenizer that corresponds to the model.
    TrainingArguments,     # A comprehensive class to configure all aspects of the training.
    Trainer,               # A high-level class that orchestrates the entire training loop.
    DataCollatorForLanguageModeling  # Creates batches of data suitable for language modeling.
)
from datasets import load_dataset  # A library for easily loading and processing datasets.
import os  # Used here to dynamically set the number of data loader workers.
from pathlib import Path  # Used for creating directories in a platform-agnostic way.


class GraphPreTrainer:
    """
    Handles the continued pre-training task for embedding schema knowledge.
    This class encapsulates all the logic for loading the model, preparing the
    dataset, and running the training loop.
    """

    def __init__(
        self,
        base_model: str,
        corpus_path: str,
        output_dir: str
    ):
        """
        Initializes the pre-trainer with the core configuration.

        Args:
            base_model (str): The identifier of the base model from Hugging Face.
                              Example: "microsoft/Phi-3-mini-4k-base".
            corpus_path (str): The file path to our custom training corpus.
            output_dir (str): The directory where the final trained model will be saved.
        """
        self.base_model = base_model
        self.corpus_path = corpus_path
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        print(f"Initializing pre-training with base model: {self.base_model}")

    def load_model_and_tokenizer(self):
        """
        Loads the specified base model and its corresponding tokenizer from Hugging Face.
        This step downloads the model weights if they are not already cached locally.
        """
        print("Loading model and tokenizer...")

        # The tokenizer converts text into a sequence of numbers (tokens) that the model can understand.
        # `trust_remote_code=True` is required for some newer models like Phi-3 that have custom code.
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        
        # Models need a "padding token" to make all sequences in a batch the same length.
        # If it's not set, we use the "end of sentence" token as a sensible default.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # This loads the actual model weights.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,  # `bfloat16` is a modern data type that saves memory on compatible GPUs (like Ampere series) without sacrificing much precision.
            device_map="auto",           # Automatically distributes the model across available GPUs, or uses the CPU if no GPU is found.
            trust_remote_code=True       # Required for Phi-3.
        )

        print(f"✓ Model loaded: {self.base_model}")
        print(f"  - Parameters: {self.model.num_parameters():,}")
        print(f"  - Memory Footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")

    def prepare_dataset(self, max_length: int = 1024):
        """
        Loads the text corpus from the file and prepares it for training by tokenizing it.
        """
        print(f"\nPreparing dataset from: {self.corpus_path}")

        # `load_dataset` can read various formats. Here, we're just loading a plain text file.
        dataset = load_dataset('text', data_files={'train': self.corpus_path}, split='train')
        print(f"  - Loaded {len(dataset)} raw text examples (lines from the file).")

        # Filter out empty lines to prevent runtime errors with empty tensors
        original_rows = len(dataset)
        dataset = dataset.filter(lambda example: example['text'] is not None and len(example['text'].strip()) > 0)
        filtered_rows = len(dataset)
        if original_rows > filtered_rows:
            print(f"  - Filtered out {original_rows - filtered_rows} empty or whitespace-only lines.")

        # This is the function that will be applied to every example in our dataset.
        def tokenize_function(examples):
            # The tokenizer converts the text to token IDs.
            return self.tokenizer(
                examples['text'],
                truncation=True,      # Truncate examples longer than `max_length`.
                max_length=max_length,
                padding=False,        # We don't pad here; the data collator will handle it later.
                return_tensors=None   # Return Python lists instead of PyTorch tensors.
            )

        print("  - Tokenizing dataset...")
        # The `.map()` function is highly efficient. It applies `tokenize_function` to the entire dataset,
        # using multiple processes and caching the results.
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,              # Process multiple examples at once for speed.
            remove_columns=['text'],   # We no longer need the original text column after tokenization.
            desc="Running tokenizer on dataset"
        )
        
        print(f"✓ Dataset prepared with {len(tokenized_dataset)} tokenized examples.")
        return tokenized_dataset

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        gradient_accumulation_steps: int
    ):
        """Configures and runs the main training loop using the Hugging Face Trainer."""
        self.load_model_and_tokenizer()
        train_dataset = self.prepare_dataset()

        # The Data Collator is responsible for taking a list of examples from our dataset
        # and dynamically padding them to the same length to form a "batch".
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal Language Modeling (predicting the next token), not Masked Language Modeling.
        )

        # `TrainingArguments` is a powerful class that holds all the hyperparameters for the training run.
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,  # How many examples to process on one GPU at a time.
            gradient_accumulation_steps=gradient_accumulation_steps, # Simulate a larger batch size.
            learning_rate=learning_rate,
            warmup_steps=500,  # Gradually increase the learning rate for the first 500 steps to stabilize training.
            logging_dir=f"{self.output_dir}/logs", # Directory for TensorBoard logs.
            logging_steps=50, # Log training loss every 50 steps.
            save_steps=1000, # Save a checkpoint of the model every 1000 steps.
            save_total_limit=3, # Only keep the last 3 checkpoints to save disk space.
            bf16=True,  # Enable bfloat16 mixed-precision training for speed and memory savings.
            optim="paged_adamw_8bit", # A memory-efficient optimizer that pages states to CPU RAM.
            lr_scheduler_type="cosine", # Cosine learning rate scheduler often leads to better results.
            weight_decay=0.01, # A regularization technique to prevent overfitting.
            max_grad_norm=1.0, # Gradient clipping to prevent exploding gradients.
            dataloader_num_workers=os.cpu_count() // 2, # Use multiple CPU cores to load data in the background.
            remove_unused_columns=False,
            report_to=["tensorboard"], # Enable logging to TensorBoard.
        )

        print("\n--- Training Configuration ---")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Per-Device Batch Size: {batch_size}")
        print(f"  - Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"  - Effective Global Batch Size: {batch_size * gradient_accumulation_steps * torch.cuda.device_count()}")
        print(f"  - Learning Rate: {learning_rate}")
        print(f"  - Output Directory: {self.output_dir}")
        print("----------------------------\n")

        # The `Trainer` object abstracts away the entire training loop.
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Explicitly disable cache for training stability with this model version
        self.model.config.use_cache = False

        print("Starting Continued Pre-Training...")
        # This single line starts the entire training process.
        trainer.train()

        # After training is complete, save the final model and tokenizer.
        final_path = f"{self.output_dir}/final"
        print(f"\nSaving final model to: {final_path}")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)

        print("\n✓ Pre-training complete!")
        print(f"✓ Model saved to: {final_path}")


def main():
    """
    Main execution function. This is where we define the specific configuration
    for our training run and kick off the process.
    """
    
    # --- Configuration ---
    # All hyperparameters are defined in this dictionary for easy access and modification.
    config = {
        "base_model": "models/phi-3-mini-4k-instruct", # Changed to a local path
        "corpus_path": "training_data/schema_corpus.txt",
        "output_dir": "models/phi3_schema_pretrained",
        "num_epochs": 3,
        "batch_size": 1,  # Set to 1 to be safe on most GPUs. Increase if you have more VRAM.
        "learning_rate": 2e-5, # A common, effective learning rate for fine-tuning.
        "gradient_accumulation_steps": 16 # This simulates a larger batch size of (1 * 16 = 16), which helps stabilize training without using more memory.
    }

    # Ensure the directory to save the model exists.
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Create an instance of our trainer class.
    trainer = GraphPreTrainer(
        base_model=config["base_model"],
        corpus_path=config["corpus_path"],
        output_dir=config["output_dir"]
    )

    # Start the training process with the specified hyperparameters.
    trainer.train(
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"]
    )


if __name__ == "__main__":
    # This standard Python construct ensures that `main()` is called only when the script is executed directly.
    main()
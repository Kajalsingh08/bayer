# Schema-Aware SLM Implementation Guide

## Overview

This guide provides step-by-step implementation code for training a schema-aware SLM that internalizes your GraphRAG metadata structure.

**Prerequisites:**
- Python 3.10+
- Access to graph data (`data/full_meat.json`, `data/business_taxonomy.json`)
- GPU with 40GB+ VRAM (or 16GB with QLoRA)
- 500GB disk space

---

## Step 1: Corpus Generation

### A. Install Dependencies

```bash
# Create virtual environment
python -m venv venv_slm
source venv_slm/bin/activate  # On Windows: venv_slm\Scripts\activate

# Install required packages
pip install transformers==4.36.0
pip install datasets==2.16.0
pip install peft==0.7.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.3
pip install torch==2.1.0
pip install sentencepiece
pip install protobuf
```

### B. Corpus Generation Script

Create `scripts/generate_corpus.py`:

```python
"""
Graph-to-Text Corpus Generator
Converts Neo4j graph metadata into natural language training corpus
"""

import json
from typing import List, Dict, Set
from pathlib import Path
import hashlib


class GraphCorpusGenerator:
    """Generate natural language corpus from graph metadata"""
    
    def __init__(self, metadata_path: str, taxonomy_path: str):
        """
        Args:
            metadata_path: Path to full_meat.json
            taxonomy_path: Path to business_taxonomy.json
        """
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        with open(taxonomy_path) as f:
            self.taxonomy = json.load(f)
        
        self.corpus_parts = []
        self.seen_entities = set()  # Prevent duplicates
    
    def generate_cube_description(self, cube: Dict) -> str:
        """Generate detailed natural language description of a cube"""
        
        cube_name = cube.get('name', 'Unknown')
        cube_title = cube.get('title', cube_name)
        
        # Start with overview
        desc = f"# {cube_title} Cube\n\n"
        desc += f"The {cube_name} is a data cube "
        
        # Add description if available
        if 'description' in cube:
            desc += f"that {cube['description'].lower()}\n\n"
        else:
            desc += f"used for data analysis.\n\n"
        
        # Cube properties
        desc += "## Cube Properties\n\n"
        desc += f"- **Name:** {cube_name}\n"
        desc += f"- **Title:** {cube_title}\n"
        desc += f"- **Type:** Cube\n"
        
        if 'is_visible' in cube:
            desc += f"- **Visible:** {'Yes' if cube['is_visible'] else 'No'}\n"
        if 'is_public' in cube:
            desc += f"- **Public:** {'Yes' if cube['is_public'] else 'No'}\n"
        
        desc += "\n"
        
        # Measures
        measures = cube.get('measures', [])
        if measures:
            desc += f"## Measures ({len(measures)} total)\n\n"
            desc += f"The {cube_name} cube contains {len(measures)} measures:\n\n"
            
            for i, measure in enumerate(measures, 1):
                m_name = measure.get('name', 'unknown')
                m_title = measure.get('title', m_name)
                m_type = measure.get('type', 'unknown')
                agg_type = measure.get('aggType', 'unknown')
                
                desc += f"{i}. **{m_name}**\n"
                desc += f"   - Title: {m_title}\n"
                desc += f"   - Type: {m_type}\n"
                desc += f"   - Aggregation: {agg_type}\n"
                
                if 'description' in measure:
                    desc += f"   - Description: {measure['description']}\n"
                
                desc += "\n"
        
        # Dimensions
        dimensions = cube.get('dimensions', [])
        if dimensions:
            desc += f"## Dimensions ({len(dimensions)} total)\n\n"
            desc += f"The {cube_name} cube contains {len(dimensions)} dimensions:\n\n"
            
            for i, dim in enumerate(dimensions, 1):
                d_name = dim.get('name', 'unknown')
                d_title = dim.get('title', d_name)
                d_type = dim.get('type', 'unknown')
                
                desc += f"{i}. **{d_name}**\n"
                desc += f"   - Title: {d_title}\n"
                desc += f"   - Type: {d_type}\n"
                
                if dim.get('primaryKey'):
                    desc += f"   - **Primary Key:** Yes\n"
                
                if 'description' in dim:
                    desc += f"   - Description: {dim['description']}\n"
                
                desc += "\n"
        
        desc += "\n---\n\n"
        return desc
    
    def generate_measure_description(self, measure: Dict, cube_name: str) -> str:
        """Generate standalone measure description"""
        
        m_name = measure.get('name', 'unknown')
        m_title = measure.get('title', m_name)
        full_name = f"{cube_name}.{m_name}"
        
        desc = f"# Measure: {m_title}\n\n"
        desc += f"The **{m_name}** is a measure in the {cube_name} cube.\n\n"
        desc += f"- **Full Name:** {full_name}\n"
        desc += f"- **Title:** {m_title}\n"
        
        if 'shortTitle' in measure:
            desc += f"- **Short Title:** {measure['shortTitle']}\n"
        
        desc += f"- **Type:** {measure.get('type', 'unknown')}\n"
        desc += f"- **Aggregation:** {measure.get('aggType', 'unknown')}\n"
        
        if 'description' in measure:
            desc += f"\n**Description:** {measure['description']}\n"
        
        desc += f"\nThis measure belongs to the {cube_name} cube.\n\n"
        desc += "---\n\n"
        return desc
    
    def generate_hierarchy_description(self) -> str:
        """Generate business hierarchy descriptions"""
        
        desc = "# Business Hierarchy\n\n"
        desc += "## Organizational Structure\n\n"
        
        org_name = self.taxonomy.get('organization', 'Organization')
        desc += f"The **{org_name}** is the top-level organization.\n\n"
        
        divisions = self.taxonomy.get('divisions', {})
        
        for div_name, div_data in divisions.items():
            desc += f"### Division: {div_name}\n\n"
            desc += f"The {org_name} has a division called **{div_name}**.\n\n"
            
            business_units = div_data.get('business_units', {})
            
            for bu_name, bu_data in business_units.items():
                desc += f"#### Business Unit: {bu_name}\n\n"
                desc += f"The {div_name} division contains the **{bu_name}** business unit.\n\n"
                
                subdivisions = bu_data.get('subdivisions', {})
                
                for subdiv_name, subdiv_data in subdivisions.items():
                    desc += f"##### Subdivision: {subdiv_name}\n\n"
                    desc += f"The {bu_name} business unit has a **{subdiv_name}** subdivision.\n\n"
                    
                    functional_areas = subdiv_data.get('functional_areas', [])
                    if functional_areas:
                        desc += f"**Functional Areas:** {', '.join(functional_areas)}\n\n"
                    
                    views = subdiv_data.get('views', [])
                    if views:
                        desc += f"**Views:** {', '.join(views)}\n\n"
        
        desc += "---\n\n"
        return desc
    
    def generate_query_patterns(self) -> str:
        """Generate Q&A patterns for common queries"""
        
        desc = "# Common Query Patterns\n\n"
        
        # For each cube, generate Q&A pairs
        cubes = self.metadata.get('cubes', [])
        
        for cube in cubes[:10]:  # Limit for brevity
            cube_name = cube.get('name', 'Unknown')
            measures = cube.get('measures', [])
            dimensions = cube.get('dimensions', [])
            
            # Measure query
            if measures:
                desc += f"**Question:** What measures are in {cube_name}?\n\n"
                desc += f"**Answer:** The {cube_name} cube has {len(measures)} measures: "
                measure_names = [m.get('name', 'unknown') for m in measures]
                desc += ", ".join(measure_names) + ".\n\n"
            
            # Primary key query
            pk_dims = [d for d in dimensions if d.get('primaryKey')]
            if pk_dims:
                desc += f"**Question:** What is the primary key of {cube_name}?\n\n"
                desc += f"**Answer:** The primary key is {pk_dims[0].get('name')}, "
                desc += f"which is a {pk_dims[0].get('type', 'unknown')} dimension.\n\n"
            
            # Dimension query
            if dimensions:
                desc += f"**Question:** How many dimensions does {cube_name} have?\n\n"
                desc += f"**Answer:** {cube_name} has {len(dimensions)} dimensions.\n\n"
        
        desc += "---\n\n"
        return desc
    
    def generate_relationship_patterns(self) -> str:
        """Generate relationship descriptions"""
        
        desc = "# Cube Relationships\n\n"
        
        # Look for foreign key relationships in dimensions
        cubes = self.metadata.get('cubes', [])
        
        for cube in cubes:
            cube_name = cube.get('name', 'Unknown')
            dimensions = cube.get('dimensions', [])
            
            for dim in dimensions:
                dim_name = dim.get('name', '')
                
                # Heuristic: if dimension name ends with _id and isn't primary key
                if dim_name.endswith('_id') and not dim.get('primaryKey'):
                    # Infer relationship
                    target_cube = dim_name.replace('_id', '').title() + 'ID'
                    
                    desc += f"**Relationship:** {cube_name} → {target_cube}\n\n"
                    desc += f"The {cube_name} cube references the {target_cube} cube "
                    desc += f"through the {dim_name} dimension.\n\n"
                    desc += f"This is a many-to-one relationship where multiple records "
                    desc += f"in {cube_name} can reference a single record in {target_cube}.\n\n"
        
        desc += "---\n\n"
        return desc
    
    def generate_full_corpus(self) -> str:
        """Generate complete training corpus"""
        
        print("Generating corpus...")
        
        corpus_parts = []
        
        # 1. Business Hierarchy (10% of corpus)
        print("  - Generating hierarchy descriptions...")
        corpus_parts.append(self.generate_hierarchy_description())
        
        # 2. Cube Descriptions (40% of corpus)
        print("  - Generating cube descriptions...")
        cubes = self.metadata.get('cubes', [])
        for i, cube in enumerate(cubes):
            if i % 10 == 0:
                print(f"    Processed {i}/{len(cubes)} cubes")
            corpus_parts.append(self.generate_cube_description(cube))
        
        # 3. Measure Descriptions (20% of corpus)
        print("  - Generating measure descriptions...")
        for cube in cubes:
            cube_name = cube.get('name', 'Unknown')
            measures = cube.get('measures', [])
            for measure in measures:
                corpus_parts.append(self.generate_measure_description(measure, cube_name))
        
        # 4. Query Patterns (20% of corpus)
        print("  - Generating query patterns...")
        corpus_parts.append(self.generate_query_patterns())
        
        # 5. Relationships (10% of corpus)
        print("  - Generating relationship patterns...")
        corpus_parts.append(self.generate_relationship_patterns())
        
        # Combine all parts
        full_corpus = "\n".join(corpus_parts)
        
        # Calculate stats
        char_count = len(full_corpus)
        word_count = len(full_corpus.split())
        token_estimate = int(word_count * 1.3)  # Rough estimate
        
        print(f"\nCorpus Statistics:")
        print(f"  - Characters: {char_count:,}")
        print(f"  - Words: {word_count:,}")
        print(f"  - Estimated Tokens: {token_estimate:,}")
        
        return full_corpus
    
    def save_corpus(self, output_path: str):
        """Generate and save corpus to file"""
        
        corpus = self.generate_full_corpus()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(corpus)
        
        print(f"\n✓ Corpus saved to: {output_path}")
        return corpus


def main():
    """Main execution"""
    
    # Paths
    metadata_path = "data/full_meat.json"
    taxonomy_path = "data/business_taxonomy.json"
    output_path = "training_data/graph_corpus_v1.txt"
    
    # Generate corpus
    generator = GraphCorpusGenerator(metadata_path, taxonomy_path)
    generator.save_corpus(output_path)


if __name__ == "__main__":
    main()
```

**Run corpus generation:**

```bash
python scripts/generate_corpus.py
```

**Expected output:**
```
Generating corpus...
  - Generating hierarchy descriptions...
  - Generating cube descriptions...
    Processed 0/200 cubes
    Processed 10/200 cubes
    ...
  - Generating measure descriptions...
  - Generating query patterns...
  - Generating relationship patterns...

Corpus Statistics:
  - Characters: 1,245,678
  - Words: 187,543
  - Estimated Tokens: 243,806

✓ Corpus saved to: training_data/graph_corpus_v1.txt
```

---

## Step 2: Continued Pre-Training

### A. Pre-Training Script

Create `scripts/pretrain_model.py`:

```python
"""
Continued Pre-Training on Graph Corpus
Embeds graph knowledge into model weights
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import os
from pathlib import Path


class GraphPreTrainer:
    """Continued pre-training for graph knowledge"""
    
    def __init__(
        self,
        base_model: str = "mistralai/Mistral-7B-v0.1",
        corpus_path: str = "training_data/graph_corpus_v1.txt",
        output_dir: str = "models/mistral_graph_pretrained"
    ):
        self.base_model = base_model
        self.corpus_path = corpus_path
        self.output_dir = output_dir
        
        print(f"Initializing pre-training with base model: {base_model}")
        
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        
        print("Loading model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✓ Model loaded: {self.base_model}")
        print(f"  - Parameters: {self.model.num_parameters():,}")
        print(f"  - Memory: {self.model.get_memory_footprint() / 1e9:.2f} GB")
    
    def prepare_dataset(self, max_length: int = 2048):
        """Prepare dataset from corpus"""
        
        print(f"\nPreparing dataset from: {self.corpus_path}")
        
        # Load corpus as dataset
        dataset = load_dataset(
            'text',
            data_files={'train': self.corpus_path},
            split='train'
        )
        
        print(f"  - Loaded {len(dataset)} examples")
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )
        
        print("  - Tokenizing...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing"
        )
        
        print(f"✓ Dataset prepared: {len(tokenized_dataset)} examples")
        
        return tokenized_dataset
    
    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        gradient_accumulation_steps: int = 8
    ):
        """Run continued pre-training"""
        
        # Load model
        self.load_model_and_tokenizer()
        
        # Prepare dataset
        train_dataset = self.prepare_dataset()
        
        # Data collator for causal LM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=500,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=50,
            save_steps=1000,
            save_total_limit=3,
            fp16=False,
            bf16=True,  # Use bfloat16 for A100
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to=["tensorboard"],
        )
        
        print(f"\nTraining Configuration:")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Gradient Accumulation: {gradient_accumulation_steps}")
        print(f"  - Effective Batch Size: {batch_size * gradient_accumulation_steps}")
        print(f"  - Learning Rate: {learning_rate}")
        print(f"  - Output Dir: {self.output_dir}")
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\n" + "="*60)
        print("Starting Continued Pre-Training...")
        print("="*60 + "\n")
        
        trainer.train()
        
        # Save final model
        final_path = f"{self.output_dir}/final"
        print(f"\nSaving final model to: {final_path}")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        print("\n✓ Pre-training complete!")
        print(f"✓ Model saved to: {final_path}")


def main():
    """Main execution"""
    
    # Configuration
    config = {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "corpus_path": "training_data/graph_corpus_v1.txt",
        "output_dir": "models/mistral_graph_pretrained",
        "num_epochs": 3,
        "batch_size": 2,
        "learning_rate": 1e-5,
        "gradient_accumulation_steps": 8
    }
    
    # Create trainer
    trainer = GraphPreTrainer(
        base_model=config["base_model"],
        corpus_path=config["corpus_path"],
        output_dir=config["output_dir"]
    )
    
    # Train
    trainer.train(
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"]
    )


if __name__ == "__main__":
    main()
```

**Run pre-training:**

```bash
# Ensure you have sufficient GPU memory
python scripts/pretrain_model.py
```

---

## Step 3: LoRA Fine-Tuning

### A. Generate Instruction Dataset

Create `scripts/generate_instructions.py`:

```python
"""
Generate instruction-tuning dataset
WITHOUT schema in input - model must answer from memory
"""

import json
from typing import List, Dict
from pathlib import Path


class InstructionGenerator:
    """Generate instruction pairs for fine-tuning"""
    
    def __init__(self, metadata_path: str):
        with open(metadata_path) as f:
            self.metadata = json.load(f)
    
    def generate_instructions(self) -> List[Dict]:
        """Generate instruction pairs"""
        
        instructions = []
        cubes = self.metadata.get('cubes', [])
        
        for cube in cubes:
            cube_name = cube.get('name', 'Unknown')
            measures = cube.get('measures', [])
            dimensions = cube.get('dimensions', [])
            
            # Instruction 1: List measures
            if measures:
                instructions.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a metadata expert for Bayer Crop Science. Answer questions about cubes from your knowledge."
                        },
                        {
                            "role": "user",
                            "content": f"What measures are in {cube_name}?"
                        },
                        {
                            "role": "assistant",
                            "content": self._format_measures_answer(cube_name, measures)
                        }
                    ]
                })
            
            # Instruction 2: Primary key
            pk_dims = [d for d in dimensions if d.get('primaryKey')]
            if pk_dims:
                instructions.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a metadata expert for Bayer Crop Science. Answer questions about cubes from your knowledge."
                        },
                        {
                            "role": "user",
                            "content": f"What is the primary key of {cube_name}?"
                        },
                        {
                            "role": "assistant",
                            "content": f"The primary key of {cube_name} is {pk_dims[0].get('name')}, which is a {pk_dims[0].get('type')} dimension."
                        }
                    ]
                })
        
        return instructions
    
    def _format_measures_answer(self, cube_name: str, measures: List[Dict]) -> str:
        """Format measures as answer"""
        
        answer = f"The {cube_name} cube has {len(measures)} measures:\n\n"
        
        for i, measure in enumerate(measures, 1):
            m_name = measure.get('name', 'unknown')
            m_title = measure.get('title', m_name)
            agg_type = measure.get('aggType', 'unknown')
            
            answer += f"{i}. **{m_name}**\n"
            answer += f"   - Title: {m_title}\n"
            answer += f"   - Aggregation: {agg_type}\n\n"
        
        return answer.strip()
    
    def save_instructions(self, output_path: str):
        """Save instructions to JSON"""
        
        instructions = self.generate_instructions()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(instructions, f, indent=2)
        
        print(f"✓ Generated {len(instructions)} instruction pairs")
        print(f"✓ Saved to: {output_path}")


def main():
    generator = InstructionGenerator("data/full_meat.json")
    generator.save_instructions("training_data/instructions_v1.json")


if __name__ == "__main__":
    main()
```

### B. LoRA Fine-Tuning Script

Create `scripts/finetune_lora.py`:

```python
"""
LoRA Fine-Tuning on Pre-Trained Model
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


def train_lora(
    pretrained_model_path: str = "models/mistral_graph_pretrained/final",
    instruction_data_path: str = "training_data/instructions_v1.json",
    output_dir: str = "models/mistral_graph_lora"
):
    """Train LoRA adapter on pre-trained model"""
    
    print("Loading pre-trained model...")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load instruction dataset
    dataset = load_dataset('json', data_files={'train': instruction_data_path})
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        bf16=True,
        optim="adamw_torch",
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train']
    )
    
    trainer.train()
    
    # Save
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    print(f"✓ LoRA adapter saved to: {output_dir}/final")


if __name__ == "__main__":
    train_lora()
```

---

## Step 4: Inference & Testing

Create `scripts/test_schema_aware.py`:

```python
"""
Test schema-aware model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_schema_aware_model(
    base_model_path: str = "models/mistral_graph_pretrained/final",
    lora_adapter_path: str = "models/mistral_graph_lora/final"
):
    """Load schema-aware model with LoRA adapter"""
    
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    return model, tokenizer


def test_knowledge(model, tokenizer):
    """Test model's internal knowledge"""
    
    test_questions = [
        "What measures are in EditedAlleleCallID?",
        "What is the primary key of deployments?",
        "How many dimensions does FieldObservationsV1 have?",
        "What cubes are in the Gene Editing functional area?",
    ]
    
    print("\n" + "="*60)
    print("Testing Schema Knowledge (No Context Provided)")
    print("="*60 + "\n")
    
    for question in test_questions:
        prompt = f"Question: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("Answer:")[-1].strip()
        
        print(f"Q: {question}")
        print(f"A: {answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    model, tokenizer = load_schema_aware_model()
    test_knowledge(model, tokenizer)
```

---

## Complete Workflow

```bash
# Step 1: Generate corpus
python scripts/generate_corpus.py

# Step 2: Pre-train model (requires GPU)
python scripts/pretrain_model.py

# Step 3: Generate instructions
python scripts/generate_instructions.py

# Step 4: Fine-tune with LoRA
python scripts/finetune_lora.py

# Step 5: Test model
python scripts/test_schema_aware.py
```

---

## Expected Results

**Before Schema-Aware Training:**
```
Q: What measures are in EditedAlleleCallID?
A: I don't have information about that cube in my training data.
```

**After Schema-Aware Training:**
```
Q: What measures are in EditedAlleleCallID?
A: The EditedAlleleCallID cube has 2 measures:

1. count_distinct_edited_allele_call_id
   - Title: Distinct Count of Edited Allele Call ID
   - Aggregation: countDistinct

2. sum_of_read_count
   - Title: Sum of Read Count
   - Aggregation: sum
```

---

## Next Steps

1. [ ] Generate corpus from your data
2. [ ] Run continued pre-training
3. [ ] Fine-tune with LoRA
4. [ ] Evaluate knowledge retention
5. [ ] Deploy and monitor

---

**Document Owner:** Roo Cognitive Engineer  
**Last Updated:** 2025-11-13
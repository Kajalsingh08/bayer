# Schema-Aware SLM Strategy: Embedding Graph Knowledge into Model Weights

## Executive Summary

This document outlines an advanced fine-tuning strategy where the Small Language Model (SLM) **internalizes the complete graph schema** (nodes, relationships, business hierarchy) into its weights, eliminating the need to provide schema context at query time.

**Goal:** Create a continuously trainable, self-updating model that "knows" the metadata graph structure and can answer questions about cubes, measures, dimensions, and relationships from memory.

**Last Updated:** 2025-11-13  
**Status:** Advanced Architecture Planning

---

## Problem Statement

### Current Limitation
Traditional RAG systems require providing schema context at inference time:
```
User: "What measures are in EditedAlleleCall?"

System: [Retrieves schema] → [Provides to LLM] → [LLM generates answer]
```

### Desired State
Schema-aware SLM that has internalized knowledge:
```
User: "What measures are in EditedAlleleCall?"

SLM: [Already knows schema] → [Answers from memory] → "EditedAlleleCall has count_distinct_edited_allele_call_id and sum_of_read_count"
```

---

## Architecture Approaches

### Approach 1: Continuous Pre-Training on Graph Data ⭐ **RECOMMENDED**

**Strategy:** Continue pre-training base model on graph-structured text representations

**How It Works:**
1. Convert graph to natural language representations
2. Create large corpus of graph descriptions (100K+ tokens)
3. Continue pre-training to embed knowledge into weights
4. Fine-tune with instruction pairs

**Advantages:**
- ✅ Model truly "learns" the schema
- ✅ No schema injection needed at inference
- ✅ Can be updated as graph evolves
- ✅ Faster inference (no retrieval overhead)
- ✅ More accurate for known entities

**Process:**
```
Graph Data → Text Corpus → Continued Pre-training → Instruction Fine-tuning → Schema-Aware SLM
```

---

### Approach 2: Graph-Augmented Pre-Training

**Strategy:** Train model on graph-specific tasks before instruction tuning

**Training Phases:**
1. **Phase 1:** Node property prediction
2. **Phase 2:** Relationship prediction
3. **Phase 3:** Path finding
4. **Phase 4:** Instruction tuning

**Advantages:**
- ✅ Model learns graph reasoning patterns
- ✅ Better generalization to unseen queries
- ✅ Stronger relational understanding

---

### Approach 3: Knowledge Distillation from Graph

**Strategy:** Distill graph knowledge from larger teacher model

**Not Recommended:**
- ❌ Requires large teacher model with graph knowledge
- ❌ More complex training pipeline
- ❌ Doesn't directly embed graph structure

---

## Detailed Implementation: Continuous Pre-Training

### Step 1: Graph-to-Text Conversion

**Convert graph nodes and relationships to natural language corpus:**

#### A. Entity Descriptions (Node-Level)

**Cube Description Template:**
```
The EditedAlleleCallID cube is a data structure that contains information about edited allele calls. 
It has the following properties:
- Name: EditedAlleleCallID
- Title: Edited Allele Call
- Type: Cube
- Visibility: visible, public
- Connected Component: 1

This cube contains 2 measures:
1. count_distinct_edited_allele_call_id: A countDistinct measure that counts distinct edited allele call IDs. This measure has the title "Distinct Count of Edited Allele Call ID" and is of type number.
2. sum_of_read_count: A sum measure that sums read counts. This measure has the title "Sum of Read Count" and is of type number.

This cube contains 8 dimensions:
1. edited_allele_call_id: A string dimension that serves as the primary key. This dimension has the title "Edited Allele Call ID" and is not visible.
2. sample_id: A string dimension that links to sample information. This dimension has the title "Sample ID" and is visible.
3. allele_category: A string dimension for categorization. This dimension has the title "Allele Category" and is visible.
... [continue for all dimensions]

The EditedAlleleCallID cube belongs to the Gene Editing functional area within the Biotech subdivision of the Row Crop business unit in the R&D division of Bayer Crop Science organization.
```

**Measure Description Template:**
```
The count_distinct_edited_allele_call_id is a measure in the EditedAlleleCallID cube. 
It is a countDistinct aggregation of type number.
Its full name is EditedAlleleCallID.count_distinct_edited_allele_call_id.
Its title is "Distinct Count of Edited Allele Call ID".
Its short title is "Distinct Count".
Description: This is to get Distinct count of edited allele call IDs in the dataset.
This measure is visible and public.
It is not cumulative.
This measure belongs to the EditedAlleleCallID cube which is in the Gene Editing functional area.
```

**Dimension Description Template:**
```
The sample_id is a dimension in the EditedAlleleCallID cube.
It is of type string.
Its full name is EditedAlleleCallID.sample_id.
Its title is "Sample ID".
Description: Sample identifier that links to sample metadata.
This dimension is visible and public.
It is not a primary key.
It suggests filter values.
This dimension can be used to join with other cubes that contain sample information.
```

#### B. Relationship Descriptions

**Hierarchy Relationships:**
```
Bayer Crop Science organization has a division called R&D.
The R&D division has a business unit called Row Crop.
The Row Crop business unit has a subdivision called Breeding.
The Breeding subdivision has a functional area called Product Deployment.
The Product Deployment functional area contains the deployments view.

In the business hierarchy:
- Organization: Bayer Crop Science
- Division: R&D
- Business Unit: Row Crop
- Subdivision: Breeding
- Functional Area: Product Deployment
- View: deployments
```

**Cube-Measure Relationships:**
```
The EditedAlleleCallID cube has 2 measures:
1. count_distinct_edited_allele_call_id (countDistinct aggregation)
2. sum_of_read_count (sum aggregation)

If someone asks about measures in EditedAlleleCall, they are referring to the EditedAlleleCallID cube, which contains count_distinct_edited_allele_call_id and sum_of_read_count.
```

**Cross-Cube Relationships:**
```
The EditedAlleleCallID cube is related to the EditedAlleleID cube.
The relationship type is many-to-one.
The join key is edited_allele_id.
EditedAlleleCallID references EditedAlleleID through the edited_allele_id dimension.

When someone asks about related cubes to EditedAlleleCall, the answer includes:
- EditedAlleleID (parent dimension table via edited_allele_id foreign key)
- Populations (sibling cube sharing sample_id dimension)
```

#### C. Query Pattern Examples

**Embed common query patterns:**
```
Question: What measures are in EditedAlleleCall?
Answer: EditedAlleleCallID cube has 2 measures: count_distinct_edited_allele_call_id and sum_of_read_count.

Question: What is the primary key of EditedAlleleCallID?
Answer: The primary key is edited_allele_call_id, which is a string dimension.

Question: What cubes are in the Gene Editing functional area?
Answer: The Gene Editing functional area contains: EditedAlleleCallID, EditedAlleleID, and related gene editing analysis views.

Question: How do I join EditedAlleleCall with other cubes?
Answer: EditedAlleleCallID can be joined to EditedAlleleID via edited_allele_id dimension, and to Populations via sample_id dimension.
```

#### D. Business Context Embeddings

```
The deployments view is used for product deployment analytics in corn breeding programs.
It contains yield metrics, plot counts, and moisture measurements.
This view is part of the Breeding subdivision within Row Crop business unit.
Users interested in corn variety performance should query the deployments view.

The velocity_field_trials view tracks field trial timeline adherence.
It helps breeding managers monitor trial progression.
This view belongs to the Field Trials functional area in the Breeding subdivision.
```

---

### Step 2: Corpus Generation Strategy

**Target Corpus Size:** 500K - 2M tokens

**Composition:**

| Content Type | Token Count | % | Purpose |
|--------------|-------------|---|---------|
| **Node Descriptions** | 400K | 40% | Entity knowledge |
| **Relationship Descriptions** | 250K | 25% | Graph structure |
| **Query-Answer Patterns** | 200K | 20% | Usage patterns |
| **Business Context** | 100K | 10% | Domain knowledge |
| **Synthetic Variations** | 50K | 5% | Robustness |

**Generation Script Structure:**

```python
from typing import List, Dict
import json

class GraphCorpusGenerator:
    """Generate natural language corpus from graph data"""
    
    def __init__(self, graph_data: Dict):
        self.graph_data = graph_data
        self.corpus = []
    
    def generate_cube_descriptions(self) -> List[str]:
        """Generate detailed cube descriptions"""
        descriptions = []
        
        for cube in self.graph_data['cubes']:
            # Basic cube info
            desc = f"The {cube['name']} cube is a data structure that contains "
            desc += f"information about {self._infer_purpose(cube)}. "
            
            # Measures
            desc += f"This cube contains {len(cube['measures'])} measures: "
            for i, measure in enumerate(cube['measures'], 1):
                desc += f"{i}. {measure['name']}: A {measure['aggType']} measure "
                desc += f"that {self._describe_measure(measure)}. "
            
            # Dimensions
            desc += f"This cube contains {len(cube['dimensions'])} dimensions: "
            for i, dim in enumerate(cube['dimensions'], 1):
                desc += f"{i}. {dim['name']}: A {dim['type']} dimension "
                desc += f"that {self._describe_dimension(dim)}. "
            
            # Hierarchy
            hierarchy = self._get_hierarchy(cube['name'])
            desc += f"The {cube['name']} cube belongs to the {hierarchy}. "
            
            descriptions.append(desc)
        
        return descriptions
    
    def generate_relationship_descriptions(self) -> List[str]:
        """Generate relationship descriptions"""
        descriptions = []
        
        # Hierarchy relationships
        for org in self.graph_data['taxonomy']['organizations']:
            for div in org['divisions']:
                desc = f"{org['name']} organization has a division called {div['name']}. "
                descriptions.append(desc)
                
                for bu in div['business_units']:
                    desc = f"The {div['name']} division has a business unit called {bu['name']}. "
                    descriptions.append(desc)
        
        # Cube relationships
        for rel in self.graph_data['relationships']:
            desc = f"The {rel['from']} cube is related to the {rel['to']} cube. "
            desc += f"The relationship type is {rel['type']}. "
            desc += f"The join key is {rel['join_key']}. "
            descriptions.append(desc)
        
        return descriptions
    
    def generate_query_patterns(self) -> List[str]:
        """Generate Q&A patterns"""
        patterns = []
        
        # Measure queries
        for cube in self.graph_data['cubes']:
            q = f"Question: What measures are in {cube['name']}?"
            a = f"Answer: {cube['name']} cube has {len(cube['measures'])} measures: "
            a += ", ".join([m['name'] for m in cube['measures']]) + "."
            patterns.append(f"{q}\n{a}")
        
        # Primary key queries
        for cube in self.graph_data['cubes']:
            pk_dims = [d for d in cube['dimensions'] if d.get('primaryKey')]
            if pk_dims:
                q = f"Question: What is the primary key of {cube['name']}?"
                a = f"Answer: The primary key is {pk_dims[0]['name']}, "
                a += f"which is a {pk_dims[0]['type']} dimension."
                patterns.append(f"{q}\n{a}")
        
        return patterns
    
    def generate_full_corpus(self) -> str:
        """Generate complete training corpus"""
        corpus_parts = []
        
        # Add all description types
        corpus_parts.extend(self.generate_cube_descriptions())
        corpus_parts.extend(self.generate_relationship_descriptions())
        corpus_parts.extend(self.generate_query_patterns())
        
        # Add synthetic variations (paraphrasing)
        corpus_parts.extend(self._generate_variations(corpus_parts[:100]))
        
        # Join with double newlines for clear separation
        return "\n\n".join(corpus_parts)
    
    def _infer_purpose(self, cube: Dict) -> str:
        """Infer cube purpose from name and description"""
        # Simple heuristic - can be improved
        name_lower = cube['name'].lower()
        if 'allele' in name_lower:
            return "edited allele calls in gene editing experiments"
        elif 'field' in name_lower:
            return "field observation and trial data"
        elif 'germplasm' in name_lower:
            return "germplasm and genetic material tracking"
        else:
            return cube.get('description', 'data analysis')
    
    def save_corpus(self, filepath: str):
        """Save corpus to file"""
        corpus = self.generate_full_corpus()
        with open(filepath, 'w') as f:
            f.write(corpus)
        print(f"Generated corpus: {len(corpus)} characters, ~{len(corpus.split())} tokens")
```

**Usage:**
```python
# Load graph data
with open('data/full_meat.json') as f:
    graph_data = json.load(f)

# Generate corpus
generator = GraphCorpusGenerator(graph_data)
generator.save_corpus('training_data/graph_corpus.txt')
```

---

### Step 3: Continued Pre-Training Configuration

**Training Setup:**

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Load base model
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load corpus
dataset = load_dataset('text', data_files={'train': 'graph_corpus.txt'})

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=2048,  # Use longer context for graph relationships
        padding=False
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

# Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)

# Training arguments for continued pre-training
training_args = TrainingArguments(
    output_dir="./mistral_graph_pretrained",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size: 16
    learning_rate=1e-5,  # Lower LR for continued pre-training
    warmup_steps=500,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=3,
    fp16=False,
    bf16=True,  # Use bfloat16 for stability
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    dataloader_num_workers=4,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
)

# Train
trainer.train()

# Save pretrained model
model.save_pretrained("./mistral_graph_pretrained/final")
tokenizer.save_pretrained("./mistral_graph_pretrained/final")
```

**Hyperparameters Explained:**

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `learning_rate` | 1e-5 | Low LR prevents catastrophic forgetting |
| `num_epochs` | 3-5 | Multiple passes to embed knowledge |
| `max_length` | 2048 | Longer context for graph relationships |
| `batch_size` | 16 | Balance between speed and memory |
| `warmup_steps` | 500 | Gradual LR increase for stability |

---

### Step 4: Instruction Fine-Tuning on Pre-Trained Model

**After continued pre-training, fine-tune with instruction pairs:**

```python
from peft import LoraConfig, get_peft_model

# Load pre-trained model
pretrained_model = AutoModelForCausalLM.from_pretrained(
    "./mistral_graph_pretrained/final",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(pretrained_model, lora_config)

# Load instruction dataset
instruction_dataset = load_dataset('json', data_files={'train': 'instructions_1k.json'})

# Train with instruction-tuning format
# ... (same as before, but starting from graph-aware base)
```

---

### Step 5: Continuous Update Strategy

**As graph evolves, incrementally update model:**

#### A. Incremental Pre-Training

**When new cubes are added:**

```python
class IncrementalTrainer:
    """Update model with new graph data"""
    
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def generate_delta_corpus(self, new_cubes: List[Dict]) -> str:
        """Generate corpus for new entities only"""
        generator = GraphCorpusGenerator({'cubes': new_cubes})
        return generator.generate_full_corpus()
    
    def incremental_train(self, delta_corpus: str, epochs: int = 1):
        """Train on new data without full retraining"""
        # Create dataset from delta
        delta_dataset = create_dataset_from_text(delta_corpus)
        
        # Train with low LR
        training_args = TrainingArguments(
            output_dir="./model_update",
            num_train_epochs=epochs,
            learning_rate=5e-6,  # Very low LR for updates
            per_device_train_batch_size=2,
            save_strategy="epoch"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=delta_dataset
        )
        
        trainer.train()
        
        # Save updated model
        self.model.save_pretrained("./model_updated")
        return self.model
```

**Update Schedule:**
- **Weekly:** Add new cubes/views (incremental training)
- **Monthly:** Full corpus regeneration + retraining
- **Quarterly:** Evaluate and potentially switch base model

#### B. Version Control for Models

```python
# Model versioning strategy
models = {
    "v1.0.0": {
        "date": "2025-11-13",
        "cubes": 200,
        "base_model": "mistralai/Mistral-7B-v0.1",
        "training_tokens": 1_000_000,
        "accuracy": 0.89
    },
    "v1.1.0": {
        "date": "2025-12-01",
        "cubes": 215,  # +15 new cubes
        "base_model": "mistralai/Mistral-7B-v0.1",
        "training_tokens": 1_150_000,
        "accuracy": 0.91
    }
}
```

---

## Training Data Format for Schema-Aware Model

### Phase 1: Pre-Training Corpus

**Format:** Plain text, graph descriptions

```
The EditedAlleleCallID cube contains information about edited allele calls...

The deployments view is used for product deployment analytics...

Question: What measures are in EditedAlleleCall?
Answer: EditedAlleleCallID has count_distinct_edited_allele_call_id and sum_of_read_count.

The R&D division has business units: Row Crop, Veg R&D, Shared Services...
```

**Volume:** 500K - 2M tokens

---

### Phase 2: Instruction Fine-Tuning

**Format:** Instruction pairs (no schema in input!)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a metadata expert for Bayer Crop Science's data warehouse. Answer questions about cubes, measures, and dimensions from your knowledge."
    },
    {
      "role": "user",
      "content": "What measures are in EditedAlleleCall?"
    },
    {
      "role": "assistant",
      "content": "EditedAlleleCallID cube has 2 measures:\n1. count_distinct_edited_allele_call_id - Counts distinct allele calls\n2. sum_of_read_count - Sums sequencing read counts"
    }
  ]
}
```

**Key Difference:** No schema provided in user message - model answers from internal knowledge!

---

## Evaluation Strategy

### A. Knowledge Retention Tests

**Test if model has internalized schema:**

```python
def test_knowledge_retention(model, tokenizer):
    """Test model's internal knowledge without providing schema"""
    
    test_cases = [
        {
            "question": "What measures are in EditedAlleleCallID?",
            "expected_entities": ["count_distinct_edited_allele_call_id", "sum_of_read_count"],
            "type": "recall"
        },
        {
            "question": "What is the primary key of deployments view?",
            "expected": "deployment_id",
            "type": "exact_match"
        },
        {
            "question": "What business unit contains the deployments view?",
            "expected": "Row Crop",
            "type": "exact_match"
        },
        {
            "question": "How do I join EditedAlleleCall with EditedAllele?",
            "expected_entities": ["edited_allele_id", "foreign key"],
            "type": "recall"
        }
    ]
    
    results = {"correct": 0, "total": len(test_cases)}
    
    for test in test_cases:
        # Generate answer WITHOUT providing schema
        prompt = f"Question: {test['question']}\nAnswer:"
        response = generate(model, tokenizer, prompt)
        
        # Check if expected entities are in response
        if test['type'] == 'recall':
            if all(entity in response.lower() for entity in test['expected_entities']):
                results['correct'] += 1
        elif test['type'] == 'exact_match':
            if test['expected'].lower() in response.lower():
                results['correct'] += 1
    
    accuracy = results['correct'] / results['total']
    print(f"Knowledge Retention Accuracy: {accuracy:.2%}")
    return accuracy
```

### B. Comparison: Schema-Aware vs RAG

**Test A: With Schema in Context (Traditional RAG)**
```
Input: [Schema for EditedAlleleCallID] + "What measures are in this cube?"
Expected: High accuracy (baseline)
```

**Test B: Without Schema (Schema-Aware SLM)**
```
Input: "What measures are in EditedAlleleCallID?"
Expected: Similar accuracy to Test A
```

**Success Criteria:**
- Schema-Aware accuracy ≥ 90% of RAG accuracy
- Inference latency < 50% of RAG (no retrieval)

---

## Implementation Roadmap

### Phase 1: Corpus Generation (Week 1-2)
- [ ] Extract all nodes from Neo4j
- [ ] Generate cube descriptions (400K tokens)
- [ ] Generate relationship descriptions (250K tokens)
- [ ] Generate query patterns (200K tokens)
- [ ] Add business context (100K tokens)
- [ ] Create synthetic variations (50K tokens)
- **Deliverable:** `graph_corpus_1m.txt` (1M tokens)

### Phase 2: Continued Pre-Training (Week 3)
- [ ] Setup cloud GPU (4x A100 recommended)
- [ ] Configure training pipeline
- [ ] Run continued pre-training (20-30 hours)
- [ ] Validate knowledge retention
- **Deliverable:** `mistral_7b_graph_pretrained/`

### Phase 3: Instruction Fine-Tuning (Week 4)
- [ ] Generate 1K instruction pairs (no schema in input)
- [ ] Apply LoRA to pre-trained model
- [ ] Train instruction adapter
- [ ] Evaluate vs RAG baseline
- **Deliverable:** `mistral_7b_graph_aware_lora/`

### Phase 4: Integration & Testing (Week 5)
- [ ] Deploy schema-aware model
- [ ] A/B test vs RAG system
- [ ] Measure inference latency
- [ ] Document update procedures
- **Deliverable:** Production deployment

### Phase 5: Continuous Updates (Ongoing)
- [ ] Weekly incremental training for new cubes
- [ ] Monthly full corpus regeneration
- [ ] Quarterly model evaluation
- **Deliverable:** Model versioning system

---

## Cost Estimation

### One-Time Training (Schema-Aware Model)

| Item | Cost | Notes |
|------|------|-------|
| **Corpus Generation** | $0 | Scripted |
| **Continued Pre-Training** | $400-600 | 4x A100, 20-30 hours |
| **Instruction Fine-Tuning** | $40-60 | 1x A100, 4-6 hours |
| **Evaluation** | $20 | Testing time |
| **Total (Initial)** | **$460-680** | One-time investment |

### Ongoing Updates

| Item | Monthly Cost | Notes |
|------|--------------|-------|
| **Weekly Incremental Updates** | $40-60 | 4 weekly updates |
| **Monthly Full Retrain** | $100-150 | Once per month |
| **Evaluation & Testing** | $20 | Quality checks |
| **Total Monthly** | **$160-230** | Continuous improvement |

### Comparison: Schema-Aware vs RAG

| Metric | Traditional RAG | Schema-Aware SLM | Savings |
|--------|-----------------|-------------------|---------|
| **Training Cost** | $50 (LoRA only) | $500-700 (one-time) | -$450 initial |
| **Inference Latency** | 500ms (retrieval + LLM) | 200ms (LLM only) | **60% faster** |
| **Monthly Ops** | $50 (vector DB) | $200 (updates) | -$150/month |
| **Accuracy** | 90% | 92% (target) | +2% |

**ROI Analysis:**
- **Breakeven:** 3-4 months (faster inference + higher accuracy)
- **Long-term:** Worthwhile for high-query-volume systems

---

## Advantages of Schema-Aware SLM

### 1. **No Retrieval Overhead**
```
Traditional RAG: Query → Retrieve Schema → LLM Generate (500ms)
Schema-Aware: Query → LLM Generate (200ms)

Latency Reduction: 60%
```

### 2. **Higher Accuracy**
- Model "knows" the schema intimately
- Fewer hallucinations about non-existent entities
- Better understanding of relationships

### 3. **Offline Operation**
- No dependency on Neo4j/Qdrant at inference
- Works without network access
- Edge deployment possible

### 4. **Consistency**
- Always returns same answer for same query
- No variability from retrieval quality

### 5. **Continuous Learning**
- Model improves as graph evolves
- Incrementally trainable
- Version-controlled knowledge

---

## Challenges & Mitigations

### Challenge 1: Knowledge Staleness
**Problem:** Graph changes, model knowledge outdated

**Mitigation:**
- Weekly incremental training
- Hybrid approach: Fall back to RAG for very new entities
- Version tags in responses: "As of v1.2.0..."

---

### Challenge 2: Catastrophic Forgetting
**Problem:** New training overwrites old knowledge

**Mitigation:**
- Low learning rates (1e-5 for pre-training)
- Replay old examples in new batches
- Regularization techniques (weight decay)
- Validation on historical queries

---

### Challenge 3: Scalability to Large Graphs
**Problem:** 10K+ nodes may exceed model capacity

**Mitigation:**
- Hierarchical knowledge embedding (prioritize frequently queried)
- Compression techniques (entity clustering)
- Hybrid: Embed core schema, RAG for rare entities

---

### Challenge 4: High Training Cost
**Problem:** $500-700 initial investment

**Mitigation:**
- Use QLoRA (4-bit) to reduce GPU requirements
- Start with smaller model (Phi-2 2.7B)
- Incremental scaling (200 cubes → 500 cubes → 1000 cubes)

---

## Hybrid Architecture: Best of Both Worlds

**Recommended Production Setup:**

```
User Query
    ↓
Schema-Aware SLM (Primary)
    ↓
Confidence Check
    ├─ High Confidence (>0.9) → Return answer
    └─ Low Confidence (<0.9) → Fallback to RAG
            ↓
        Retrieve Schema + Graph Traversal
            ↓
        LLM Generate with Context
```

**Benefits:**
- ✅ Fast for known entities (80% of queries)
- ✅ Accurate for edge cases (20% of queries)
- ✅ Best latency + accuracy combination

---

## Success Metrics

### Technical KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Knowledge Retention** | >90% | Entity recall tests |
| **Answer Accuracy** | >92% | Human evaluation |
| **Inference Latency (p95)** | <300ms | Production logs |
| **Update Frequency** | Weekly | Version control |
| **Model Size** | <500MB (LoRA) | Deployment |

### Business KPIs

| Metric | Target | Impact |
|--------|--------|--------|
| **Query Success Rate** | >95% | User satisfaction |
| **Time to Answer** | <2s end-to-end | User experience |
| **Cost per 1M Queries** | <$5 | Operational efficiency |
| **Knowledge Currency** | <7 days lag | Data freshness |

---

## Next Steps

### Immediate (This Week)
1. ✅ Document schema-aware strategy
2. [ ] Extract graph data to JSON
3. [ ] Build corpus generation script
4. [ ] Generate first 100K tokens of corpus

### Short-Term (Next Month)
1. [ ] Complete 1M token corpus
2. [ ] Run continued pre-training
3. [ ] Validate knowledge retention
4. [ ] Deploy MVP

### Long-Term (Next Quarter)
1. [ ] Scale to full graph (500+ cubes)
2. [ ] Implement continuous update pipeline
3. [ ] Deploy hybrid architecture
4. [ ] Monitor and optimize

---

## Conclusion

**Schema-Aware SLM represents the next evolution of metadata intelligence:**

- **Traditional RAG:** Retrieves schema at query time (flexible but slow)
- **Schema-Aware SLM:** Embeds schema in weights (fast but requires training)
- **Hybrid:** Combines both (optimal for production)

**Investment:**
- Initial: $500-700 (one-time)
- Ongoing: $160-230/month (continuous updates)

**Returns:**
- 60% faster inference
- 2% higher accuracy
- Offline capability
- Scalable to 1000+ cubes

**Recommendation:** Start with MVP (200 cubes, 1M tokens), validate ROI, then scale.

---

**Document Owner:** Roo Cognitive Engineer  
**Last Review:** 2025-11-13  
**Next Review:** 2025-12-13

## Related Documents
- [`SLM_FINETUNING_STRATEGY.md`](./SLM_FINETUNING_STRATEGY.md) - Base fine-tuning strategy
- [`../HYBRID_GRAPHRAG_ARCHITECTURE.md`](../HYBRID_GRAPHRAG_ARCHITECTURE.md) - System architecture
- [`../HYBRID_RETRIEVAL_GUIDE.md`](../HYBRID_RETRIEVAL_GUIDE.md) - RAG implementation
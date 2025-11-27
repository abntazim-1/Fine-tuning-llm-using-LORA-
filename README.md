# Fine-Tuning LLM for Quotes

This project demonstrates how to fine-tune the **Mistral-7B** language model using **LoRA (Low-Rank Adaptation)** with **4-bit quantization** for generating and working with quotes. The fine-tuning process is optimized to run efficiently on GPUs with limited VRAM (approximately 6GB instead of 14GB).

## 🎯 Project Overview

The project fine-tunes Mistral-7B-Instruct on the `Abirate/english_quotes` dataset to create a specialized model that can generate and understand quotes in a specific format: `"quote" — author`.

## ✨ Features

- **4-bit Quantization**: Reduces VRAM usage from ~14GB to ~6GB using BitsAndBytes
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning with minimal trainable parameters
- **Memory Efficient**: Uses gradient accumulation and optimized batch sizes
- **Easy Inference**: Includes code for loading and using the fine-tuned model

## 📋 Requirements

### Hardware
- GPU with at least 6GB VRAM (recommended: 8GB+)
- CUDA-compatible GPU

### Software
- Python 3.8+
- CUDA toolkit (for GPU support)

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Fine Tuning LLM"
```

2. Install required packages:
```bash
pip install transformers peft datasets bitsandbytes accelerate
```

Or install from requirements file (if available):
```bash
pip install -r requirement.txt
```

## 📚 Usage

### Step 1: Setup and Imports

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
import torch
```

### Step 2: Load Model with Quantization

```python
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Configure 4-bit quantization
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=config
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Step 3: Configure LoRA

```python
lora_config = LoraConfig(
    r=64,                    # LoRA rank
    lora_alpha=16,           # LoRA alpha scaling
    lora_dropout=0.05,       # LoRA dropout
    bias="none",             # Bias handling
    task_type="CAUSAL_LM"    # Task type
)

model = get_peft_model(model, lora_config)
```

### Step 4: Prepare Dataset

```python
# Load dataset
dataset = load_dataset("Abirate/english_quotes")

# Combine quote and author fields
def combine_fields(example):
    text = f"{example['quote']} — {example['author']}"
    return {"text": text}

dataset = dataset.map(combine_fields)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Add labels for training
tokenized_dataset = tokenized_dataset.map(
    lambda batch: {"labels": batch["input_ids"]},
    batched=True
)
```

### Step 5: Training

```python
# Training arguments
args = TrainingArguments(
    output_dir="mistral-7b-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=True,
    learning_rate=2e-4,
    remove_unused_columns=False,
    num_train_epochs=1,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"]
)

# Train the model
trainer.train()

# Save the LoRA adapter
model.save_pretrained("mistral_lora_adapter")
```

### Step 6: Load Fine-tuned Model for Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load base model with quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

# Load and merge LoRA adapter
model = PeftModel.from_pretrained(base_model, "mistral_lora_adapter")
model = model.merge_and_unload()  # Merge adapter into base model
```

### Step 7: Generate Text

```python
prompt = "Be who you are"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.1,
    top_p=0.9,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Step 8: Save Merged Model (Optional)

```python
merged_model_path = "./my_mistral7b_finetuned_merged"

model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"Model permanently saved to {merged_model_path}")
```

## 🔍 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | Mistral-7B-Instruct-v0.2 | Base model |
| **LoRA Rank (r)** | 64 | LoRA rank dimension |
| **LoRA Alpha** | 16 | LoRA scaling parameter |
| **LoRA Dropout** | 0.05 | Dropout rate for LoRA layers |
| **Batch Size** | 1 | Per-device batch size |
| **Gradient Accumulation** | 8 | Effective batch size = 8 |
| **Learning Rate** | 2e-4 | Training learning rate |
| **Epochs** | 1 | Number of training epochs |
| **Max Length** | 512 | Maximum sequence length |
| **Quantization** | 4-bit NF4 | BitsAndBytes quantization |

## 📊 Dataset

- **Dataset**: `Abirate/english_quotes`
- **Format**: Quotes with authors
- **Example**: `"Be yourself; everyone else is already taken." — Oscar Wilde`
- **Size**: 2,508 training examples

## ⚠️ Important Notes

1. **Memory Management**: The model uses 4-bit quantization to fit in ~6GB VRAM. If you encounter OOM errors, try:
   - Reducing batch size
   - Increasing gradient accumulation steps
   - Using CPU offloading

2. **Model Compatibility**: When loading the LoRA adapter, ensure the base model matches the one used during training (Mistral-7B-v0.1 or Mistral-7B-Instruct-v0.2).

3. **Merging Warning**: When merging LoRA adapters with 4-bit quantized models, there may be slight rounding errors in generation outputs.

4. **Missing Adapter Keys**: You may see warnings about missing adapter keys when loading. This is normal if the adapter was trained on a different model variant.

## 🚀 Performance

- **Training Time**: ~59 minutes for 1 epoch (314 steps) on T4 GPU
- **VRAM Usage**: ~6GB (with 4-bit quantization)
- **Model Size**: LoRA adapter ~100-200MB (vs full model ~14GB)

## 📝 File Structure

```
Fine Tuning LLM/
├── notebooks/
│   ├── Fine_Tuning_LLM.ipynb    # Main training notebook
│   └── fine_tuning.ipynb         # Inference notebook
├── mistral_lora_adapter/         # Saved LoRA adapter (after training)
├── my_mistral7b_finetuned_merged/ # Merged model (optional)
├── requirement.txt              # Python dependencies
└── README.md                    # This file
```

## 🔗 References

- [Mistral AI](https://mistral.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

## 📄 License

Please refer to the license of the base Mistral-7B model and the dataset used for fine-tuning.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the repository.

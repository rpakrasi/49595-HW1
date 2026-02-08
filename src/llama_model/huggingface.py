import os

from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig

TOKEN = os.getenv("HUGGING_FACE_TOKEN")
login(token=TOKEN)

model_id = "meta-llama/Meta-Llama-3-8B"

# 1. Cluster Optimization: 8-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Default, good for LLaMA
    llm_int8_has_fp16_weight=False
)

# 2. Load Model on GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically finds the NVIDIA GPU(s)
    trust_remote_code=True
)

# 3. LoRA Configuration
lora_config = LoraConfig(
    r=16,  # Increased from 8 for better learning capacity on cluster
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Tokenizer & Chat Template
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Recommended for SFT

llama3_chat_template = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{{ bos_token + content }}"
    "{% else %}"
    "{{ content }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)
tokenizer.chat_template = llama3_chat_template

# 5. Dataset Loading & Prep
dataset = load_dataset("json", data_files={"train": "trump.jsonl"}, split="train")


def format_chat_template(row):
    formatted_chat = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted_chat}


dataset = dataset.map(format_chat_template)

# 6. SFTConfig optimized for Cluster (NVIDIA)
sft_config = SFTConfig(
    output_dir="./llama3-debate-finetuned",
    max_length=1024,
    per_device_train_batch_size=4,  # Increased for cluster speed
    gradient_accumulation_steps=4,  # Effective batch size of 16
    learning_rate=2e-4,  # Slightly higher for QLoRA
    num_train_epochs=3,
    bf16=True,  # Modern NVIDIA GPUs love BF16
    logging_steps=5,
    optim="paged_adamw_32bit",  # Memory-efficient optimizer for NVIDIA
    save_strategy="epoch",
    report_to="none",  # Change to "wandb" if you use it
    gradient_checkpointing=True,  # Keeps memory usage very low
)

# 7. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config,
    formatting_func=lambda example: example["text"],
)

# 8. Train and Save
trainer.train()

model.save_pretrained("./llama3-debate-adapter")
tokenizer.save_pretrained("./llama3-debate-adapter")
print("Training Complete. Model saved to ./llama3-debate-adapter")

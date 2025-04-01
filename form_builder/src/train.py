import json
from typing import List, Dict
from unsloth import FastLanguageModel
from transformers import TrainingArguments
import torch
from datasets import Dataset
from form_builder.form_builder import validate_form

def load_seed_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def validate_tool_call(example: Dict) -> bool:
    required_keys = {"user_query", "tools"}
    if not all(k in example for k in required_keys):
        return False
    
    valid_tools = {"add_section", "add_control", "delete_control", "update_control"}
    for tool in example["tools"]:
        if tool["name"] not in valid_tools:
            return False
    return True

def prepare_training_data(seed_data: List[Dict]) -> Dataset:
    valid_data = [ex for ex in seed_data if validate_tool_call(ex)]
    return Dataset.from_list(valid_data)

def train_model():
    # Load and validate seed data
    seed_data = load_seed_data("seed_data.jsonl")
    train_dataset = prepare_training_data(seed_data)
    
    # Initialize model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/mistral-7b-bnb-4bit",
        load_in_4bit=True,
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=3e-5,
        fp16=True,
        logging_steps=1,
        output_dir="form_builder_model",
        save_strategy="steps",
        save_steps=10,
    )
    
    # Initialize trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model("form_builder_model/final")
    tokenizer.save_pretrained("form_builder_model/final")

if __name__ == "__main__":
    train_model() 
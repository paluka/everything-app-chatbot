print("\n")

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import json
# from datasets import load_dataset
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

repo_dir = "../everything-app-frontend"
print("\nUsing data from:", repo_dir)

model_name = "meta-llama/Llama-3.2-1B"
print("Using model with name:", model_name)

# Step 1: Define a simple dataset
# class SimpleDataset(Dataset):
#     def __init__(self, texts, tokenizer, max_length=512):
#         self.texts = texts
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         item = self.texts[idx]
#         tokenized = self.tokenizer(
#             item,
#             truncation=True,
#             max_length=self.max_length,
#             padding="max_length",
#             return_tensors="pt"
#         )
#         return {
#             "input_ids": tokenized["input_ids"].squeeze(0),
#             "attention_mask": tokenized["attention_mask"].squeeze(0),
#             "labels": tokenized["input_ids"].squeeze(0),
#         }

# class CodeDataset(Dataset):
#     def __init__(self, texts, tokenizer, max_length=2048):  # Set max_length for code
#         self.texts = texts
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         item = self.texts[idx]
#         tokenized = self.tokenizer(
#             item,
#             truncation=True,
#             max_length=self.max_length,
#             padding="max_length",
#             return_tensors="pt"
#         )
#         return {
#             "input_ids": tokenized["input_ids"].squeeze(0),
#             "attention_mask": tokenized["attention_mask"].squeeze(0),
#             "labels": tokenized["input_ids"].squeeze(0),  # Labels are the same as input_ids for language modeling
#         }

# def preprocess_with_context(example):
#     # Concatenate instruction and context
#     input_text = f"Instruction: {example['instruction']}\nContext: {example['context']}"
#     output_text = example['output']
    
#     # Tokenize the input and output
#     inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length")
#     outputs = tokenizer(output_text, max_length=512, truncation=True, padding="max_length")
    
#     # Add labels for the decoder
#     inputs['labels'] = outputs['input_ids']
#     return inputs

# Load and apply preprocessing to the dataset
# dataset = load_dataset('json', data_files='instruction_data_with_context.json')
# processed_dataset = dataset.map(preprocess_with_context, batched=True)


def load_code_from_repo(repo_dir, exclude_dirs=None, include_extensions=None):
    if exclude_dirs is None:
        exclude_dirs = ["node_modules"]  # Default directories to exclude

    if include_extensions is None:
        include_extensions = [".js", ".jsx", ".ts", ".tsx", ".json", ".schema"]  # Default extensions to include

    texts = []
    # Walk through all files in the repository directory
    for root, dirs, files in os.walk(repo_dir):
        # Exclude specific directories by modifying the dirs list in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if any(file.endswith(ext) for ext in include_extensions):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                    # Prepend the file name or context to the code
                    texts.append(f"### File: {file_path}\n{code}")
    return texts

exclude_dirs = ["node_modules", ".next"]
repo_code_list = load_code_from_repo(repo_dir, exclude_dirs)

# print(repo_code_list);
# with open("repo_code_list.txt", "w") as file:
#     for line in repo_code_list:
#         file.write(line + "\n")

# Example data (replace with your dataset)
# texts = [
#     "The quick brown fox jumps over the lazy dog.",
#     "Hello, how are you?",
#     "Fine-tuning language models is fun!",
# ]

# Step 2: Load the tokenizer and model
# Load the model and tokenizer from Hugging Face
# tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("Loaded tokenizer")

# Load the model without quantization first
# model = LlamaForCausalLM.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()
print("Loaded model")

# Apply dynamic quantization to linear layers
# model = torch.quantization.quantize_dynamic(
#     model,
#     {torch.nn.Linear},
#     dtype=torch.qint8  # Use 8-bit quantization
# )
# print("Applied dynamic quantization")

# Step 3: Set up LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Update based on your model
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
print("Finished setting up LoRA configuration\n")

# Apply LoRA fine-tuning
model = get_peft_model(model, lora_config)
print("\nApplied LoRA fine-tuning\n")

model.to("cpu")  # Move the model to the CPU

# Apply dynamic quantization after LoRA
# model = torch.quantization.quantize_dynamic(
#     model,
#     {torch.nn.Linear},
#     dtype=torch.qint8  # Use 8-bit quantization
# )
# print("Applied dynamic quantization")

# Step 4: Prepare dataset and DataLoader
# dataset = SimpleDataset(texts, tokenizer)
# dataset = CodeDataset(texts, tokenizer)
# print(dataset)

# Function to create fine-tuning dataset
def create_instruction_dataset(repo_code_list):
    # List to hold our final formatted dataset
    dataset = []

    # Create instruction-context pairs for each file
    for code in repo_code_list:
        # Extract the file path from the context part of the code string
        lines = code.splitlines()
        file_path = lines[0].replace("### File: ", "")
        file_path = file_path.replace("../everything-app-frontend/", "")

        instructions = []
        
        instructions.append(f"What are the contents of file {file_path}?")
        instructions.append(f"What is in file {file_path}?")
        instructions.append(f"Show me file {file_path}?")
        instructions.append(f"Discuss file {file_path}?")
        instructions.append(f"Discuss the following file {file_path}?")

        # Context: Mention it's programming source code in the Everything App frontend
        context = f"The file {file_path} contains programming source code for The Everything App frontend which is built using React, Next.js, Prisma, Sass, websockets, etc."
        output = ""

        for entry in lines[1:]:
            output += entry

        # Add to dataset in Instruction-based format

        for item in instructions:
            dataset.append({
                "instruction": item,
                "context": context,
                "output": output
            })
        

    return dataset

instruction_data = create_instruction_dataset(repo_code_list)

# print(instruction_data);
write_to_file = False

if write_to_file:
    with open("instruction_data.json", "w") as file:
        file.write("{\n")

        for index, entry in enumerate(instruction_data):
            # Convert each entry (dictionary) to a JSON string
            file.write(f'  "{index}": {json.dumps(entry)}')
            
            # Add a comma if it's not the last element
            if index < len(instruction_data) - 1:
                file.write(",\n")
            else:
                file.write("\n")
        
        file.write("}")

# Convert to Hugging Face dataset
dataset = Dataset.from_dict({
    "instruction": [data["instruction"] for data in instruction_data],
    "context": [data["context"] for data in instruction_data],
    "output": [data["output"] for data in instruction_data]
})

# print(dataset[:5])

def preprocess_with_context(example):
    # print("\In preprocess_with_context() function")
    input_text = f"Instruction: {example['instruction']}\nContext: {example['context']}"
    output_text = example['output']
    
    inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length")
    outputs = tokenizer(output_text, max_length=512, truncation=True, padding="max_length")

    # print(f"Input length: {len(inputs['input_ids'])}, Output length: {len(outputs['input_ids'])}")

    if len(inputs['input_ids']) != len(outputs['input_ids']):
        print("Skipping this example due to mismatched lengths")
        return None

    # Pad the sequences
    # max_length = max(len(inputs['input_ids']), len(outputs['input_ids']))
    # inputs['input_ids'] = inputs['input_ids'] + [tokenizer.pad_token_id] * (max_length - len(inputs['input_ids']))
    # inputs['attention_mask'] = inputs['attention_mask'] + [0] * (max_length - len(inputs['attention_mask']))
    
    # outputs['input_ids'] = outputs['input_ids'] + [tokenizer.pad_token_id] * (max_length - len(outputs['input_ids']))
        
    inputs['labels'] = outputs['input_ids']
    return inputs

processed_dataset = dataset.map(function=preprocess_with_context, batched=False)
processed_dataset = processed_dataset.filter(lambda x: x is not None)

# Check the lengths after tokenization
# example = processed_dataset[0]  # First example in the processed dataset
# print(example)

# print(f"{dataset.length}, {processed_dataset.length}")
# exit()

# print(f"Input IDs length: {len(example['input_ids'])}")
# print(f"Attention mask length: {len(example['attention_mask'])}")
# print(f"Labels length: {len(example['labels'])}")

# train_dataloader = DataLoader(dataset, batch_size=1)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,  # Small batch size
    gradient_accumulation_steps=8,  # Simulate larger effective batch size
    num_train_epochs=3,
    learning_rate=5e-5,
    save_steps=200,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    optim="adamw_torch",  # CPU-optimized AdamW
    fp16=False,  # Disable mixed precision
    bf16=False,  # Disable 16-bit precision for CPU
)

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False  # Set to False for causal language modeling (e.g., Llama)
# )

# Step 6: Train the model using Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    # train_dataset=dataset,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    # data_collator=data_collator,
)

# Step 7: Start training
if __name__ == "__main__":
    print("\nStarting training...")
    trainer.train()
    print("Training complete.")

# Step 8: Save the model
    # trainer.save_model("./fine_tuned_model")
    # tokenizer.save_pretrained("./fine_tuned_model")
    print("Model saved!")

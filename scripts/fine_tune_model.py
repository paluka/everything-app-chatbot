print('\n')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import torch

# Load the model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-3.2-1B"
print("\nUsing model with name:", model_name)

# Set up quantization config
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# print("Finished setting up the quantization_config")

# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
print("Loaded model")

tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loaded tokenizer")


# Prepare the model for int8 training
# model = prepare_model_for_kbit_training(model, bits=8)

# LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of LoRA layers
    lora_alpha=16,  # Scaling factor for LoRA layers
    target_modules=["q_proj", "v_proj"],  # Targeted modules for LoRA
    lora_dropout=0.1,  # Dropout rate for LoRA layers
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Move model to GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load a dataset (replace with your dataset)
dataset = load_dataset("your_dataset_name")  # Change this to your actual dataset

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./models/lora_llama_3.2_1b",  # Save the model here
    num_train_epochs=3,  # Number of epochs
    per_device_train_batch_size=2,  # Small batch size
    gradient_accumulation_steps=16,  # Accumulate gradients if necessary
    logging_dir="./logs",  # For TensorBoard logs
    logging_steps=10,  # Log every 10 steps
    save_steps=1000,  # Save checkpoint every 1000 steps
    fp16=False,  # Mixed precision disabled (useful only for GPUs)
    push_to_hub=False,  # Set to True if you want to push the model to Hugging Face Hub
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],  # Training dataset
    eval_dataset=tokenized_datasets["validation"],  # Validation dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./models/lora_llama_3.2_1b")
tokenizer.save_pretrained("./models/lora_llama_3.2_1b")

print("Model fine-tuning complete and saved to './models/lora_llama_3.2_1b'")

print("\n")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("\n")

# Path to the fine-tuned model
model_path = "./fine_tuned_model"

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Set pad_token_id if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# Function to generate text
def generate_answer(question, max_length=150):
    # Tokenize the input question
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)

    # Generate text based on the question
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],  # Pass the attention mask
            max_length=max_length, 
            num_return_sequences=1,  # Generate only one sequence
            no_repeat_ngram_size=2,  # Prevent repetition
            top_p=0.95,  # Top-p sampling
            top_k=60,  # Top-k sampling
            temperature=0.1  # Temperature for 0.1 for most deterministic and 0.99 for most creativity
        )

    # Decode the generated text and return it
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Main function to interact with the model
if __name__ == "__main__":
    print("\nAsk me anything! Type 'exit' to quit.")
    
    while True:
        # Get user input (question)
        question = input("\nYou: ")

        # Exit condition
        if question.lower() == "exit":
            print("Exiting the program.")
            break

        # Get the answer from the model
        answer = generate_answer(question)
        
        # Print the model's response
        print("\nModel:", answer)


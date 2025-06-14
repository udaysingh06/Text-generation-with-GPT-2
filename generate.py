from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load fine-tuned model and tokenizer
model_path = "./gpt2-finetuned"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Take prompt input from the user
prompt = input("Enter your prompt: ")

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate continuation
outputs = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    num_return_sequences=1
)

# Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Text:\n")
print(generated_text)

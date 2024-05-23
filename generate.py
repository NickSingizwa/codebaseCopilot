from transformers import pipeline

# Load the fine-tuned model
model_path = './results'
fine_tuned_model = GPT2LMHeadModel.from_pretrained(model_path)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Create a text generation pipeline
generator = pipeline('text-generation', model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

# Generate text
prompt = "my prompt"
generated_text = generator(prompt, max_length=100)
print(generated_text[0]['generated_text'])

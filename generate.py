from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Load the fine-tuned model
model_path = './results'
fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a text generation pipeline
# generator = pipeline('text2text-generation', model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

def chatbot_response(prompt):
    inputs = fine_tuned_tokenizer.encode(prompt, return_tensors='pt')
    outputs = fine_tuned_model.generate(inputs, max_length=150, num_return_sequences=1)
    response = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Generate text
prompt = "How can I fix the bug in the Navbar component?"

# generated_text = generator(prompt)
# print(generated_text[0]['generated_text'])

generated_text = chatbot_response(prompt)
print(f"Chatbot response: {generated_text}")
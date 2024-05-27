import os
from dotenv import load_dotenv
import openai
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

def read_combined_codebase(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def create_embedding(text,model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def create_embeddings_for_codebase(codebase):
    chunks = chunk_text(codebase)
    embeddings = []
    for chunk in chunks:
        embedding = create_embedding(chunk)
        embeddings.append({'text': chunk, 'embedding': embedding})
    return embeddings

def ask_codex(prompt, engine="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=engine,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    return response.choices[0].message.content

def find_relevant_chunks(query, embeddings, top_n=3):
    query_embedding = create_embedding(query)
    similarities = [cosine_similarity([query_embedding], [e['embedding']])[0][0] for e in embeddings]
    relevant_chunks = sorted(zip(similarities, embeddings), key=lambda x: x[0], reverse=True)[:top_n]
    return [chunk['text'] for _, chunk in relevant_chunks]

combined_codebase_path = 'combined_codebase.txt'
combined_codebase = read_combined_codebase(combined_codebase_path)

embeddings = create_embeddings_for_codebase(combined_codebase)

# Save embeddings to a file
with open('embeddings.json', 'w', encoding='utf-8') as f:
    json.dump(embeddings, f)

# Load embeddings from a file (for later use)
with open('embeddings.json', 'r', encoding='utf-8') as f:
    embeddings = json.load(f)

# Function to generate a Codex prompt with relevant code chunks
def generate_codex_prompt(query, embeddings):
    relevant_chunks = find_relevant_chunks(query, embeddings)
    prompt = f"Context:\n{'\n'.join(relevant_chunks)}\n\nQuestion: {query}\nAnswer:"
    return prompt

# Test the chatbot with a query
query = "How can I fix the error in the Navbar component?"
codex_prompt = generate_codex_prompt(query, embeddings)
response = ask_codex(codex_prompt)
print(f"Codex response: {response}")

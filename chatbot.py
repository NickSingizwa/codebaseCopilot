import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, render_template, jsonify

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

def ask_codex(prompt, engine="gpt-4"):
    response = client.chat.completions.create(
        # temperature=0.5,
        # max_tokens=150,
        # top_p=1.0,
        # frequency_penalty=0.0,
        # presence_penalty=0.0

        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=engine,
    )
    return response.choices[0].message.content

def find_relevant_chunks(query, embeddings, top_n=3):
    query_embedding = create_embedding(query)
    similarities = [cosine_similarity([query_embedding], [e['embedding']])[0][0] for e in embeddings]
    relevant_chunks = sorted(zip(similarities, embeddings), key=lambda x: x[0], reverse=True)[:top_n]
    return [chunk['text'] for _, chunk in relevant_chunks]

combined_codebase_path = 'combined_codebase.txt'
combined_codebase = read_combined_codebase(combined_codebase_path)
# Check if embeddings.json exists, and create embeddings if it doesn't
embeddings_file_path = 'embeddings.json'
if not os.path.exists(embeddings_file_path):
    embeddings = create_embeddings_for_codebase(combined_codebase)
    # Save embeddings to a file
    with open(embeddings_file_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f)
else:
    # Load embeddings from a file
    with open(embeddings_file_path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)

# generating a Codex prompt with relevant code chunks
def generate_codex_prompt(query, embeddings):
    relevant_chunks = find_relevant_chunks(query, embeddings)
    prompt = f"Context:\n{'\n'.join(relevant_chunks)}\n\nQuestion: {query}\nAnswer:"
    return prompt
    
# chatbot
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['query']
    codex_prompt = generate_codex_prompt(query, embeddings)
    response = ask_codex(codex_prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

# query = "Can you change the text links of the navbar to About me, Projects, and Contact? Return me the whole navbar code."
# codex_prompt = generate_codex_prompt(query, embeddings)
# response = ask_codex(codex_prompt)
# print(f"Codex response: {response}")

import os
import json
import zipfile
from flask import Flask, request, render_template, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from directoryReader import read_files

# Load environment variables
load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def read_combined_codebase(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def create_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def create_embeddings_for_codebase(codebase):
    chunks = chunk_text(codebase)
    embeddings = []
    for chunk in chunks:
        embedding = create_embedding(chunk)
        embeddings.append({'text': chunk, 'embedding': embedding})
    return embeddings

def ask_codex(prompt, engine="gpt-4"):
    response = client.chat.completions.create(
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and file.filename.endswith('.zip'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        # Unzip the file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(app.config['UPLOAD_FOLDER'])
        # Read and process files
        codebase_data = read_files(app.config['UPLOAD_FOLDER'])
        combined_codebase_path = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_codebase.txt')
        with open(combined_codebase_path, 'w', encoding='utf-8') as f:
            f.write(codebase_data)
        # Create embeddings
        embeddings = create_embeddings_for_codebase(codebase_data)
        embeddings_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.json')
        with open(embeddings_file_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f)
        return jsonify({'success': 'File successfully uploaded and processed'})
    else:
        return jsonify({'error': 'Only .zip files are allowed'})

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['query']
    embeddings_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.json')
    with open(embeddings_file_path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)
    codex_prompt = generate_codex_prompt(query, embeddings)
    response = ask_codex(codex_prompt)
    return jsonify({'response': response})

def generate_codex_prompt(query, embeddings):
    relevant_chunks = find_relevant_chunks(query, embeddings)
    prompt = f"Context:\n{'\n'.join(relevant_chunks)}\n\nQuestion: {query}\nAnswer:"
    return prompt

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

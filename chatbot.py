import os
import json
import zipfile
from flask import Flask, request, render_template, jsonify, session, send_from_directory
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
app = Flask(__name__, static_folder='./irembo-codebase-copilot/dist', static_url_path='/')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = os.urandom(24)  # To use Flask sessions

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
def serve():
    # return render_template('index.html')
    return send_from_directory(app.static_folder, 'index.html')

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

        session.clear()
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
        # session['embeddings'] = embeddings
        return jsonify({'success': 'File successfully uploaded and processed'})
    else:
        return jsonify({'error': 'Only .zip files are allowed'})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query')
    # query = request.form['query']
    if 'history' not in session:
        session['history'] = []
    session['history'].append({'role': 'user', 'content': query})
    embeddings_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.json')
    with open(embeddings_file_path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)
    # embeddings = session.get('embeddings')
    if not embeddings:
        return jsonify({'response': 'Embeddings not found. Please upload a codebase first.'})
    codex_prompt = generate_codex_prompt(query, embeddings, session['history'])
    response = ask_codex(codex_prompt)
    session['history'].append({'role': 'assistant', 'content': response})

    # Format the response with code blocks
    # formatted_response = format_response_with_code(response)
    # return jsonify({'response': formatted_response})

    return jsonify({'response': response})

def format_response_with_code(response):
    lines = response.split('\n')
    in_code_block = False
    formatted_lines = []

    for line in lines:
        if line.strip().startswith('```'):
            if in_code_block:
                formatted_lines.append('</code></pre>')
                in_code_block = False
            else:
                formatted_lines.append('<pre><code class="language-javascript">')
                in_code_block = True
        else:
            formatted_lines.append(line)

    if in_code_block:
        formatted_lines.append('</code></pre>')

    return '\n'.join(formatted_lines)

def generate_codex_prompt(query, embeddings, history):
    relevant_chunks = find_relevant_chunks(query, embeddings)
    context = "\n".join([f"{h['role']}: {h['content']}" for h in history])
    prompt = f"Context:\n{context}\n{'\n'.join(relevant_chunks)}\n\nQuestion: {query}\nAnswer:"
    return prompt

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=False)

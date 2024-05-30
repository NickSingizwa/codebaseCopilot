import os
import json
import zipfile
import streamlit as st
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

UPLOAD_FOLDER = 'uploads'
HISTORY_FILE = 'history.json'

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

def format_response_with_code(response):
    lines = response.split('\n')
    in_code_block = False
    formatted_lines = []

    for line in lines:
        if line.strip().startswith('```'):
            if in_code_block:
                formatted_lines.append('```')
                in_code_block = False
            else:
                formatted_lines.append('```')
                in_code_block = True
        else:
            formatted_lines.append(line)

    if in_code_block:
        formatted_lines.append('```')

    return '\n'.join(formatted_lines)

def generate_codex_prompt(query, embeddings, history):
    relevant_chunks = find_relevant_chunks(query, embeddings)
    context = "\n".join([f"{h['role']}: {h['content']}" for h in history])
    # prompt = f"Context:\n{context}\n{'\n'.join(relevant_chunks)}\n\nQuestion: {query}\nAnswer:",
    prompt = (
    f"Context:\n{context}\n" +
    '\n'.join(relevant_chunks) + 
    f"\n\nQuestion: {query}\nAnswer:"
)

    return prompt

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def main():
    st.title('Javascript Codebase Copilot')

    # Initialize session state for the query, response, and file processing status
    if 'query' not in st.session_state:
        st.session_state.query = ''
    if 'response' not in st.session_state:
        st.session_state.response = ''
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'processing_file' not in st.session_state:
        st.session_state.processing_file = False

    file = st.file_uploader("Upload your codebase (only zip file is allowed)", type=['zip'])
    history = load_history()
    embeddings_file_path = os.path.join(UPLOAD_FOLDER, 'embeddings.json')

    if file is not None and not st.session_state.file_uploaded:
        st.session_state.processing_file = True
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())

        # Clear history when a new file is uploaded
        clear_history()
        history = []

        # Unzip the file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_FOLDER)
        
        # Read and process files
        codebase_data = read_files(UPLOAD_FOLDER)
        combined_codebase_path = os.path.join(UPLOAD_FOLDER, 'combined_codebase.txt')
        with open(combined_codebase_path, 'w', encoding='utf-8') as f:
            f.write(codebase_data)
        
        # Create embeddings
        embeddings = create_embeddings_for_codebase(codebase_data)
        with open(embeddings_file_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f)
        
        st.success('File successfully uploaded and processed')
        st.session_state.file_uploaded = True
        st.session_state.processing_file = False

    if st.session_state.processing_file:
        st.info("Processing the uploaded file. Please wait...")
    elif st.session_state.file_uploaded or os.path.exists(embeddings_file_path):
        query = st.text_input('Enter your prompt:', key='query', on_change=lambda: st.session_state.update(ask_disabled=not st.session_state.query.strip()))
        # query = st.text_area('Enter your prompt:', key='query', height=70, on_change=lambda: st.session_state.update(ask_disabled=not st.session_state.query.strip()))
        ask_button = st.button('Ask', key='ask', disabled=not st.session_state.query.strip())

        if ask_button:
            with st.spinner('Processing...'):
                with open(embeddings_file_path, 'r', encoding='utf-8') as f:
                    embeddings = json.load(f)

                prompt = generate_codex_prompt(query, embeddings, history)
                response = ask_codex(prompt)
                formatted_response = format_response_with_code(response)
                st.session_state.response = formatted_response

                history.append({'role': 'user', 'content': query})
                history.append({'role': 'assistant', 'content': response})
                save_history(history)

        if st.session_state.response:
            st.markdown(f"### Response:\n```\n{st.session_state.response}\n```")

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    main()

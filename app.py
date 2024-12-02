import json
import os
import fitz  
import numpy as np
import requests
from flask import Flask, request, jsonify, Response  
import faiss
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)

# Variables to be used in the process
MODEL_URL = os.getenv("MODEL_URL")
EMBEDDING_MODEL_URL = os.getenv("EMBEDDING_MODEL_URL")
TOKEN = os.getenv("TOKEN")
embedding_dim = 768
index = faiss.IndexFlatL2(embedding_dim)
document_chunks = []

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text

# Method to get Embeddings
def get_embeddings(text):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "Metric-AI/armenian-text-embeddings-1",
        "input": [text],
        "encoding_format": "float",
        "truncate_prompt_tokens": 1,
        "add_special_tokens": False,
        "priority": 0
    }

    response = requests.post(EMBEDDING_MODEL_URL, json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        embedding = response_data.get("data", [])[0].get("embedding", [])
        return embedding
    else:
        raise Exception(f"Failed to get embeddings: {response.status_code} - {response.text}")

# Method to add Faiss
def add_to_faiss(embeddings, text_chunk):
    np_embeddings = np.array(embeddings).astype('float32').reshape(1,-1)
    index.add(np_embeddings)
    document_chunks.append(text_chunk)  

# Method to interact with model
def get_custom_model_answer(system_message, user_message, token):
    headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {token}'
    }

    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 5000,
        "temperature": 0.7,
        "stream": True  
    }

    def sanitize_input(input_text):
        return input_text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    system_message = sanitize_input(system_message)
    user_message = sanitize_input(user_message)

    try:
        response = requests.post(MODEL_URL, headers=headers, data=json.dumps(payload), stream=True)
        
        if response.status_code != 200:
            print(f"Error Response: {response.text}")
            return f"Error: {response.status_code}, {response.text}"

        # If you're using streaming, yield chunks of data
        for chunk in response.iter_lines():
            if chunk:
                yield chunk.decode('utf-8')  
        
    except requests.exceptions.RequestException as e:
        print(f"Error while making the API call: {e}")
        return f"Error: {e}"

# Route to upload Pdf and ask
@app.route('/upload_pdf_and_ask', methods=['POST'])
def upload_pdf_and_ask():
    if 'file' not in request.files:
        return jsonify({"error": "Missing file"}), 400
    file = request.files['file']
    
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400
    
    file_path = "uploaded.pdf"
    file.save(file_path)
    text = extract_text_from_pdf(file_path)
    chunks = text.split("\n\n")
    for chunk in chunks:
        embeddings = get_embeddings(chunk)
        add_to_faiss(embeddings, chunk)
    question_embedding = get_embeddings(question)
    D, I = index.search(np.array(question_embedding).astype('float32').reshape(1,-1), 1)  
    best_match = document_chunks[I[0][0]] if I[0][0] < len(document_chunks) else ""
    system_message = "You are a helpful assistant that answers questions to the point based on the provided document."
    user_message = f"Question: {question}\nContext: {best_match}"
    os.remove(file_path)
    return Response(get_custom_model_answer(system_message, user_message, TOKEN), content_type='application/json;charset=utf-8', status=200)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8000)

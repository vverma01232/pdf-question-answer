import os
import json
import pytesseract
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF
import requests
import io
import psycopg
from flask import Flask, request, jsonify, Response
from langchain.text_splitter import CharacterTextSplitter
from pgvector.psycopg import register_vector
from flask_cors import CORS
import numpy as np
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)
load_dotenv()
CORS(app, resources={r"/*": {"origins": "*"}})
MODEL_URL = os.getenv("MODEL_URL")  
TOKEN = os.getenv("TOKEN")
EMBEDDING_MODEL_URL = os.getenv("EMBEDDING_MODEL_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


# Tesseract OCR path configuration
pytesseract.pytesseract.tesseract_cmd = os.path.join(os.getcwd(), 'tesseract')

# PostgreSQL connection setup
def get_db_connection():
    conn = psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    return conn

def create_document_vectors_table():
    conn = get_db_connection()
    cursor = conn.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS document_vectors (
        id SERIAL PRIMARY KEY,
        document_name TEXT NOT NULL,
        chunk TEXT NOT NULL,
        embedding VECTOR(1536)
    );
    """
    cursor.execute(create_table_query)
    conn.commit()

    cursor.close()
    conn.close()

def create_pgvector_extension():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    cursor.close()
    conn.close()


# Helper function to extract text from PDF
def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.point(lambda p: p > 150 and 255)  # Binarization
    img = img.convert('RGB')  # Convert back to RGB for enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Increase contrast to make text stand out more
    return img

# Helper function to extract text from PDF (OCR + Text extraction)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "" 
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_text = page.get_text() 
        
        
        if page_text.strip(): 
            full_text += page_text
        else:  
            pix = page.get_pixmap(dpi=300)  # Use higher DPI for better OCR quality
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            img = preprocess_image(img)
            ocr_text = pytesseract.image_to_string(img, lang='eng')
            full_text += ocr_text

    return full_text



def adjust_embedding_size(embedding, desired_size=1536):
    if not isinstance(embedding, list):
        raise TypeError("Expected 'embedding' to be a list.")
    if len(embedding) > desired_size:
        embedding = embedding[:desired_size]  
    elif len(embedding) < desired_size:
        embedding = np.pad(embedding, (0, desired_size - len(embedding)), 'constant')  
    
    return embedding

# Helper function to get embeddings for a given text
def get_embeddings(text):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
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
        
        if isinstance(embedding, list):
            adjusted_embeddings = adjust_embedding_size(embedding, desired_size=1536)
            return adjusted_embeddings
        else:
            raise TypeError("Embedding response should be a list of floats.")
    else:
        raise Exception(f"Failed to get embeddings: {response.status_code} - {response.text}")
 
    

# Function to add embeddings to PostgreSQL using pgvector
def add_to_pgvector(embeddings, text_chunk, source):
    conn = get_db_connection()
    cursor = conn.cursor()
    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
        embedding_vector = embeddings.tolist()  
       
    else:
        raise TypeError("Expected embeddings[0] to be a list or array, but got: {}".format(type(embeddings[0])))

    insert_query = """
    INSERT INTO document_vectors (document_name, chunk, embedding)
    VALUES (%s, %s, %s);  -- No casting in query
    """

    # Execute the insert query
    cursor.execute(insert_query, (source, text_chunk, embedding_vector))
    conn.commit()

    cursor.close()
    conn.close()

# Function to search PGVector
def search_pgvector(query_embedding, top_k=5):
    if isinstance(query_embedding, np.ndarray):
        query_vector = query_embedding.tolist() 
    else:
        raise TypeError(f"Expected query_embedding to be a numpy array, but got: {type(query_embedding)}")
    conn = get_db_connection()
    cursor = conn.cursor()

    
    search_query = """
    SELECT document_name, chunk, embedding
    FROM document_vectors
    ORDER BY embedding <=> %s::vector(1536)  
    LIMIT %s;
    """

    # Execute the query with the proper casting of query_vector
    cursor.execute(search_query, (query_vector, top_k))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results

# Function to split text into smaller chunks using Langchain's splitter
def split_text_into_chunks(text, max_chunk_size=512):
    text_splitter = CharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Event stream generator
def event_generator(system_message, user_message, token, streaming):
    """This generator handles the event stream and yields data."""
    try:
        for chunk in get_custom_model_answer(system_message, user_message, token, streaming):
            yield f"{chunk}\n\n"
    except Exception as e:
        yield f"data: Error - {str(e)}\n\n"

def delete_from_pgvector(source):
    conn = get_db_connection()
    cursor = conn.cursor()
 
    delete_query = """
    DELETE FROM document_vectors
    WHERE document_name = %s;
    """
    cursor.execute(delete_query, (source,))
    conn.commit()
 
    cursor.close()

# Function to get the response from the model API
def get_custom_model_answer(system_message, user_message, token, streaming):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    def sanitize_input(input_text):
        return input_text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

    system_message = sanitize_input(system_message)
    user_message = sanitize_input(user_message)

    try:
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 5000,
            "temperature": 0.7,
            "stream": streaming
        }
        response = requests.post(MODEL_URL, headers=headers, data=json.dumps(payload), stream=True)
        if response.status_code != 200:
            print(f"Error Response: {response.text}")
            return f"Error: {response.status_code}, {response.text}"

        # Yield chunks if response is streamed
        for chunk in response.iter_lines():
            if chunk:
                yield chunk.decode('utf-8')

    except requests.exceptions.RequestException as e:
        print(f"Error while making the API call: {e}")
        return f"Error: {e}"

# Route to upload multiple PDFs and ask questions
@app.route('/upload_pdf_and_ask', methods=['POST'])
def upload_pdf_and_ask():
    if 'file' not in request.files:
        return jsonify({"error": "Missing files"}), 400

    files = request.files.getlist('file')
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400
    streaming = request.form.get("stream") == 'true'

    for file in files:
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        text = extract_text_from_pdf(file_path)

        chunks = split_text_into_chunks(text, max_chunk_size=512)

        for chunk in chunks:
            embeddings = get_embeddings(chunk)
            add_to_pgvector(embeddings, chunk, file.filename)

    
    question_embedding = get_embeddings(question)
    search_results = search_pgvector(question_embedding)

    for file in files:
        file_path = os.path.join('uploads', file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    # Prepare response context
    if search_results:
        best_match = search_results[0][1]
        matched_document = search_results[0][0]
    else:
        best_match = "No relevant match found."
        matched_document = "Unknown"

    max_tokens = 4096
    context = best_match[:max_tokens]

    # Prepare system and user messages for the model
    system_message = "You are a helpful assistant that answers questions to the point based on the provided documents.. Please limit your answer to 200 words."
    user_message = f"Question: {question}\nContext: {context}"

    for file in files:
        delete_from_pgvector(file.filename)

    if streaming:
        return Response(event_generator(system_message, user_message, TOKEN, streaming=True),
                        content_type='text/event-stream;charset=utf-8', status=200 )
    else:
        return Response(event_generator(system_message, user_message, TOKEN, streaming=False),
                        content_type='application/json', status=200 )

# Run the Flask app
if __name__ == '__main__':
    conn = get_db_connection()
    conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    create_pgvector_extension()
    create_document_vectors_table()
    register_vector(conn)
    app.run(debug=False, host="0.0.0.0", port=8000)

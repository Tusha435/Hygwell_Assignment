from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict
import requests
from bs4 import BeautifulSoup
import uuid
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

# Initialize FastAPI app
app = FastAPI()

# Initialize the model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory storage for content
content_store: Dict[str, str] = {}

# Define request body schema for URL processing
class URLRequest(BaseModel):
    url: str

# Define request body schema for Chat API
class ChatRequest(BaseModel):
    chat_id: str
    question: str

# Endpoint to process web URL
@app.post("/process_url")
async def process_url(request: URLRequest):
    try:
        # Scrape the content from the URL
        response = requests.get(request.url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to retrieve content from URL")

        # Use BeautifulSoup to clean and parse the content
        soup = BeautifulSoup(response.text, "html.parser")
        cleaned_content = soup.get_text(separator=' ', strip=True)

        # Generate a unique chat ID
        chat_id = str(uuid.uuid4())

        # Store the cleaned content with the chat_id
        content_store[chat_id] = cleaned_content

        return {
            "chat_id": chat_id,
            "message": "URL content processed and stored successfully."
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the URL")

# Endpoint to process PDF documents
@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...)):
    try:
        # Read the PDF content
        pdf_reader = PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Clean the extracted text
        cleaned_text = ' '.join(text.split())

        # Generate a unique chat ID
        chat_id = str(uuid.uuid4())

        # Store the cleaned text with the chat_id
        content_store[chat_id] = cleaned_text
        print(content_store)
        return {
            "chat_id": chat_id,
            "message": "PDF content processed and stored successfully."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the PDF document")


@app.post("/chat")
async def chat(request: ChatRequest):
    # Retrieve the stored content based on chat_id
    content = content_store.get(request.chat_id)

    if not content:
        raise HTTPException(status_code=404, detail="Chat ID not found")

    try:
        # Generate embeddings for both the stored content and the user's question
        content_embedding = model.encode(content, convert_to_tensor=True)
        question_embedding = model.encode(request.question, convert_to_tensor=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

    try:
        # Calculate the cosine similarity between the content and the question
        similarity_score = util.cos_sim(content_embedding, question_embedding)

        # For simplicity, we assume the entire document is relevant.
        # You can modify this logic to return specific parts if needed.
        response_text = "The main idea of the document is..."

        return {
            "response": response_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the chat request: {str(e)}")
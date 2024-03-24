from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import os
from test import *  # Assuming test module contains necessary functions
from PyPDF2 import PdfReader

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define a class for chat messages
class ChatMessage:
    def __init__(self, content: str, sender: str):
        self.content = content
        self.sender = sender


# Define a class for chat history
class ChatHistory:
    def __init__(self):
        self.messages = []

    def add_message(self, content: str, sender: str):
        self.messages.append(ChatMessage(content=content, sender=sender))

    def as_dict(self):
        return {"chat_history": [msg.__dict__ for msg in self.messages]}


@app.post("/chatpdf/")
async def chatpdf(file: UploadFile = File(...), user_question: str = None):
    try:
        # Check if file exists
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Check if the uploaded file is a PDF
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(file.file.read())

        # Process the PDF file
        raw_text = get_pdf_text("temp.pdf")

        # Process text chunks and conversation chain
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore)

        # Handle user input
        chat_history = ChatHistory()
        chat_history.add_message(content=user_question, sender="user")
        chat_history.add_message(
            content=handle_user_input(user_question, conversation_chain), sender="AI"
        )

        return chat_history.as_dict()

    finally:
        # Delete the temporary file
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")


# Define the chaturl endpoint
@app.post("/chaturl/")
async def chaturl(user_question: str = None, url: str = None):
    try:
        if user_question is None or url is None:
            raise HTTPException(
                status_code=400, detail="Both user_question and url are required."
            )

        # Get text from the URL
        raw_text = get_text_from_url(url)

        # Process text chunks and conversation chain
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore)

        # Handle user input
        chat_history = ChatHistory()
        chat_history.add_message(content=user_question, sender="user")
        chat_history.add_message(
            content=handle_user_input(user_question, conversation_chain), sender="AI"
        )

        return chat_history.as_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Hello, world!"}

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import requests
from dotenv import load_dotenv


def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_from_url(url):
    response = requests.get(url)
    if response.status_code:
        text = response.text
        return text
    else:
        return None



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory, chain_type="stuff"
    )
    return conversation_chain


def handle_user_input(user_question, conversation_chain):
    response = conversation_chain({"question": user_question})
    chat_history = response["chat_history"]
    return chat_history


# def main():
#     load_dotenv()

#     # Get user input
#     file_type = input(
#         "Enter 'PDF' if you want to provide a PDF file or 'URL' if you want to provide a URL: "
#     )

#     # Process based on user choice
#     if file_type.upper() == "PDF":
#         pdf_file_path = input("Enter the path to the PDF file: ")
#         if os.path.isfile(pdf_file_path):
#             raw_text = get_pdf_text(pdf_file_path)
#         else:
#             print("Invalid file path or file does not exist.")
#             return
#     elif file_type.upper() == "URL":
#         url = input("Enter the URL: ")
#         raw_text = get_text_from_url(url)
#         if raw_text is None:
#             print("Failed to fetch text from URL. Please make sure the URL is correct.")
#             return
#     else:
#         print("Invalid input. Please enter 'PDF' or 'URL'.")
#         return

#     # Process text chunks and conversation chain
#     text_chunks = get_text_chunks(raw_text)
#     vectorstore = get_vectorstore(text_chunks)
#     conversation_chain = get_conversation_chain(vectorstore)

#     user_question = input("Enter your question: ")


#     # Handle user input
#     chat_history = handle_user_input(user_question, conversation_chain)
#     for i, message in enumerate(chat_history):
#         if i % 2 == 0:
#             print("User:", message.content)
#         else:
#             print("Bot:", message.content)


# if __name__ == "__main__":
#     main()

def chatpdf(falipath, user_question):

    # Process based on user choice
    if os.path.isfile(falipath):
        raw_text = get_pdf_text(falipath)

    # Process text chunks and conversation chain
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)

    # Handle user input
    chat_history = handle_user_input(user_question, conversation_chain)
    return chat_history


def chaturl(url, user_question):

    raw_text = get_text_from_url(url)

    # Process text chunks and conversation chain
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)

    # Handle user input
    chat_history = handle_user_input(user_question, conversation_chain)
    return chat_history

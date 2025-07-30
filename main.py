from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import os

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load Groq LLM
GROQ_APIKEY = os.getenv("GROQ_APIKEY")
llm = ChatGroq(
    model="mixtral-8x7b",
    temperature=0.4,
    reasoning_format="parsed",
    max_retries=2,
    api_key=GROQ_APIKEY
)

# Smaller embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

@app.post("/upload")
async def upload_file(file: UploadFile, user_id: str = Form(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(content={"error": "Only PDF files are supported."}, status_code=400)

    contents = await file.read()
    pdf_path = f"temp_{user_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(contents)

    # Load and split PDF
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    # Save vectors
    vectorstore_dir = f"chroma_db/{user_id}"
    os.makedirs(vectorstore_dir, exist_ok=True)
    db = Chroma.from_documents(
        docs,
        embedding=embedding_model,
        persist_directory=vectorstore_dir
    )
    db.persist()

    os.remove(pdf_path)
    return JSONResponse(content={"status": "success"})

@app.post("/ask")
async def ask(question: str = Form(...), user_id: str = Form(...)):
    vectorstore_dir = f"chroma_db/{user_id}"
    if not os.path.exists(vectorstore_dir):
        return JSONResponse(content={"answer": "No document uploaded yet."})

    db = Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=embedding_model
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )
    result = qa.run(question)
    return JSONResponse(content={"answer": result})

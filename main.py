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

# Load environment variables
load_dotenv()
GROQ_APIKEY = os.getenv('GROQ_APIKEY')

# Setup FastAPI app
app = FastAPI()
# Add to your FastAPI app startup
@app.on_event("startup")
async def startup_event():
    # Warm up the model
    dummy_text = "warmup"
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2").embed_query(dummy_text)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Set up LLM
llm = ChatGroq(
    model="mixtral-8x7b",
    temperature=0.4,
    reasoning_format="parsed",
    max_retries=2,
    api_key=GROQ_APIKEY
)

# Ensure uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Upload PDF and process
@app.post("/upload")
async def upload_file(file: UploadFile, user_id: str = Form(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(content={"error": "Only PDF files are supported."}, status_code=400)

    # Save to uploads/
    pdf_path = os.path.join(UPLOAD_DIR, f"{user_id}.pdf")
    try:
        contents = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(contents)

        # Load and split PDF
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

        # Save vector store in chroma_db/user_id
        vectorstore_dir = f"chroma_db/{user_id}"
        os.makedirs(vectorstore_dir, exist_ok=True)

        db = Chroma.from_documents(
            docs,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            persist_directory=vectorstore_dir
        )
        db.persist()
        os.remove(pdf_path)

        return JSONResponse(content={"status": "success"})

    except Exception as e:
        return JSONResponse(content={"error": f"Failed to process file: {str(e)}"}, status_code=500)

# Ask question endpoint
@app.post("/ask")
async def ask(question: str = Form(...), user_id: str = Form(...)):
    vectorstore_dir = f"chroma_db/{user_id}"
    if not os.path.exists(vectorstore_dir):
        return JSONResponse(content={"answer": "No document uploaded yet."})

    try:
        db = Chroma(
            persist_directory=vectorstore_dir,
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever()
        )
        result = qa.run(question)
        return JSONResponse(content={"answer": result})

    except Exception as e:
        return JSONResponse(content={"error": f"Failed to process question: {str(e)}"}, status_code=500)

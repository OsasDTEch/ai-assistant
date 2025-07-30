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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env
load_dotenv()
GROQ_APIKEY = os.getenv('GROQ_APIKEY')
if not GROQ_APIKEY:
    logger.error("❌ GROQ_APIKEY not set in environment.")
    raise ValueError("Missing GROQ_APIKEY in .env")

# Initialize FastAPI
app = FastAPI()

# Warm-up embedding model
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("⚡ Warming up HuggingFace embeddings...")
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2").embed_query("warmup")
        logger.info("✅ Embeddings ready.")
    except Exception as e:
        logger.error(f"❌ Failed to warm up embeddings: {str(e)}")

# CORS
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

# Directories
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(content={"error": "Only PDF files are supported."}, status_code=400)

    pdf_path = os.path.join(UPLOAD_DIR, f"{user_id}.pdf")
    try:
        contents = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(contents)

        # Load and split
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

        vectorstore_dir = f"chroma_db/{user_id}"
        os.makedirs(vectorstore_dir, exist_ok=True)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=vectorstore_dir)
        db.persist()

        os.remove(pdf_path)
        return JSONResponse(content={"status": "success"})

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return JSONResponse(content={"error": f"Failed to process file: {str(e)}"}, status_code=500)

@app.post("/ask")
async def ask(question: str = Form(...), user_id: str = Form(...)):
    vectorstore_dir = f"chroma_db/{user_id}"
    if not os.path.exists(vectorstore_dir):
        return JSONResponse(content={"answer": "No document uploaded yet."})

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
        result = qa.run(question)
        return JSONResponse(content={"answer": result})

    except Exception as e:
        logger.error(f"Question handling failed: {str(e)}")
        return JSONResponse(content={"error": f"Failed to process question: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

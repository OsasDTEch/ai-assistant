from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

import os



def process_and_store(file_path:str , user_id:str):
    loader= PyPDFLoader(file_path)
    documents =loader.load()
    splitter= RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks= splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")
    vectorstore= Chroma(
        collection_name=user_id,
        persist_directory='./chroma_db',
        embedding_function= embeddings
    )

    vectorstore.add_documents(chunks)
    vectorstore.persist()
    return True

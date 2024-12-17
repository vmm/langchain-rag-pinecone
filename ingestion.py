import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document

def validate_environment():
    """Validate required environment variables are set."""
    required_vars = ["OPENAI_API_KEY", "PINECONE_INDEX_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

def process_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 0
) -> List[Document]:
    """Split documents into chunks and process them."""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")
    return texts

def ingest_documents(texts: List[Document], index_name: str):
    """Ingest documents into Pinecone vector store."""
    try:
        embeddings = OpenAIEmbeddings()
        return PineconeVectorStore.from_documents(
            texts,
            embeddings,
            index_name=index_name
        )
    except ConnectionError as e:
        raise ConnectionError(f"Failed to connect to Pinecone: {str(e)}") from e
    except ValueError as e:
        raise ValueError(f"Error processing documents: {str(e)}") from e

def ingest_text(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 0):
    """Ingest a text file into the vector store."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    print(f"Loading text file: {file_path}")
    raw_documents = TextLoader(file_path).load()
    return process_and_ingest(raw_documents, chunk_size, chunk_overlap)

def ingest_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 0):
    """Ingest a PDF file into the vector store."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    print(f"Loading PDF file: {file_path}")
    loader = PyPDFLoader(file_path=file_path)
    raw_documents = loader.load()
    return process_and_ingest(raw_documents, chunk_size, chunk_overlap)

def process_and_ingest(raw_documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 0):
    """Process and ingest documents into the vector store."""
    load_dotenv()
    validate_environment()

    texts = process_documents(raw_documents, chunk_size, chunk_overlap)
    print("Embedding and ingesting documents...")
    return ingest_documents(texts, os.environ["PINECONE_INDEX_NAME"])

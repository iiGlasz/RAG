from dotenv import load_dotenv
from llama_cloud.client import LlamaCloud
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_parse import LlamaParse
from llama_index.core import Settings
from pinecone import Pinecone
from pinecone import ServerlessSpec
import os

# removes asyncio error on windows
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables from .env file
load_dotenv()
pineconeKey = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
llamaClient = LlamaCloud()

llm = GoogleGenAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash")
embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")

# prompt the user for their PDF file
pdfFile = input("Enter the path to your PDF file: ")

# initialize the parser for PDF files
parser = LlamaParse(
    result_type="markdown",
    verbose=True
)

# settings for chunking the documents
Settings.chunk_size = 256
Settings.chunk_overlap = 60
Settings.top_k = 5

# use the parser for PDF files and convert them into LlamaIndex objects
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    input_files=[pdfFile], # Path to your local PDF
    file_extractor=file_extractor
).load_data()

# ensure pinecone index exists
pineconeIndexName = "firstrag-py1"
if not pineconeKey.has_index(pineconeIndexName):
    pineconeKey.create_index(
        name=pineconeIndexName,
        vector_type="dense",
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
            "environment": "development"
        }
    )

# initialize pinecone vector store and storage context
pinecone_index = pineconeKey.Index(pineconeIndexName)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create the vector store index from the documents, storage context, and embedding model
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    embed_model=embed_model
)

# prompt the user for their prompt to gemini
userPrompt = input("Enter your prompt for Gemini: ")

# get and print the response from gemini
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query(userPrompt)
print(response)
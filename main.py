from dotenv import load_dotenv
from llama_cloud import LlamaCloud, AsyncLlamaCloud
from pinecone import Pinecone
from google import genai
import os

load_dotenv()
# Load environment variables from .env file
pineconeKey = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
llamaClient = LlamaCloud(api_key=os.getenv("LLAMA_API_KEY"))
geminiClient = genai.Client()

# prompt the user for their prompt to gemini
userPrompt = input("Enter your prompt for Gemini: ")

# ensure pinecone index exists
pineconeIndex = "firstrag-py"
if not pineconeKey.has_index(pineconeIndex):
    pineconeKey.create_index_for_model(
        name=pineconeIndex,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2", # not sure if correct model
            "field_map":{"text": "chunk_text"}
        }
    )

# use gemini 3 to generate a response based on user prompt
response = geminiClient.models.generate_content(
    model="gemini-3-flash-preview", contents=userPrompt
)
print(response.text)
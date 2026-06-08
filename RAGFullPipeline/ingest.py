import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_EMBEDDING"))

class GeminiEmbeddingWrapper(Embeddings):
    def __init__(self, client, model="gemini-embedding-2"):
        self.client = client
        self.model = model

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=f"task: clustering | {text}",
                config=types.EmbedContentConfig(
                    output_dimensionality=768
                )
            )

            embeddings.append(response.embeddings[0].values)

        return embeddings
    
    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=self.model,
            contents=f"task: search_query | {text}",
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values

gemini_embeddings = GeminiEmbeddingWrapper(client)

print("Loading and splitting PDF...")
loader = PyPDFLoader("docs/rbi_notification.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=330)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

print("Generating embeddings and saving to ChromaDB (this may take a moment)...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=gemini_embeddings,
    persist_directory="./chroma_db"
)

print("Indexing Complete!")
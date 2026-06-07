import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

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
        """Embed search queries"""
        response = self.client.models.embed_content(
            model=self.model,
            contents=f"task: search_query | {text}",
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values

# Initialize our custom wrapper
gemini_embeddings = GeminiEmbeddingWrapper(client)

print("Loading and splitting PDF...")
loader = PyPDFLoader("documents/Docker CheatSheet ApnaCollege.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

print("Generating embeddings and saving to ChromaDB (this may take a moment)...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=gemini_embeddings,
    persist_directory="./chroma_db"
)

print("Indexing Complete!")

# retriever = vectorstore.as_retriever(
#     search_kwargs={"k": 3}
# )

# print("\nRAG Chatbot Ready!")
# print("Type 'exit' to quit.\n")

# # ---------------------------
# # Chat Loop
# # ---------------------------

# while True:

#     question = input("You: ")

#     if question.lower() in ["exit", "quit"]:
#         break

#     docs = retriever.invoke(question)

#     context = "\n\n".join(
#         [doc.page_content for doc in docs]
#     )

#     prompt = f"""
# You are a helpful assistant.

# Answer ONLY using the provided context.

# If the answer is not present in the context,
# say:
# "I could not find that information in the document."

# Context:
# {context}

# Question:
# {question}
# """

#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=prompt
#     )

#     print("\nBot:", response.text)
#     print("\n" + "=" * 60 + "\n")
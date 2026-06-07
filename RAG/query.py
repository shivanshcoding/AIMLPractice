import os

from google import genai
from dotenv import load_dotenv
from ingest import vectorstore
load_dotenv()

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("\nRAG Chatbot Ready!")
print("Type 'exit' to quit.\n")

# ---------------------------
# Chat Loop
# ---------------------------

while True:

    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful assistant.

Answer using the provided context.

If the answer is not present in the context,

Then make a best effort to answer the question based on your general knowledge.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nBot:", response.text)
    print("\n" + "=" * 60 + "\n")
from fastapi import FastAPI
import chromadb
import ollama
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

MODEL_NAME = os.getenv("MODEL_NAME", "tinyllama")
logging.info(f"Using model: {MODEL_NAME}")

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")
# ollama_client = ollama.Client(host="http://host.docker.internal:11434")

@app.post("/query")
def query(q: str):
    logging.info(f"Received query: {q}")
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    # answer = ollama_client.generate(
    answer = ollama.generate(
        model=MODEL_NAME,
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
    )

    return {"answer": answer["response"]}


@app.post("/add")
def add_knowledge(text: str):
    """Add new content to the knowledge base dynamically."""
    logging.info(f"Received new text: {text}")
    try:
        # Generate a unique ID for this document
        import uuid
        doc_id = str(uuid.uuid4())
        
        # Add the text to Chroma collection
        collection.add(documents=[text], ids=[doc_id])
        
        return {
            "status": "success",
            "message": "Content added to knowledge base",
            "id": doc_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }



@app.get("/health")
def health():
    """Liveness probe endpoint for Kubernetes or other orchestrators."""
    return {"status": "ok"}



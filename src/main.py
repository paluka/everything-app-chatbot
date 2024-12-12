from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

from config import DELETE_VECTOR_STORE, EVERYTHING_APP_FRONTEND_PATH, ALREADY_POPULATED_VECTOR_STORE
from vector_store import VectorStore
from chatbot import Chatbot
from utils import load_files

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ALREADY_POPULATED_VECTOR_STORE:
        # Load files from the `data` directory
        documents = load_files(EVERYTHING_APP_FRONTEND_PATH)

        # if DELETE_VECTOR_STORE:
        #     # Check if the documents already exist in the vector store
        #     existing_docs = set()  # Track the documents already added
        #     for doc in documents:
        #         # Check if the document is already in the vector store
        #         # You might want to search by the content or a unique identifier
        #         if not vector_store.is_document_in_store(doc):  # Method to be implemented
        #             existing_docs.add(doc)
            
        #     # Only add new documents to the vector store
        #     if existing_docs:
        #         vector_store.load_documents(existing_docs)  # Load only new documents
        #         print(f"Loaded {len(existing_docs)} new documents into the vector store.")
        #     else:
        #         print("No new documents to load into the vector store.")
        # else:
        #     vector_store.load_documents(documents)

        vector_store.load_documents(documents)

    yield

app = FastAPI(lifespan=lifespan)

# Initialize VectorStore and Chatbot
vector_store = VectorStore()
chatbot = Chatbot(vector_store)

@app.post("/query/")
async def query_chatbot(query: str):
    response = chatbot.get_response(query)
    if response:
        return {"query": query, "response": response}
    else:
        raise HTTPException(status_code=500, detail="Could not generate response.")
    
@app.get("/query/")
async def query_chatbot(query: str):
    response = chatbot.get_response(query)
    if response:
        return JSONResponse(content={"query": query, "response": response})
    else:
        raise HTTPException(status_code=500, detail="Could not generate response.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
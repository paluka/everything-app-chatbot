from chatbot.haystack_chatbot import HayStackChatbot
from chatbot.langchain_chatbot import LangChainChatbot
from utils.load_files import load_files
from chatbot.chatbot import Chatbot
from vector_store.pinecone_vector_store import PineconeVectorStore
from vector_store.faiss_vector_store import FaissVectorStore
from config import USE_LANGCHAIN, USE_HAYSTACK, EVERYTHING_APP_FRONTEND_PATH, ALREADY_POPULATED_PINECONE, USE_PINECONE_VECTORSTORE
import uvicorn
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
# import pdb

# pdb.set_trace()

print(
    f"\n\nUSE_PINECONE_VECTORSTORE: {USE_PINECONE_VECTORSTORE}, USE_LANGCHAIN: {USE_LANGCHAIN}, USE_HAYSTACK: {USE_HAYSTACK}\n\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting the app in lifespan")

    if USE_PINECONE_VECTORSTORE:
        if not ALREADY_POPULATED_PINECONE:
            documents = load_files(EVERYTHING_APP_FRONTEND_PATH)
            vector_store.load_documents(documents)
    else:
        if len(vector_store.getMetadata()) == 0:
            documents = load_files(EVERYTHING_APP_FRONTEND_PATH)
            vector_store.load_documents(documents)

    print("About to yield")
    yield

app = FastAPI(lifespan=lifespan)

# Initialize VectorStore and Chatbot
vector_store = None

if USE_PINECONE_VECTORSTORE:
    vector_store = PineconeVectorStore()
else:
    vector_store = FaissVectorStore()

chatbot = None

if USE_LANGCHAIN:
    chatbot = LangChainChatbot(vector_store)

elif USE_HAYSTACK:
    chatbot = HayStackChatbot(vector_store)

else:
    chatbot = Chatbot(vector_store)


@app.post("/query/")
async def query_chatbot(query: str):
    response = chatbot.get_response(query)
    if response:
        return {"query": query, "response": response}
    else:
        raise HTTPException(
            status_code=500, detail="Could not generate response.")


@app.get("/query/")
async def query_chatbot(query: str):
    print(f"Received query {query}")

    response = chatbot.get_response(query)
    if response:
        return JSONResponse(content={"query": query, "response": response})
    else:
        raise HTTPException(
            status_code=500, detail="Could not generate response.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

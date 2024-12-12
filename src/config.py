import os

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = 'chatbot-code-index'
EVERYTHING_APP_FRONTEND_PATH = '../everything-app-frontend'
LLM_NAME = "meta-llama/Llama-3.2-1B"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DELETE_VECTOR_STORE = False
ALREADY_POPULATED_VECTOR_STORE=True
FETCH_FILES_FOR_QUERY=True

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from config import DELETE_VECTOR_STORE, EMBEDDING_MODEL_NAME, INDEX_NAME, PINECONE_API_KEY


class PineconeVectorStore:
    def __init__(self):
        # Initialize Pinecone and the embedding model
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        dimension = 384
        metric = "cosine"  # Common choice for text embeddings
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )

        if DELETE_VECTOR_STORE:
            if any(index['name'] == INDEX_NAME for index in pinecone_client.list_indexes()):
                pinecone_client.delete_index(INDEX_NAME)
                print(f"Index '{INDEX_NAME}' deleted.")
            else:
                print(f"Index '{INDEX_NAME}' does not exist.")

        # Create or connect to Pinecone index
        if not any(index['name'] == INDEX_NAME for index in pinecone_client.list_indexes()):
            pinecone_client.create_index(
                INDEX_NAME, dimension=dimension, metric=metric, spec=spec)

        self.index = pinecone_client.Index(INDEX_NAME)

    def load_documents(self, documents):
        """Convert documents to embeddings and load them into the vector store."""
        embeddings = self.model.encode(documents, show_progress_bar=True)
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            self.index.upsert([(str(i), embedding, {"text": doc})])
        print(f"Loaded {len(documents)} documents into Pinecone.")

    def query(self, query, top_k=5):
        """Search the vector store for relevant documents."""

        query_embedding = self.model.encode(
            query)  # Convert the query into an embedding

        results = self.index.query(
            vector=query_embedding.tolist(),  # The query vector
            top_k=top_k,                      # Number of nearest neighbors to return
            # Include metadata (in this case, the "text")
            include_metadata=True
        )
        return [result["metadata"]["text"] for result in results["matches"]]

    def is_document_in_store(self, doc):
        # Convert the document to an embedding
        # Assume your model is SentenceTransformer
        embedding = self.model.encode([doc])[0]
        # Query the vector store (e.g., Pinecone) to check if the embedding already exists
        results = self.index.query(
            vector=embedding.tolist(), top_k=1, include_metadata=True)

        # If the result contains the document text, it means the document is already in the store
        if results['matches']:
            return True

        return False

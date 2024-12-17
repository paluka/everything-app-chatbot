import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from config import DELETE_VECTOR_STORE, EMBEDDING_MODEL_NAME, INDEX_NAME, PINECONE_API_KEY


class FaissVectorStore:
    def __init__(self):
        # Initialize the embedding model
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.dimension = 384
        self.index = None
        self.metadata = []

        if DELETE_VECTOR_STORE:
            """Delete the FAISS index file and metadata."""

            if os.path.exists(f"{INDEX_NAME}.index"):
                os.remove(f"{INDEX_NAME}.index")
                print(f"Index '{INDEX_NAME}.index' deleted.")

            if os.path.exists(f"{INDEX_NAME}.metadata"):
                os.remove(f"{INDEX_NAME}.metadata")
                print(f"Metadata '{INDEX_NAME}.metadata' deleted.")

        try:
            self.index = faiss.read_index(f"{INDEX_NAME}.index")
            print(f"Loaded existing index '{INDEX_NAME}.index'.")

            with open(f"{INDEX_NAME}.metadata", "rb") as f:
                self.metadata = pickle.load(f)
                print(f"Loaded metadata from '{INDEX_NAME}.metadata'.")

        except (FileNotFoundError, RuntimeError):
            self.index = faiss.IndexFlatIP(
                self.dimension)  # Inner product index
            print(f"Created new FAISS index with dimension {self.dimension}.")

    def getMetadata(self):
        return self.metadata

    def load_documents(self, documents):
        """Convert documents to embeddings and load them into the vector store."""
        embeddings = self.model.encode(
            documents, show_progress_bar=True).astype(np.float32)

        # Add embeddings to the FAISS index
        self.index.add(embeddings)

        # Add document metadata (e.g., text)
        self.metadata.extend(documents)
        print(f"Loaded {len(documents)} documents into FAISS.")

        # Save the index to disk
        faiss.write_index(self.index, f"{INDEX_NAME}.index")

        with open(f"{INDEX_NAME}.metadata", "wb") as f:
            pickle.dump(self.metadata, f)

    def query(self, query, top_k=5):
        """Search the vector store for relevant documents."""
        query_embedding = self.model.encode([query]).astype(np.float32)

        # Search the FAISS index for the top-k nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve the corresponding documents
        results = []
        for idx in indices[0]:
            if idx != -1:  # Valid index
                results.append(self.metadata[idx])

        return results

    def is_document_in_store(self, doc):
        """Check if a document is in the vector store."""
        # Convert the document to an embedding
        embedding = self.model.encode([doc])[0].astype(np.float32)

        # Query the index to check for similarity
        if self.index.ntotal > 0:
            _, I = self.index.search(np.array([embedding]), 1)  # Top-1 result
            if I[0][0] != -1:  # If a match exists
                return True

        return False

from transformers import AutoModelForCausalLM, AutoTokenizer
from config import LLM_NAME, FETCH_FILES_FOR_QUERY

class Chatbot:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
        # Use transformers directly if HuggingFace import fails
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(LLM_NAME)

    def get_response(self, query):
        context = self.vector_store.query(query)
        if not context:
            return "I couldn't find any relevant information in the codebase."
        
        # prompt = f"Context: The following files relate to the user query: {context}\n\nUser Query: {query}\n\nResponse:"
        prompt = ""

        if not FETCH_FILES_FOR_QUERY:
            file_paths = "\n".join(context)
            prompt = f"""
                Context: The following files may contain relevant information related to your query. These are the locations of the files that might contain information related to the query. Please infer its meaning from the contents of these files:
                {file_paths}

                User Query: {query}

                Response:
                """
        else:
            # Read the content of the files specified in the context
            documents = []
            for file_path in context:
                with open(file_path.replace("File: ", ""), "r", encoding="utf-8") as file:
                    documents.append(f"{file_path} {file.read()}")

            # Join the document content into a single string
            # context_content = "\n".join(documents)

            # Construct the prompt with the file contents
            prompt = f"""
            Context: The following content from relevant files may help answer the query:

            {documents[0]}

            User Query: {query}

            Response:
            """

        print(f"\n\nContext: {context}\nPrompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

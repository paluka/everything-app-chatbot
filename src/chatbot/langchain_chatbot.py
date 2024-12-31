from config import LLM_NAME, FETCH_FILES_FOR_QUERY
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage


class LangChainChatbot:
    def __init__(self, vector_store):
        self.vector_store = vector_store

        # self.tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
        # self.model = AutoModelForCausalLM.from_pretrained(LLM_NAME)

        # self.tokenizer.chat_template = """
        #     <system>{system_message}</system>
        #     <user>{user_message}</user>
        #     <assistant>
        # """
        # chat = [
        #     {"role": "user", "content": "Hello, how are you?"},
        #     {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        #     {"role": "user", "content": "I'd like to show off how chat templating works!"},
        # ]
        # self.tokenizer.apply_chat_template(chat, tokenize=False)

        # pipe = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     max_length=10240,
        #     max_time=300,
        #     num_return_sequences=1,  # Generate only one sequence
        #     no_repeat_ngram_size=2,  # Prevent repetition
        #     top_p=0.95,  # Top-p sampling
        #     top_k=60,  # Top-k sampling
        #     temperature=0.1  # Temperature for 0.1 for most deterministic and 0.99 for most creativity

        # )

        # self.chat_model = ChatHuggingFace(
        #     llm=HuggingFacePipeline(pipeline=pipe))

        llm = HuggingFacePipeline.from_model_id(
            model_id="meta-llama/Llama-2-7b-chat-hf",
            task="text-generation",
            pipeline_kwargs=dict(
                max_length=10240,
                max_time=300,
                num_return_sequences=1,  # Generate only one sequence
                no_repeat_ngram_size=2,  # Prevent repetition
                top_p=0.95,  # Top-p sampling
                top_k=60,  # Top-k sampling
                temperature=0.1  # Temperature for 0.1 for most deterministic and 0.99 for most creativity
            ),
        )

        self.chat_model = ChatHuggingFace(llm=llm)

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
                    # documents.append(f"{file_path} {file.read()}")
                    documents.append(f"{file.read()}")

            # Join the document content into a single string
            # context_content = "\n".join(documents)

            # Construct the prompt with the file contents
            # prompt = f"""
            # Context: The following content from relevant files may help answer the query:

            # {documents[0]}

            # User Query: {query}

            # Response:
            # """

            max_length_file_contents = 1500000

            prompt = f"""
            #####
            Context: The following query discusses the file at path {context[0]}. Below are the contents of this file:\n\n{documents[0][:max_length_file_contents]}
            #####
            #####
            Query: {query}
            #####
            #####
            Response:
            """

        # print(f"\n\nContext: {context}\nPrompt: {prompt}")

        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs,
        #                               # self.tokenizer.model_max_length # 512 self.model.config.max_position_embeddings
        #                               max_length=10240,
        #                               max_time=300,
        #                               num_return_sequences=1,  # Generate only one sequence
        #                               no_repeat_ngram_size=2,  # Prevent repetition
        #                               top_p=0.95,  # Top-p sampling
        #                               top_k=60,  # Top-k sampling
        #                               temperature=0.1  # Temperature for 0.1 for most deterministic and 0.99 for most creativity
        #                               )

        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # response = self.conversation_chain.run(input=query, context=context)

        messages = [
            SystemMessage(
                content="You're a helpful assistant who answers questions related to a software system called 'The Everything App'. The following query discusses the file at path {context[0]}. Below are the contents of this file:\n\n{documents[0]}"),
            HumanMessage(
                content=query
            ),
        ]

        print(f"\n\n{messages}\n\n")

        response = self.chat_model.invoke(messages)

        print(f"\n\n{response}\n\n")

        return response.strip()

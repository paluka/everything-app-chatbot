from langchain_huggingface import HuggingFaceEmbeddings
from .youtube import get_channel_id, get_latest_video_ids, get_video_info, get_video_transcripts
from functools import lru_cache, wraps
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from typing import Any, Iterator, List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.errors import GraphInterrupt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
import uuid

# Set the environment variable to disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from langchain_community.llms.gpt4all import GPT4All
# from gpt4all import GPT4All


print()

NUMBER_OF_VIDEOS = 10
LLM_TEMPERATURE = 0

base_llm = ChatOllama(model="llama3.2", temperature=LLM_TEMPERATURE)
# model_path_and_file = "/Users/erikpaluka/Library/Application Support/nomic.ai/GPT4All/Llama-3.2-1B-Instruct-Q4_0.gguf"
# base_llm = GPT4All(model=model_path_and_file)
# print(base_llm.generate(["What is an apple?"]))
# + [AskHuman]) #, parallel_tool_calls=False


web_search_tool = TavilySearchResults(
    max_results=1,
    max_tokens=1000,
    # search_depth="advanced",
    include_answer=True,
    # include_raw_content=True,
    # include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    name="tavily_search_results_json",
    description=(
        "A search engine. Only use when you cannot answer the question yourself."
    ),
    # args_schema=...,       # overwrite default args_schema: BaseModel
)


@tool
@lru_cache(maxsize=100)
def youtube_tool(
    channel_name: Optional[str] = None,
    user_search_terms: Optional[str] = None
):
    """Get the part of the YouTube channel's videos' transcripts that
    discusses topics relevant to the user's search terms.

    Args:
        channel_name (str): the name of the YouTube channel.
        user_search_terms (str): the user's search terms
    """

    try:
        print(f"\n\n{channel_name}, {user_search_terms}\n\n")
        # channel_name_query = input('Enter a YouTube channel name: ')

        # if not channel_id or channel_id == 'NULL' or channel_id == 'NONE':

        if not channel_name or channel_name == 'NULL' or channel_name == 'NONE':
            return {"messages": [
                {"type": "error", "content": "Channel ID or Channel Name must be provided."}
            ]}

        channel_id, channel_title = get_channel_id(
            channel_name)

        print(f"\n\n{channel_id}, {channel_title}, {channel_name}\n\n")

        video_ids = get_latest_video_ids(
            channel_id, max_results=NUMBER_OF_VIDEOS)

        transcripts = get_video_transcripts(video_ids)

        # print(f"\n\nTranscripts:\n{transcripts}")

        relevant_transcripts = []

        for video_id, transcript in transcripts.items():
            print(f"\nVideo ID:\n{video_id}\n")
            # Check if transcript is a string before processing
            if isinstance(transcript, str):
                print(transcript)  # Prints "Transcript not available"
                continue  # Skip to the next iteration if it's a string

            data = {"video_id": video_id, "transcript": []}

            # Ensure 'segments' is a list before iterating
            # if isinstance(transcript.get('segments'), list):
            for entry in transcript['segments']:
                if user_search_terms in entry['text']:
                    chunk = f"{entry['start']} - {entry['text']}"
                    print(f"{chunk}")
                    data["transcript"].append(
                        {"time": entry['start'], "text": entry['text']})

            if len(data['transcript']) > 0:
                video_info = get_video_info(video_id)
                data['video_info'] = {
                    "publishedAt": video_info["snippet"]['publishedAt'],
                    "channelId": video_info["snippet"]['channelId'],
                    "title": video_info["snippet"]['title'],
                    "description": video_info["snippet"]['description'],
                    "duration": video_info['contentDetails']['duration'],
                }
                print(f"\n\nVideo Info:\n{data['video_info']}")
                relevant_transcripts.append(data)

            print("-" * 80)

        # NOW FIND user_search_terms in transcripts
        return {"documents": relevant_transcripts}
    except Exception as error:
        print(f"youtube_tool error: {error}")


@tool
class AskHuman(BaseModel):
    """Ask the human a question"""

    question: str


@tool
@lru_cache(maxsize=100)
def ask_large_language_model(
    question: str
):
    """Ask the large language model (LLM) a question.

    Args:
        question (str): the question to ask the large language model (LLM).
    """

    message = [
        # SystemMessage(
        #     content="Do not use tools. Answer the question by yourself."),
        HumanMessage(content=question),
        # (
        #     "user",
        #     "Use the search tool to ask the user where they are, then look up the weather there",
        # )
    ]

    print(f"\n\nAsking LLM a question: {question}\n\n")

    answer = base_llm.invoke(message)

    print(f"\n\nAsking LLM's answer: {answer}\n\n")

    # tool_message = [{
    #     "type": "tool",
    #     "name": 'ask_large_language_model',
    #     'args': {'question': 'What is 1 + 2?'},
    #     "content": answer
    # }]

    return {"llm_question": question, "llm_answer": answer.content}


class State(MessagesState):
    documents: list[str]
    selected_tools: list[str]
    human_question: str
    human_answer: str
    llm_question: str
    llm_answer: str
    verification: str
# class State(TypedDict):
    # messages: Annotated[list[Union[HumanMessage, AIMessage]], add_messages]


graph_builder = StateGraph(State)


tools = [web_search_tool, youtube_tool, AskHuman, ask_large_language_model]
tools_registry = {tool.name: tool for tool in tools}
# print(tools_registry)
# print()
# raise Exception

# llm_with_tools = base_llm.bind_tools(tools)


tool_documents = [
    Document(
        page_content=tool.description,
        id=id,
        metadata={"tool_name": tool.name},
    )
    for id, tool in tools_registry.items()
]
# print(tool_documents)
# print()
# raise Exception

###############
# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vector_store = InMemoryVectorStore(embedding=embedding)
# document_ids = vector_store.add_documents(tool_documents)
###############


def agent(state: State):
    try:
        # print(f"\n\nagent's full state: {state}\n\n")
        # print(f"\n\nagent's messages:\n{state['messages']}\n\n")
        print(f"\n\nagent's selected_tools:\n{state['selected_tools']}\n\n")
        selected_tools = [tools_registry[id] for id in state["selected_tools"]]
        # print(f"\n\nPast selected tools:\n{selected_tools}\n\n")
        llm_with_tools = base_llm.bind_tools(selected_tools)
        # print(f"\n\nBound tools\n\n")
        llm_response = llm_with_tools.invoke(state["messages"])
        print(f"\n\nagent's llm_response:\n{llm_response}\n\n")
        return {"messages": [llm_response]}
    except Exception as error:
        print(f"\n\nagent's exception:\n{error}\n\n")


def ask_human(state: State):
    try:
        last_message = state["messages"][-1]

        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:

            if hasattr(last_message, 'artifact') and last_message.artifact:
                query = last_message.artifact['query']
                results = last_message.artifact['results'][0]['content']

            elif hasattr(last_message, 'content') and last_message.content:
                content_dict = json.loads(last_message.content)
                query = content_dict['llm_question']
                results = content_dict['llm_answer']

            verification = interrupt({"question": f"""Is the following results appropriate for the specified query (Y\\N)? Query: {query}\n
                            Results: {results}\n\n"""})

            if verification in 'yes':
                return

            # last_message.artifact = None
            last_message.content = '{"verification": "The results are not appropriate"}'
            return
        else:
            last_message_tool_call = last_message.tool_calls[0]
            tool_call_question = last_message_tool_call["args"]["question"]

            print(f"\nask_human:\n{tool_call_question}\n")

            answer = interrupt(
                {"question": tool_call_question},
            )
            print(f"\n\nAnswer:\n{answer}")

            tool_message = [{
                "tool_call_id": last_message_tool_call["id"],
                "type": "tool",
                "content": answer
            }]

            return {"messages": tool_message, "human_question": tool_call_question, "human_answer": answer}
    except GraphInterrupt as resumable_error:
        # Let the resumable exception propagate
        raise resumable_error
    except Exception as error:
        print(f"\n\nask_human's exception:\n{error}\n\n")


def should_continue(state: State):
    try:
        last_message = state["messages"][-1]

        # print(f"\nShould continue?: {last_message}\n")

        if not last_message.tool_calls:
            return END
        elif last_message.tool_calls[0]["name"] == "AskHuman":
            return "ask_human"
        else:
            question = f"Do you want to invoke the following tool: {last_message.tool_calls[0]['name']}? (Y/N)"
            human_review = interrupt({"question": question})

            if human_review.lower() in 'yes':
                print(f"\n\nshould_continue: tools\n\n")
                return "tools"

            print(f"\n\nshould_continue: agent\n\n")
            return "agent"
    except GraphInterrupt as resumable_error:
        # Let the resumable exception propagate
        raise resumable_error
    except Exception as error:
        print(f"\n\should_continue's exception:\n{error}\n\n")


def select_tools(state: State):
    try:
        # last_user_message = state["messages"][-1]
        # query = last_user_message.content
        # tools_documents = vector_store.similarity_search(query, k=2)
        # print(
        #     f"""\n\nselect_tools: \nquery - -> {query}\n\ntools_document: \n
        #     {[document.id for document in tools_documents]}\n\n""")
        # return {"selected_tools": [document.id for document in tools_documents]}
        return {"selected_tools": [t.name for t in tools]}
    except Exception as error:
        print(f"\n\select_tools's exception:\n{error}\n\n")


tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_node("select_tools", select_tools)

graph_builder.add_node("agent", agent)

graph_builder.add_node("ask_human", ask_human)

graph_builder.add_edge(START, "select_tools")


graph_builder.add_edge("select_tools", "agent")

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    path_map=[END, "ask_human", "tools", "agent"]


)

graph_builder.add_edge("tools", "ask_human")
graph_builder.add_edge("ask_human", "agent")
# graph_builder.set_entry_point("agent")


graph_builder.add_edge("agent", END)


graph = graph_builder.compile(
    checkpointer=MemorySaver())  # , interrupt_before=["ask_human"],)  # , interrupt_before=["tools"],)

thread_config = {"configurable": {"thread_id": uuid.uuid4(), }}


def stream_graph_updates(graph_input: dict):
    for event in graph.stream(graph_input, thread_config, stream_mode="values"):
        # print(f"Event: {event} \n")

        if "messages" in event:
            event["messages"][-1].pretty_print()
        # for value in event.values():
        #     if value["messages"][-1].content.pretty_print():
        #         print("\nAssistant:", value["messages"][-1].content, "\n")
        #     else:
        #         print("\n...\n")
        #         print(value.pretty_print())
        #         print("\n...\n")

    snapshot = graph.get_state(thread_config)

    if snapshot.next:
        print(f"\nInterrupt Snapshot Next: \n{snapshot.next}\n")
        print(f"\nInterrupt Snapshot Tasks: \n{snapshot.tasks}\n")
        interrupt = snapshot.tasks[0].interrupts[0]
        print(f"\nActual interrupt: \n{interrupt}\n")

        print(
            f"\n\nTasks:\n{interrupt.value}\n")
        try:
            user_input = input(f"{interrupt.value['question']} User: ")

            if interrupt.resumable:
                # if user_input.lower() == 'y':
                stream_graph_updates(Command(resume=user_input))
        except Exception as error:
            print(f"Unable to resume graph traversal: {error}")
    else:
        print(f"\nEnd of graph traversal\n")
        # print(f"\nEnd of graph traversal:\n\n{snapshot}\n\n")


flag = False

while True:

    try:
        if flag:
            user_input = input("\nUser: ")
        else:
            # user_input = "Ask a human what age they are."
            user_input = "What is 1 + 2?"
            # user_input = "I want to see where Joe Rogan talks about fasting on his YouTube channel."
            flag = True

        if user_input.lower() in ["quit", "exit"]:
            break

        graph_input = {"messages": [
            # SystemMessage(
            #     content="Do not use tools. Answer the question by yourself."),
            HumanMessage(content=user_input),
            # (
            #     "user",
            #     "Use the search tool to ask the user where they are, then look up the weather there",
            # )
        ]}
        stream_graph_updates(graph_input)
    except:
        pass

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
#     ("user", "Question: {question}\nContext: {context}")
# ])

# output_parser = StrOutputParser()

# chain = prompt | model | output_parser

# output = chain.invoke(
#     {
#         "question": "What does this file do?",
#         "context": "The file talks about LangChain.",
#     }
# )

# print(output)
print("\n")

from langchain_ollama import ChatOllama
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

# from src.langgraph_agent.langgraph_chat import graph


app = FastAPI(
    title="My server: FastAPI",
    version="1.0",
    description="LangChain Server for LangServe Experimentation",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# add_routes(app, graph, path="/graph")

add_routes(
    app,
    ChatOllama(model="llama.3.2"),
    path="/ollama",
)


# model = ChatAnthropic(model="claude-3-haiku-20240307")
# prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
# add_routes(
#     app,
#     prompt | model,
#     path="/joke",
# )

if __name__ == "__main__":
    print(f"\n\nlangserve_api.py\n\n")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

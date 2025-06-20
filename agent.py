"""LangGraph Agent"""

import os

from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import ArxivLoader, WikipediaLoader

# https://python.langchain.com.cn/docs/modules/data_connection/text_embedding/integrations/sentence_transformers
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# pip install -qU  langchain_milvus
# pip install arxiv
# pip install pymupdf
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.

    Args:
        a: first int
        b: second int
    """
    return a - b


@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.

    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.

    Args:
        a: first int
        b: second int
    """
    return a % b


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.

    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"wiki_results": formatted_search_docs}


@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.

    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(input=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc["url"]}" />\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"web_results": formatted_search_docs}


@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.

    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["Title"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arvix_results": formatted_search_docs}


# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(content=system_prompt)

URI = "./milvus_example.db"

# https://zhuanlan.zhihu.com/p/29949362142
embedding_function = SentenceTransformerEmbeddings(model_name="moka-ai/m3e-base")

vector_store = Milvus(
    embedding_function=embedding_function,
    connection_args={"uri": URI},
    collection_name="documents",
)

retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="QuestionSearch",
    description="A tool to retrieve similar questions from a vector store.",
)


tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    web_search,
    arvix_search,
    retriever_tool,
]


# Build graph function
def build_graph(provider: str = "openai"):
    """Build the graph"""
    # Load environment variables from .env file
    if provider == "openai":
        llm = ChatOpenAI(
            model=os.environ["OPENAI_MODEL"],
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = vector_store.similarity_search(
            state["messages"][0].content, k=1
        )
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()


# test
if __name__ == "__main__":
    langfuse_handler = CallbackHandler()

    question = "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer."
    # Build the graph
    graph = build_graph(provider="openai")
    # Run the graph
    messages = [HumanMessage(content=question)]
    messages = graph.invoke(
        input={"messages": messages},
        config={"callbacks": [langfuse_handler]},
    )
    for m in messages["messages"]:
        m.pretty_print()

# https://huggingface.co/spaces/agents-course/Unit4-Final-Certificate

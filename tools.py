from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv

from Retrieval import retrieve

import os

load_dotenv()

# ---------------------------------------
# Tavily Client
# ---------------------------------------

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# ---------------------------------------
# PDF Retrieval Tool
# ---------------------------------------

@tool
def retrieve_pdf(query: str):
    """
    Retrieve relevant information from the ingested PDF collection.
    """

    try:

        return retrieve(query)

    except Exception as e:

        return f"Retrieval Error: {e}"

# ---------------------------------------
# Calculator Tool
# ---------------------------------------

@tool
def calculator(expression: str):
    """
    Evaluate mathematical expressions.
    """

    try:

        result = eval(expression)

        return str(result)

    except Exception as e:

        return f"Calculation Error: {e}"

# ---------------------------------------
# Web Search Tool
# ---------------------------------------

@tool
def web_search(query: str):
    """
    Search the web for current information.
    """

    try:

        result = client.search(
            query=query,
            max_results=3
        )

        return str(result)

    except Exception as e:

        return f"Web Search Error: {e}"
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv

import os

load_dotenv()

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)



@tool
def retrieve_pdf(query: str):
    """
    Search uploaded PDF documents.
    """

    return "PDF retrieval not implemented yet."
@tool
def calculator(expression: str):
    """
    Perform arithmetic calculations.
    """

    try:

        result = eval(expression)

        return str(result)

    except Exception as e:

        return f"Calculation Error: {e}"


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
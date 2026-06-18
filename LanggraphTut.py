from typing import TypedDict
from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langchain_ollama import ChatOllama

from tools import retrieve_pdf

class State(TypedDict):

    question: str

    pdf_context: str

    answer: str
llm = ChatOllama(
    model="mistral"
)

def retrieve_pdf_node(state):

    print("Entered retrieve_pdf_node")

    pdf_data = retrieve_pdf.invoke(
        state["question"]
    )

    print("Retrieved:", pdf_data)

    state["pdf_context"] = pdf_data

    return state
def answer_node(state):

    print("Entered answer_node")

    prompt = f"""
Context:
{state['pdf_context']}

Question:
{state['question']}
"""

    answer = llm.invoke(prompt)

    print("LLM Response:", answer)

    state["answer"] = answer.content

    return state
graph = StateGraph(State)
graph.add_node(
    "retrieve_pdf",
    retrieve_pdf_node
)

graph.add_node(
    "answer",
    answer_node
)
graph.add_edge(
    START,
    "retrieve_pdf"
)

graph.add_edge(
    "retrieve_pdf",
    "answer"
)

graph.add_edge(
    "answer",
    END
)
app = graph.compile()
result = app.invoke(
    {
        "question":
        "What is Google?",

        "pdf_context":"",

        "answer":""
    }
)


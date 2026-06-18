from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from tools import (
    retrieve_pdf,
    calculator,
    web_search
)


llm = ChatOllama(
    model="qwen3"
)


agent = create_agent(
    model=llm,
    tools=[
        retrieve_pdf,
        calculator,
        web_search
    ],
    system_prompt="""
You are an Agentic Document Intelligence Assistant.

Rules:

1. For arithmetic, use ONLY calculator.
2. For current events or live information, use ONLY web_search.
3. For document questions, use ONLY retrieve_pdf.
4. Never call unnecessary tools.
5. Answer using the tool results.
"""
)



while True:

    q = input("\nAsk: ")

    if q.lower() == "exit":
        break

    try:

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": q
                    }
                ]
            }
        )
        print(response)

        print(
            "\nAnswer:",
            response["messages"][-1].content
        )

    except Exception as e:

        print(
            "\nError:",
            e
        )
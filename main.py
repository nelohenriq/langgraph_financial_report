import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()


# Define state
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


# Create the graph
builder = StateGraph(GraphState)

# Create the LLM
from pydantic import SecretStr

llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=SecretStr(
        os.getenv(
            "GROQ_API_KEY", "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"
        )
    ),
    model="llama-3.2-3b-preview",
    temperature=0.0,
)


def create_node(
    state,
    system_prompt,
):
    try:
        human_messages = [
            msg for msg in state["messages"] if isinstance(msg, HumanMessage)
        ]
        ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        system_message = [SystemMessage(content=system_prompt)]
        messages = system_message + human_messages + ai_messages

        response = llm.invoke(messages)
        # Extract the message from the response
        message = response.content
        return {"messages": [AIMessage(content=message)]}
    except Exception as e:
        print(f"Error in LLM call: {str(e)}")
        return {
            "messages": [AIMessage(content="I encountered an error. Please try again.")]
        }


# Create the nodes
analyst = lambda state: create_node(
    state,
    "You are a software requirements analyst. Review the provided instructions and generate software development requirements that a developer can understand and create code from. Be precise and clear in your requirements.",
)

# Add nodes to graph
builder.add_node("analyst", analyst)

# Add edges to graph
builder.add_edge(START, "analyst")
builder.add_edge("analyst", END)

# Compile the graph
graph = builder.compile()

# Draw the graph
try:
    graph.get_graph(xray=True).draw_mermaid_png(output_file_path="graph.png")
except Exception:
    pass


# Create a main loop
def main_loop():
    # Run the chatbot
    while True:
        # Get the user input
        user_input = input(">> ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        response = graph.invoke({"messages": [HumanMessage(content=user_input)]})
        print("Analyst:", response["messages"][-1].content)


# Run the main loop
if __name__ == "__main__":
    main_loop()

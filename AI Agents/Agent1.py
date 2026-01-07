# [markdown]
# Simple Bot:
# - Define state structure with a list of HumanMessage objects
# - Initilize a GPT-4 moel
# - Sending and handling different kinds of messages
# - Building and compiling the graph of the Agent

from typing import TypedDict, List
from langgraph.graph import StateGraph, START,END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama



class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOllama(
    model="llama3.1",      # or "llama3.1:8b", "qwen2.5", etc.
    temperature=0.2,
)


def process(state:AgentState) -> AgentState:
    response=  llm.invoke(state['messages'])
    print(f"AI: {response.content}")

user_input = input("Enter something: ")

graph=StateGraph(AgentState)
graph.add_node("processor",process)
graph.add_edge(START,"processor")
graph.add_edge("processor",END)

agent = graph.compile()



agent.invoke({"messages":[HumanMessage(content = user_input)]})



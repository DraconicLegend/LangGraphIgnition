# 
# - Learn how to create Tools
# - Learn how to create a ReAct Grpah
# - Work with ToolMessages
# - Test the robustness of our graph


from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage # Foundational class for all LangGraph message types
from langchain_core.messages import ToolMessage # Passes data back to LLM after the tool call
from langchain_core.messages import SystemMessage # Message to provide instructions to LLM

from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START,END
from langgraph.prebuilt import ToolNode

# Annotated - Annotate a data piece with metadata
# email = Annotated[str,"This has to be a valid email format!"]

# print(email.__metadata__)


# Sequence - To automatically handle state updates for sequences such as  by adding new 
# messages to the chat history

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] ## Append, don't overwrite data

@tool # Decorater to indicate to python that this function is special
def add(a:int,b:int):
    """This is an addition function that adds two numbers together"""
    return a+b


@tool 
def subtract(a:int,b:int):
    """This is an subtraction function that adds two numbers together"""
    return a-b
@tool 
def multiply(a:int,b:int):
    """This is an multiplication function that adds two numbers together"""
    return a*b
tools = [add,subtract,multiply]


llm = ChatOllama(
    model="llama3.1",
    temperature=0.2,
).bind_tools(tools) # Give the llm access to your tools

def model_call(state:AgentState) -> AgentState:
    # OLD METHOD
    # response = llm.invoke(["You are my AI assistant, please answer your query to the best of my ability"])
    system_prompt = SystemMessage(content="You are my AI assistant, please answer your query to the best of my ability")
    # We STILL NEED TO PASS IN QUERY
    response = llm.invoke([system_prompt]+state['messages'])
    
    # Below is a much preferred way to write the updated state - becasue we used add_messages(reducer function) handles the appending for us
    return {"messages":[response]}


def should_continue(state: AgentState):
    messages = state['messages']
    last_messages = messages[-1]
    if not last_messages.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools= tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end": END
    }
)
graph.add_edge("tools","our_agent")


app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()
inputs = {"messages":[("user","Add 40+12 and multiply result by 6.Also tell me a joke")]}
print_stream(app.stream(inputs,stream_mode="values"))

# This is imporntat b/c LLM doesn't actually 'think'. It determines next resulst on probability
# By using tools, we can get the LLM to do the 'right' operation every time as long as the underlying
# logic is all good
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage,ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START,END
from langgraph.prebuilt import ToolNode

#Global var to store doc content
document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]


@tool
def update(content:str) -> str:
    """Update the document with provided content"""
    global document_content # Standard procedure to interact with global variables in python
    document_content = content
    return f"Document has been updated sucessfully! The current content is: \n{document_content}"

@tool
def save(filename:str) -> str:
    """Save the current document to a text file and finish the process.
    Args:
        filenames: Name for the text file.
    """

    # Just in case the LLM doesn't sift through the .txt automatically
    if not filename.endswith(".txt"):
        filename=f"{filename}.txt"
    
    # Access the current draft
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    

tools = [update, save]


llm = ChatOllama(
    model="llama3.1",
    temperature=0.2,
).bind_tools(tools) # Give the llm access to your tools

def our_agent(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)
    # Introduction messages
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = llm.invoke(all_messages)

    # PRINT STATEMENTS THAT MAKE TERMINAL LOOK NICE
    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    #RETURN
    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state:AgentState)->str:
    """Function I made to print the messages in a more readable format"""
    messages = state['messages']
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from 'save' function
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")



graph = StateGraph(AgentState)

graph.add_node("Agent",our_agent)
graph.add_node("Tools",ToolNode(tools))

graph.set_entry_point("Agent")
graph.add_edge("Agent","Tools")

graph.add_conditional_edges("Tools",should_continue,
                            {
                                "continue":"Agent",
                                "end":END,
                            },)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()
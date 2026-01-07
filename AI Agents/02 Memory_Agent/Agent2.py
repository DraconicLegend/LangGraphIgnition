# Objectives:
# Use different Message types - HumanMessage & AIMessage
# Maintain convo history
# Create a sophisticated convo loop


from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START,END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama


class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]


llm = ChatOllama(
    model="llama3.1",
    temperature=0.2,
)

def process(state:AgentState) ->AgentState:
    """This node will make the llm solve the request you input"""
    response = llm.invoke(state['messages'])
    
    state['messages'].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    
    return state

graph=StateGraph(AgentState)
graph.add_node("processor",process)
graph.add_edge(START,"processor")
graph.add_edge("processor",END)

agent=  graph.compile()

conversation_history = []

user_input = input("Enter something: ")

while user_input!= "exit":
    conversation_history.append(HumanMessage(content = user_input))
    
    result =agent.invoke({"messages":conversation_history})
    
    conversation_history = result['messages']
    user_input = input("Enter:")
    

## CODE FOR TEXT FILE EXPORTATION

with open("logging.txt","w") as file:
    file.write("Conversation Log:\n")

    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f"You: {message.content}\n")
        if isinstance(message,AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

    ## Ideally use a database or vec db in actual production


    ## BIG PROBLEM: passing in 'conversation_history' will take too many AI Tokens,
    # so you could delete the first convo and make the limit 5 most recent or something
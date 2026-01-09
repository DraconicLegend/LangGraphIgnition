from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
#Embedding Model
from langchain_ollama import OllamaEmbeddings
#Vector Embedding DB
from langchain_chroma import Chroma
import os


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



llm = ChatOllama(
    model="llama3.1",
    temperature=0, # Minimize hallucinations
)


embeddings =OllamaEmbeddings(model = "mxbai-embed-large")

pdf_path = "Stock_Market_Performance_2024.pdf"

# Safety measure for debugging purposes :)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

## LOADS PDF
pdf_loader = PyPDFLoader(pdf_path) 


# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Need to chunk document 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)# LMI - why is overlap needed?

pages_split = text_splitter.split_documents(pages)

persist_directory =  "C:\\Users\\There\\StudioPrograms\\Agentic AI\\AI_Agents\\05_RAG"
collection_name = "stock_market"


# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # Here, we actually create the chroma database using our embeddings model
    vectorstore = Chroma.from_documents(
        documents=pages_split, # The chunks of the document that we're passing into the database
        embedding=embeddings, # The embedding model we embed the text in the database with
        persist_directory=persist_directory, # The directory we wish the db to be stored in
        collection_name=collection_name # The name of the database
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return - don't make it too low or high. 4/5 is good middle ground
)

@tool
def retriever_tool(query:str) -> str:
    """This tool searches and returns the information from the Stock Market Performance 2024 document."""
    docs = retriever.invoke(query)

    if not docs:
        # If no similarity, then return this information to the LLM
        return "I found no relevant information in the Stock Market Performance 2024 document." 
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

tools = [retriever_tool]

llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

# Our function for the conditional edge 
def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
""" # That last line prevents/alleviates hallucination

tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages # Add system prompt to beginning of message history
    message = llm.invoke(messages) # LLM Call
    return {'messages': [message]} # Updates AgentState


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    # 1) Extract Tool Calls
    tool_calls = state['messages'][-1].tool_calls
    results = []

    # 2) Loop through the tools call dictionary
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        #3) Check if each tool exists and is usable
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        #4)Execute it if it's a valid tools
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # 5) Appends the Tool Message from running it
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

# Create the graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()
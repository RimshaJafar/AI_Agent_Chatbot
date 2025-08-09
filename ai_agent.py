#Step1: Setup API Keys for Groq,OpenAI and Tavity
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

#Step2: Setup LLM & Tools

from langchain_groq import ChatGroq
from  langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain_tavily import TavilySearchResults



openai_llm=ChatOpenAI(model="gpt-4o-mini")
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")





#Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
system_prompt="Act as an AI chatbot who is smart and intelligent"

def get_response_from_ai_agent(llm_id,query,allow_search,system_prompt,provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id) 
    tools=[TavilySearchResults(max_results=2) ]if allow_search else []      
    agent = create_react_agent(
        model=groq_llm,
        tools=tools,
       # system_message=system_prompt
    )
    query = [f"System: {system_prompt}"] + query
    state={"messages":query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message,AIMessage)]
    return ai_messages[-1]
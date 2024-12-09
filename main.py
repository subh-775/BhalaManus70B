from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools import DuckDuckGoSearchRun
import streamlit as st

st.set_page_config(page_title="Bhala Manus")
st.header("No Back Abhiyan")

# Sidebar inputs
st.sidebar.markdown("### Configuration")
index_name = st.sidebar.text_input(
    "Index Name",
    value="ml-docs",  # Default value
    help="Enter the name of the index to use."
)
groq_api_key = st.sidebar.text_input("LLM API Key", type="password", help="Enter your Pinecone API key.")

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        use_vector_store = st.checkbox("Use Documents", value=True)
    with col2:
        use_chat_history = st.checkbox("Use ChatHistory (Last 2 Chats)", value=False)

use_web=st.sidebar.checkbox("Allow Internet Access", value=True)

if use_chat_history:
    use_vector_store,use_web=False,False


st.sidebar.markdown("""
--- 
**Instructions:**  
Get your *Free-API-Key*
From **[Groq](https://console.groq.com/keys)**

---
Kheliye *meating - meeting*
""")

# st.write(st.session_state)
if "vector_store" not in st.session_state:
    st.write("Connecting to Vector Database ....")
    pc = Pinecone(api_key="pcsk_6KAu86_9Zzepx9S1VcDmLRmBSUUNpPf4JRbE4BaoVmk36yW9R4nkjutPiZ3AjZvcyL4MVx")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyARa0MF9xC5YvKWnGCEVI4Rgp0LByvYpHw")
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    st.session_state["vector_store"]=vector_store
else:
    vector_store=st.session_state["vector_store"]
if groq_api_key:
    if "llm" not in st.session_state:
        llm=ChatGroq(temperature=0.2,model="llama-3.1-70b-versatile",api_key=groq_api_key)
        st.session_state["llm"]=llm
    else:
        llm=st.session_state["llm"]





def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def clean_rag_data(query,context,llm):
    llm = llm
    system = """
        You are a Highly capable Proffesor of understanding the value and context of both user queries and given data. \n
        Your Task for Documents Data is to analyze the list of document's content and properties and find the most important information regarding user's query
        Your Task for ChatHistory Data is to analyze the given ChatHistory and then provide a ChatHistroy relevant to user's query.
        Your Task for Web Data is to analyze the web scraped data then summarize only useful data regarding user's query.
        You Must adhere to User's query before answering.
        
        Output:
            For Document Data
                Conclusion:
                    ...
            For ChatHistory Data
                    User: ...
                    ...
                    Assistant: ...
            For Web Data
                Web Scarped Data:
                ...
        """

    user = """{context}
            
            User's query is given below:
            {question}
            
            """

    filtering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user)
        ]
    )

    filtering_chain = filtering_prompt | llm | StrOutputParser()

    response = filtering_chain.invoke({"context": context, "question": query})

    return response
    
def get_llm_data(query,llm):
    system = """
        You are a knowledgeable and approachable Computer Science professor with expertise in a wide range of topics, including algorithms, data structures, operating systems, databases, artificial intelligence, and software development. 

Your role is to provide clear, easy, and engaging explanations to help students understand complex concepts. When answering:
- Start with a high-level overview, then dive into details as needed.
- Use examples, analogies, or step-by-step explanations to clarify ideas.
- Ensure your answers are accurate, well-structured, and easy to follow.
- If you don’t know the answer, acknowledge it and suggest ways to explore or research further.

Maintain a tone that is patient, encouraging, and enthusiastic about learning.

        """

    user = """{query}
            """

    filtering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user)
        ]
    )

    filtering_chain = filtering_prompt | llm | StrOutputParser()

    response = filtering_chain.invoke({ "query": query})

    return response

def get_context(query):
    context=""
    if use_vector_store:
        st.write("Extracting Data From VectorStore")
        result="\n\n".join([_.page_content for _ in vector_store.similarity_search(query, k = 3)])
        clean_data=clean_rag_data(query,f"Documents Data \n\n{result}",llm)
        context+=f"Documents Data: \n\n{clean_data}"
    
    if use_chat_history:
        st.write("Extracting Data From ChatHistory")
        last_messages = st.session_state.messages[:-1][-3:]
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_messages])
        clean_data=clean_rag_data(query,f"\n\nChat History \n\n{chat_history}",llm)
        context += f"\n\nChat History: \n\n{clean_data}"
    
    if use_web:
        st.write("Extracting Data From Web")
        search = DuckDuckGoSearchRun()
        clean_data=clean_rag_data(query,search.invoke(query),llm)
        context += f"\n\nWeb Data:\n{clean_data}"

    if not use_chat_history:
        st.write("Extracting Data From ChatPPT")
        context +=f"\n\n LLM Data{get_llm_data(query,llm)}"
    
    
    return context

def respond_to_user(query,context,llm):
    system_prompt = """
You are a Assistant specialized in Machine Learning (ML) tasks. Your job is to answer the given question based on the following types of context: 

1. **Web Data**: Information retrieved from web searches.
2. **Documents Data**: Data extracted from documents (e.g., research papers, reports).
3. **Chat History**: Previous interactions or discussions in the current session.
4. **LLM Data**: Insights or completions provided by the language model.

When answering:
- Clearly identify the source(s) of the information you are using.
- If multiple sources are available, integrate them into a cohesive summary.
- If no relevant context is available, explicitly state that the answer is not known.
- Ensure your response is clear and and easy to understand even for a naive person.

Also explain how you arrived at your conclusion and what contributed to it.
"""
    user_prompt="""Question: {question} 

    Context: {context} """

    rag_chain_prompt = ChatPromptTemplate.from_messages(
        
        [
            ("system",system_prompt),
            ("user", user_prompt)
        ]
    )

    rag_chain = rag_chain_prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"question": query, "context": context})

    return response


if "messages" not in st.session_state.keys():
    content="Padhoge Likhoge Bnoge Kharab , Kheloge Khudoge Banoge Nawab "
    st.session_state.messages = [{"role": "assistant", "content": content}]

display_chat_history()

if input_data := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input_data})
    context=get_context(input_data)
    st.write("Combining jhol Jhal")
    assistant_response = respond_to_user(input_data,context,llm)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    display_chat_history()









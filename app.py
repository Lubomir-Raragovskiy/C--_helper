import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG С++")
st.title("Інтерактивна RAG-система (С++)")
st.markdown("---")

@st.cache_resource
def init_rag_system():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 6}
    )

    system_prompt = (
        "Ти експерт з C++. Використовуй наданий контекст для формування розгорнутої відповіді. "
        "Якщо в контексті є хоча б часткова інформація, синтезуй її та надай відповідь, але використовуй лише дані з контексту. "
        "Якщо ж інформації взагалі немає, скажи: 'У наданих документах відсутня необхідна інформація...'. "
        "Відповідай українською мовою. Відповідай стисло.\n\n"
        "Контекст:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name="qwen/qwen3.8-27b",
        max_tokens=1000,
        temperature=0.0,
    )
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    return rag_chain

rag_chain = init_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            st.markdown("**Джерела інформації:**")
            for i, doc in enumerate(message["sources"]):
                source_file = os.path.basename(doc.metadata.get('source', 'Невідомий файл'))
                with st.expander(f"Фрагмент {i+1} (Джерело: {source_file})"):
                    st.text(doc.page_content)


user_query = st.chat_input("Введіть пошуковий запит...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.spinner("Аналіз документів та генерація відповіді через Groq..."):
        try:
            response = rag_chain.invoke({"input": user_query})
            answer_text = response["answer"]
            context_docs = response["context"]
            
            with st.chat_message("assistant"):
                st.markdown(answer_text)
                
                st.markdown("**Джерела інформації:**")
                for i, doc in enumerate(context_docs):
                    source_file = os.path.basename(doc.metadata.get('source', 'Невідомий файл'))
                    with st.expander(f"Фрагмент {i+1} (Джерело: {source_file})"):
                        st.text(doc.page_content)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": context_docs
            })
            
        except Exception as e:
            st.error(f"Виникла помилка: {e}")
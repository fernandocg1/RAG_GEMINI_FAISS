import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import OpenAI, OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '..', '.env')
load_dotenv(dotenv_path=DOTENV_PATH) 

CAMINHO_DB = "faiss_md_index"

prompt_template_str = """
Você é um assistente de IA especializado em responder perguntas sobre estruturas condicionais em Python.

Instrução: Utilize o **Contexto** fornecido para responder à **Pergunta** de forma precisa e concisa. Se o contexto não tiver a informação necessária, responda que não sabe.

--- CONTEXTO ---
{contexto}
---

Pergunta: {pergunta}
Resposta:
"""
prompt = PromptTemplate.from_template(prompt_template_str)

@st.cache_resource
def carregar_componentes_rag():
    """Carrega o LLM, Embeddings e o Banco de Dados FAISS."""
    try:
        llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)
        funcao_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        db = FAISS.load_local(CAMINHO_DB, funcao_embeddings, allow_dangerous_deserialization=True)
        
        return llm, db
    except Exception as e:
        st.error(f"Erro ao carregar o RAG. Verifique a chave de API ou o banco de dados. Erro: {e}")
        st.stop()
        return None, None 

def gerar_resposta(llm, db, pergunta):
    """Executa a busca (Retrieval) e a geração de resposta (Augmentation)."""
    
    resultados_docs = db.similarity_search_with_relevance_scores(pergunta, k=3)
    
    if not resultados_docs:
        return " A busca não retornou documentos relevantes. Tente outra pergunta."
    
    contexto = "\n\n".join([doc.page_content for doc, score in resultados_docs])
    
    prompt_final = prompt.format(contexto=contexto, pergunta=pergunta)
    resposta = llm.invoke(prompt_final)
    
    return resposta.strip()


st.title(" Assistente RAG de Estruturas Condicionais (Python)")
st.caption("Baseado no seu documento Markdown")

llm, db = carregar_componentes_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if pergunta_usuario := st.chat_input("Faça sua pergunta sobre estruturas condicionais..."):
    
    st.session_state.messages.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Buscando e gerando resposta..."):
            resposta_ia = gerar_resposta(llm, db, pergunta_usuario)
        
        st.markdown(resposta_ia)
    
    st.session_state.messages.append({"role": "assistant", "content": resposta_ia})
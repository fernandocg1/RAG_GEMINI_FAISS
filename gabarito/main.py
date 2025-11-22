import os
from dotenv import load_dotenv

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

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)
funcao_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

try:
    db = FAISS.load_local(CAMINHO_DB, funcao_embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Erro ao carregar o Banco de Dados FAISS: {e}")
    exit()


def chat_rag():
    print("\n--- Sistema RAG Ativado ---")
    print("Digite 'sair' a qualquer momento para fechar.")
    
    while True:
        pergunta = input("\n Digite sua pergunta sobre estruturas condicionais em Python: ")
        
        if pergunta.lower() in ['sair', 'exit']:
            print("Encerrando o assistente. Até logo!")
            break
        
        if not pergunta.strip():
            continue

        resultados_docs = db.similarity_search_with_relevance_scores(pergunta, k=3)
        print(f"\nNúmero de resultados encontrados: {len(resultados_docs)}")

        if not resultados_docs:
            print("\n A busca não retornou documentos. Tente outra pergunta.")
            continue
            
        contexto = "\n\n".join([doc.page_content for doc, score in resultados_docs])

        prompt_final = prompt.format(contexto=contexto, pergunta=pergunta)
        resposta = llm.invoke(prompt_final)

        print("\n=============================================")
        print(f" Resposta da IA:\n{resposta.strip()}")
        print("=============================================")

if __name__ == "__main__":
    chat_rag()
import os
from dotenv import load_dotenv

from langchain_openai import OpenAI, OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv(dotenv_path="../.env")

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

pergunta = input("Digite sua pergunta sobre estruturas condicionais em Python: ")

funcao_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = FAISS.load_local(CAMINHO_DB, funcao_embeddings, allow_dangerous_deserialization=True)

resultados_docs = db.similarity_search_with_relevance_scores(pergunta, k=3)
print(f"Resultados brutos da busca: {resultados_docs}")
print(f"Número de resultados: {len(resultados_docs)}")

contexto = "\n\n".join([doc.page_content for doc, score in resultados_docs])

prompt_final = prompt.format(contexto=contexto, pergunta=pergunta)

resposta = llm.invoke(prompt_final)

print("\n=============================================")
print(f"Resposta da IA:\n{resposta.strip()}")
print("=============================================")
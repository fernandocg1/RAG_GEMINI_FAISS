import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(dotenv_path="../.env")

PASTA_BASE = "base/estruturas_condicionais.md"
PASTA_DB = "faiss_md_index" 

def criar_db():
    print("--- 1. Carregando documentos...")
    documentos = carregar_documentos()
    
    if not documentos:
        print("ERRO: Nenhum documento carregado. Verifique o caminho e o arquivo .md.")
        return

    print(f"Documentos carregados: {len(documentos)}")
    
    print("--- 2. Dividindo documentos em Chunks...")
    chunks = dividir_chunks(documentos) 
    print(f"Chunks criados: {len(chunks)}")
    
    print("--- 3. Criando o Banco de Dados Vetorial (FAISS)...")
    criar_vetor_db(chunks)
    print("Sucesso! Banco de Dados FAISS criado e salvo.")

def carregar_documentos():
    """Carrega o arquivo Markdown usando o UnstructuredMarkdownLoader."""
    try:

        carregador = UnstructuredMarkdownLoader(PASTA_BASE)
        documentos = carregador.load()
        return documentos
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return []

def dividir_chunks(documentos):
    """Divide os documentos carregados em pedaços menores (chunks)."""
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len,
        add_start_index=True,
    )
    chunks = separador_documentos.split_documents(documentos)
    return chunks

def criar_vetor_db(chunks):
    """Cria embeddings com OpenAI e salva no FAISS."""
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    db = FAISS.from_documents(chunks, embeddings)
    
    if not os.path.exists(PASTA_DB):
        os.makedirs(PASTA_DB)

    db.save_local(PASTA_DB)


if __name__ == "__main__":
    criar_db()
import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- SISTEMA DE PROTECCIÓN DE API KEY (COMPATIBLE LOCAL Y NUBE) ---
try:
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        os.environ["OPENAI_API_KEY"] = "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"
except Exception:
    os.environ["OPENAI_API_KEY"] = "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"

# Configuración estética de la página
st.set_page_config(page_title="VetLex Bot", page_icon="⚖️", layout="wide")

# --- BARRA LATERAL (DECORACIÓN) ---
with st.sidebar:
    st.markdown("# 🐾 VetLex Bot")
    st.markdown("### Asistente Legal Veterinario")
    st.image("https://cdn-icons-png.flaticon.com/512/8993/8993711.png", width=100)
    st.write("---")
    
    st.markdown("#### 📚 Base de Conocimiento:")
    st.info("Leyes, normas oficiales mexicanas, regulaciones sanitarias y campañas de salud animal en México.")
    
    st.write("---")
    if st.button("🗑️ Limpiar Historial de Chat"):
        st.session_state.messages = []
        st.rerun()
        
    st.caption("Desarrollado con ❤️ para el sector veterinario en México.")

# --- CUERPO PRINCIPAL ---
st.title("⚖️ VetLex Bot: Inteligencia Sanitaria")
st.subheader("Consulta la legislación de interés en Medicina Veterinaria.")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def inicializar_sistema_chat():
    try:
        embeddings = OpenAIEmbeddings()
        # Conecta con la base de datos donde procesaste tu PDF leyes_vet.pdf
        vector_db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Ajustamos el prompt para que use TODO el contexto del PDF sin autolimitarse
        prompt = ChatPromptTemplate.from_template("""
        Eres VetLex, un asistente legal experto en legislación, regulaciones sanitarias y derecho veterinario en México.
        Responde la pregunta del usuario utilizando de forma exhaustiva la información del contexto proporcionado.
        Si el tema solicitado (como una campaña o norma específica) no viene mencionado en los documentos, explícale amablemente al usuario qué temas sí cubre el documento actual y qué norma externa necesitaría agregar.
        
        Contexto:
        {context}
        
        Pregunta: {input}
        """)
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_db.as_retriever(search_kwargs={"k": 4}) # Aumentamos a 4 fragmentos para mayor contexto
        return create_retrieval_chain(retriever, document_chain)
    except Exception as e:
        st.error(f"Error al inicializar: {e}")
        return None

conversational_rag_chain = inicializar_sistema_chat()

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if user_query := st.chat_input("Escribe tu consulta sobre legislación veterinaria aquí..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with chat_container:
        with st.chat_message("user"):
            st.write(user_query)
            
        if conversational_rag_chain:
            with st.chat_message("assistant"):
                with st.spinner("Buscando en la legislación veterinaria..."):
                    try:
                        response = conversational_rag_chain.invoke({"input": user_query})
                        answer = response["answer"]
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {e}")
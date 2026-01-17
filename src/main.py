"""
Fichier principal pour exécuter l'application Chatbot AI
Fonctions : 
- Créer une interface web avec Streamlit
- Gérer l'interaction avec l'utilisateur
- Se connecter au modèle AI pour répondre aux questions
"""

# === IMPORTATION DES BIBLIOTHÈQUES NÉCESSAIRES ===
import streamlit as st  # Librairie pour créer l'interface web
from dotenv import load_dotenv  # Lire le fichier .env contenant les clés API
from seed_data import seed_milvus, seed_milvus_live  # Fonctions de traitement des données
from agent import get_retriever as get_openai_retriever, get_llm_and_agent as get_openai_agent
from local_ollama import get_retriever as get_ollama_retriever, get_llm_and_agent as get_ollama_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import os
import json
# === CONFIGURATION DE LA PAGE WEB ===
def setup_page():
    """
    Configuration de la page web de base
    """
    st.set_page_config(
        page_title="Assistant AI",  # Titre de l'onglet
        page_icon="💬",  # Icône de l'onglet
        layout="wide"  # Interface large
    )

# === INITIALISATION DE L'APPLICATION ===
def initialize_app():
    """
    Initialise les paramètres nécessaires :
    - Lire les clés API depuis le fichier .env
    - Configurer la page web
    """
    load_dotenv()  # Lire les clés API
    setup_page()   # Configurer l'interface

# === BARRE LATÉRALE DE CONFIGURATION ===
def setup_sidebar():
    """
    Crée la barre latérale avec les options de configuration
    """
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Partie 1 : Choix du modèle d'embeddings
        st.header("🔤 Modèle d'Embeddings")
        embeddings_choice = st.radio(
            "le modèle d'Embeddings :",
            ["Ollama"]
        )
        use_ollama_embeddings = (embeddings_choice == "Ollama")
        
        # Partie 2 : Choix de la source de données
        st.header("📚 Source des données")
        data_source = st.radio(
            "Choisir la source des données :",
            ["Fichier local", "URL directe"]
        )
        
        # Traiter la source en fonction du choix d'embeddings
        if data_source == "Fichier local":
            handle_local_file(use_ollama_embeddings)
        else:
            handle_url_input(use_ollama_embeddings)
            
        # Partie 3 : Nom de la collection pour la recherche
        st.header("🔍 Collection à interroger")
        collection_to_query = st.text_input(
            "Nom de la collection à interroger :",
            "finance_data",
            help="Indiquer le nom de la collection à utiliser pour la recherche"
        )
        
        # Partie 4 : Choix du modèle pour répondre
        st.header("🤖 Modèle AI")
        model_choice = st.radio(
            "le modèle AI pour répondre :",
            ["Ollama (Local)"]
        )
        
        return model_choice, collection_to_query

import pandas as pd

def handle_local_file(use_ollama_embeddings: bool):
    """
    Gérer l'import d'un fichier local via un explorateur de fichiers
    Supporte JSON et XLSX
    """
    collection_name = st.text_input(
        "Nom de la collection dans Milvus:", 
        "finance_data",
        help="Indiquer le nom de la collection pour stocker les données"
    )
    
    uploaded_file = st.file_uploader(
        "Choisir un fichier depuis ton ordinateur ",
        type=["json", "xlsx"]
    )
    
    if uploaded_file is not None and st.button("Importer les données"):
        if not collection_name:
            st.error("Veuillez saisir un nom de collection !")
            return
        
        try:
            filename = uploaded_file.name
            extension = filename.rsplit(".", 1)[-1].lower()
            
            # =======================
            # Traitement JSON
            # =======================
            if extension == "json":
                try:
                    data = json.load(uploaded_file)
                except UnicodeDecodeError:
                    # Essayer avec fallback encodage latin1 si utf-8 échoue
                    uploaded_file.seek(0)
                    data = json.load(uploaded_file.read().decode("latin1"))
                
            # =======================
            # Traitement XLSX
            # =======================
            elif extension == "xlsx":
                df = pd.read_excel(uploaded_file)
                # Convertir DataFrame en liste de dicts compatible avec seed_milvus
                data = df.to_dict(orient="records")
            
            else:
                st.error("Type de fichier non supporté !")
                return
            
            # Sauvegarder temporairement pour que seed_milvus puisse le lire
            temp_dir = "temp_data"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            with st.spinner("Importation des données en cours..."):
                seed_milvus(
                    host="localhost",
                    port="19530",
                    collection_name=collection_name,
                    filename=filename,
                    repertoire=temp_dir,
                    utiliser_ollama=use_ollama_embeddings
                )
                st.success(f"Données importées avec succès dans la collection '{collection_name}' !")
        
        except Exception as e:
            st.error(f"Erreur lors de l'importation : {str(e)}")

def handle_url_input(use_ollama_embeddings: bool):
    """
    Gérer l'import de données depuis une URL
    """
    collection_name = st.text_input(
        "Nom de la collection dans Milvus :", 
        "finance_data",
        help="Indiquer le nom de la collection pour stocker les données"
    )
    url = st.text_input("Saisir l'URL :")
    
    if st.button("Crawling des données"):
        if not collection_name:
            st.error("Veuillez saisir un nom de collection !")
            return
            
        with st.spinner("Crawling des données en cours..."):
            try:
                seed_milvus_live(
                    url, 
                    "localhost",
                     "19530",
                    collection_name, 
                    'stack-ai', 
                    utiliser_ollama=use_ollama_embeddings
                )
                st.success(f"Données crawlé et stockées dans la collection '{collection_name}' !")
            except Exception as e:
                st.error(f"Erreur lors du crawling : {str(e)}")

# === INTERFACE PRINCIPALE DE CHAT ===
def setup_chat_interface(model_choice):
    st.title("💬 Assistant AI")
    
    # Caption dynamique selon le modèle
    if model_choice == "OpenAI GPT-4":
        st.caption("🚀 Assistant AI basé sur LangChain et OpenAI GPT-4")
    elif model_choice == "OpenAI Grok":
        st.caption("🚀 Assistant AI basé sur LangChain et X.AI Grok")
    else:
        st.caption("🚀 Assistant AI basé sur Ollama LLaMA2")
    
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Je peux vous aider avec vos questions financières."}
        ]
        msgs.add_ai_message("Je peux vous aider avec vos questions financières.")

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

# === TRAITEMENT DES MESSAGES UTILISATEUR ===
def handle_user_input(msgs, agent_executor):
    """
    Traiter un message utilisateur :
    1. Afficher le message
    2. Appeler l'IA pour générer une réponse
    3. Sauvegarder dans l'historique
    """
    if prompt := st.chat_input("Posez-moi une question sur la finance ou l'économie !"):
        # Afficher le message utilisateur
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Afficher la réponse de l'AI
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Historique de chat pour le contexte
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            # Appel au modèle AI
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                {"callbacks": [st_callback]}
            )

            # Sauvegarder et afficher la réponse
            output = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)

# === FONCTION PRINCIPALE ===
def main():
    """
    Fonction principale pour exécuter l'application
    """
    initialize_app()
    model_choice, collection_to_query = setup_sidebar()
    msgs = setup_chat_interface(model_choice)
    
    # Initialiser l'agent selon le modèle choisi
    if model_choice == "OpenAI GPT-4":
        retriever = get_openai_retriever(collection_to_query)
        agent_executor = get_openai_agent(retriever, "gpt4")
    elif model_choice == "OpenAI Grok":
        retriever = get_openai_retriever(collection_to_query)
        agent_executor = get_openai_agent(retriever, "grok")
    else:
        retriever = get_ollama_retriever(collection_to_query)
        agent_executor = get_ollama_agent(retriever)
    
    handle_user_input(msgs, agent_executor)

# Lancer l'application
if __name__ == "__main__":
    main()

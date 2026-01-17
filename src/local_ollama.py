# =====================================================
# local_ollama.py
# Gestion du retriever + agent Ollama (local)
# =====================================================

from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from seed_data import connect_to_milvus

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# =====================================================
# RETRIEVER (Milvus + BM25)
# =====================================================
def obtenir_retriever(nom_collection: str = "finance_data") -> EnsembleRetriever:
    """
    Crée un retriever hybride :
    - Vectoriel (Milvus)
    - Lexical (BM25)
    """

    try:
        # Connexion correcte à Milvus
        vectorstore = connect_to_milvus(
            host="localhost",
            port="19530",
            collection_name=nom_collection,
            utiliser_ollama=True
        )

        # Retriever vectoriel
        retriever_vectoriel = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Récupérer un échantillon pour BM25
        docs = vectorstore.similarity_search("finance", k=100)

        if not docs:
            raise ValueError(f"Aucun document trouvé dans la collection '{nom_collection}'")

        documents = [
            Document(page_content=d.page_content, metadata=d.metadata)
            for d in docs
        ]

        # Retriever BM25
        retriever_bm25 = BM25Retriever.from_documents(documents)
        retriever_bm25.k = 4

        # Combinaison des deux retrievers
        return EnsembleRetriever(
            retrievers=[retriever_vectoriel, retriever_bm25],
            weights=[0.7, 0.3]
        )

    except Exception as e:
        print("❌ Erreur retriever :", e)

        # Fallback si Milvus est vide ou inaccessible
        return BM25Retriever.from_documents([
            Document(
                page_content="Aucune donnée financière disponible pour le moment.",
                metadata={"source": "erreur"}
            )
        ])


# =====================================================
# AGENT OLLAMA
# =====================================================
def obtenir_llm_et_agent(retriever):
    """
    Crée l'agent LangChain avec Ollama
    """

    outil_recherche = create_retriever_tool(
        retriever,
        "recherche_financiere",
        "Recherche des informations financières dans la base documentaire."
    )

    llm = ChatOllama(
        model="llama2",
        temperature=0,
        streaming=True
    )

    system_prompt = """
Tu es FinChatAI, un assistant financier professionnel.

Règles :
- Tu réponds uniquement à partir des documents fournis
- Si l'information n'existe pas, dis clairement : "Je ne trouve pas cette information"
- Réponds toujours en français
- Sois clair, structuré et précis
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(
        llm=llm,
        tools=[outil_recherche],
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=[outil_recherche],
        verbose=True
    )


# =====================================================
# ALIAS POUR main.py (IMPORTANT)
# =====================================================
get_retriever = obtenir_retriever
get_llm_and_agent = obtenir_llm_et_agent

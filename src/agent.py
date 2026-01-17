# =========================
# agent.py (VERSION FINALE)
# =========================

from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from seed_data import connect_to_milvus


# =========================
# RETRIEVER FINANCIER
# =========================
def obtenir_retriever(nom_collection: str = "finance_data") -> EnsembleRetriever:
    """
    Combine Milvus (vectoriel) + BM25 (texte)
    """
    try:
        vectorstore = connect_to_milvus(
            "http://localhost:19530",
            nom_collection
        )

        retriever_vectoriel = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Récupérer des documents pour BM25
        docs = vectorstore.similarity_search("", k=100)

        if not docs:
            raise ValueError("Collection Milvus vide")

        documents = [
            Document(page_content=d.page_content, metadata=d.metadata)
            for d in docs
        ]

        retriever_bm25 = BM25Retriever.from_documents(documents)
        retriever_bm25.k = 4

        return EnsembleRetriever(
            retrievers=[retriever_vectoriel, retriever_bm25],
            weights=[0.7, 0.3]
        )

    except Exception as e:
        print("❌ Erreur retriever :", e)
        return BM25Retriever.from_documents([
            Document(
                page_content="Aucune donnée financière disponible.",
                metadata={"source": "erreur"}
            )
        ])


# =========================
# AGENT FINANCIER OLLAMA
# =========================
def obtenir_llm_et_agent(retriever) -> AgentExecutor:
    """
    Agent financier utilisant Ollama en local
    """

    tool = create_retriever_tool(
        retriever,
        name="recherche_financiere",
        description="Recherche des informations financières dans la base de connaissances"
    )

    llm = ChatOllama(
        model="llama2",
        temperature=0,
        streaming=True
    )

    systeme = """
Tu es FinChatAI, un assistant financier professionnel.
Tu es expert en :
- banque
- finance
- économie
- investissements

Règles :
- Réponds UNIQUEMENT à partir des documents fournis
- Si l'information est absente, dis-le clairement
- Réponds en français
- Ne donne pas de conseils financiers personnalisés
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", systeme),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(
        llm=llm,
        tools=[tool],
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=[tool],
        verbose=True
    )


# =========================
# ALIAS POUR main.py
# =========================
get_retriever = obtenir_retriever
get_llm_and_agent = obtenir_llm_et_agent

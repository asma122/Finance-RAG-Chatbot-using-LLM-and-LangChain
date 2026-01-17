import os
import json
from uuid import uuid4
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus

# =========================
# CONFIGURATION GLOBALE
# =========================
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "finance_data"
FICHIER_JSON = "finance_minimal.json"
REPERTOIRE_DATA = "data"

# =========================
# CHARGEMENT DONNÉES LOCALES
# =========================
def charger_donnees_local(filename: str, repertoire: str):
    chemin_fichier = os.path.join(repertoire, filename)
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(f"Fichier introuvable : {chemin_fichier}")

    with open(chemin_fichier, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Données chargées depuis {chemin_fichier}")
    return data
# =========================
# SEED MILVUS (FICHIER LOCAL)
# =========================
from uuid import uuid4
from tqdm import tqdm
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus


def seed_milvus(host, port, collection_name, filename, repertoire, utiliser_ollama=True):
    print("📂 Chargement des données locales...")
    local_data = charger_donnees_local(filename, repertoire)

    if not local_data:
        print("⚠️ Aucun document trouvé dans le fichier")
        return

    embeddings = OllamaEmbeddings(model="llama2") if utiliser_ollama else None

    documents = []
    for i, doc in enumerate(local_data):
        page_content = (
            doc.get("page_content")
            or doc.get("content")
            or str(doc)
        )

        metadata = doc.get("metadata", {}) or {
            "source": doc.get("source", "local_file"),
            "content_type": "text/plain",
            "title": doc.get("title", f"document_{i}"),
            "description": doc.get("description", ""),
            "language": doc.get("language", "fr"),
            "doc_name": filename.rsplit(".", 1)[0],
            "start_index": 0,
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    # =========================
    # 🔥 LIMITATION ICI
    # =========================
    MAX_DOCS = 20
    documents = documents[:MAX_DOCS]

    print(f"📄 Nombre de documents à indexer (limité) : {len(documents)}")

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"host": host, "port": port},
        collection_name=collection_name,
        drop_old=True,
    )

    print("🧠 Génération des embeddings + insertion Milvus...")
    for doc in tqdm(documents):
        vectorstore.add_documents(
            documents=[doc],
            ids=[str(uuid4())]
        )

    print(f'✅ Vecteurs stockés dans la collection "{collection_name}"')
# =========================
# SEED MILVUS (CRAWL WEB)
# =========================
def seed_milvus_live(url, host, port, collection_name, nom_doc="document_web", utiliser_ollama=True):
    from crawl import crawl_web

    embeddings = OllamaEmbeddings(model="llama2") if utiliser_ollama else None
    documents = crawl_web(url)

    if not documents:
        raise ValueError(f"⚠️ Aucun document récupéré depuis l'URL : {url}")

    # Ajouter métadonnées
    for doc in documents:
        doc.metadata = {
            "source": url,
            "content_type": "text/html",
            "title": "",
            "description": "",
            "language": "fr",
            "doc_name": nom_doc,
            "start_index": 0,
        }

    # Limiter pour test rapide
    documents = documents[:5]
    ids = [str(uuid4()) for _ in documents]

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"host": host, "port": port},
        collection_name=collection_name,
        drop_old=True,
    )
    vectorstore.add_documents(documents=documents, ids=ids)
    print(f'✅ Données crawlé stockées dans la collection "{collection_name}"')
# =========================
# CONNEXION À MILVUS
# =========================
def connect_to_milvus(host, port, collection_name, utiliser_ollama=True):
    embeddings = OllamaEmbeddings(model="llama2") if utiliser_ollama else None

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"host": host, "port": port},
        collection_name=collection_name,
    )

    print(f"Connecté à la collection Milvus '{collection_name}'")
    return vectorstore


# =========================
# TEST TERMINAL
# =========================
if __name__ == "__main__":
    # Seed local
    seed_milvus(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        collection_name=COLLECTION_NAME,
        filename=FICHIER_JSON,
        repertoire=REPERTOIRE_DATA,
        utiliser_ollama=True,
    )

    # Test Crawl
    # seed_milvus_live(
    #     url="https://www.investopedia.com/",
    #     host=MILVUS_HOST,
    #     port=MILVUS_PORT,
    #     collection_name=COLLECTION_NAME,
    #     nom_doc="Investopedia",
    #     utiliser_ollama=True
    # )

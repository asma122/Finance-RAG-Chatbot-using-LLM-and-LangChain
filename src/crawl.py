import re
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extracteur_finance(html: str) -> str:
    """Nettoyage HTML : ne garder que les paragraphes"""
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = soup.find_all("p")
    texte = "\n".join(p.get_text() for p in paragraphs)
    return re.sub(r"\n\n+", "\n\n", texte).strip()

def crawl_web(url: str):
    """Crawl simple avec User-Agent pour éviter le blocage"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/118.0.0.0 Safari/537.36"
        }

        # Récupération du HTML
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ Impossible d'accéder à l'URL {url} (Status {response.status_code})")
            return []

        html = response.text
        texte = extracteur_finance(html)
        if not texte.strip():
            print(f"⚠️ Aucun texte trouvé sur {url}")
            return []

        # Créer un document LangChain
        doc = Document(page_content=texte, metadata={"source": url})

        # Découpage en chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = splitter.split_documents([doc])

        print(f"✅ {len(docs)} segments créés depuis {url}")
        return docs

    except Exception as e:
        print(f"❌ Erreur crawl_web : {e}")
        return []

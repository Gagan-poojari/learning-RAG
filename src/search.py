import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vector_store import FaissVectorStore

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemini-3-flash-preview"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        if not gemini_api_key:
            raise ValueError("[ERROR] GEMINI_API_KEY not found in environment. Check your .env file.")

        self.llm = ChatGoogleGenerativeAI(
            google_api_key=gemini_api_key,
            model=llm_model,
            temperature=0.2,
        )
        print(f"[INFO] Gemini LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)

        # Deduplicate by text content to avoid repeated identical chunks
        seen = set()
        unique_texts = []
        for r in results:
            text = r["metadata"].get("text", "") if r["metadata"] else ""
            if text and text not in seen:
                seen.add(text)
                unique_texts.append(text)

        context = "\n\n".join(unique_texts)
        if not context:
            return "No relevant documents found."

        prompt = f"""You are an intelligent AI assistant. Summarize the following context for the query: '{query}'

Context:
{context}

Summary:"""

        response = self.llm.invoke(prompt)
        return getattr(response, "text", response.content)


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
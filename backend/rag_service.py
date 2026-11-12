import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class RAGService:
    def __init__(self, persist_directory: str, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.init_error = None

        if not os.getenv("OPENAI_API_KEY"):
            self.init_error = "OPENAI_API_KEY is not configured"
            return

        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.vector_store = Chroma(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            )
            self.llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                                  temperature=0)
        except Exception as exc:
            self.init_error = str(exc)

        self.prompt = ChatPromptTemplate.from_template("""
You are a banking infrastructure incident analyst.
Use only the retrieved knowledge below. Separate observed facts from hypotheses.
Do not invent incident IDs, confirmed causes, or completed actions.
If the evidence is insufficient, say what must be checked.

INCIDENT:
{incident}

RETRIEVED KNOWLEDGE:
{context}

Return:
1. Incident summary
2. Potential root causes and why
3. Supporting evidence and source titles/incident IDs
4. Investigation steps
5. Immediate recommendations
6. Preventive recommendations
7. Unknowns requiring validation
""")

    def status(self):
        return {
            "available": self.retriever is not None and self.llm is not None,
            "error": self.init_error,
            "collection": self.collection_name,
            "persist_directory": self.persist_directory,
        }

    def analyze(self, incident: str):
        if self.retriever is None or self.llm is None:
            raise RuntimeError(self.init_error or "RAG service is not initialized")

        docs = self.retriever.invoke(incident)
        if not docs:
            context = "No relevant documents were retrieved."
        else:
            context = "\n\n".join(
                f"Source metadata: {doc.metadata}\nContent:\n{doc.page_content}"
                for doc in docs
            )

        messages = self.prompt.format_messages(
            incident=incident,
            context=context,
        )
        answer = self.llm.invoke(messages).content

        sources = [
            {
                "title": doc.metadata.get("title"),
                "incident_id": doc.metadata.get("incident_id"),
                "source": doc.metadata.get("source"),
                "severity": doc.metadata.get("severity"),
            }
            for doc in docs
        ]
        return {"answer": answer, "sources": sources}

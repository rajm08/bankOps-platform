
import os
from dotenv import load_dotenv

from typing import TypedDict

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END


# =====================================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not configured.")


# =====================================================
# STEP 2: CONNECT TO EXISTING CHROMADB KNOWLEDGE BASE
# =====================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="banking_incident_knowledge",
    persist_directory="./banking_knowledge_db",
    embedding_function=embeddings
)

print("Connected to ChromaDB knowledge base!")


# =====================================================
# STEP 3: CREATE RETRIEVER
# =====================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)


# =====================================================
# STEP 4: CONFIGURE LLM
# =====================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# =====================================================
# STEP 5: DEFINE LANGGRAPH STATE
# =====================================================

class IncidentState(TypedDict):

    question: str
    context: str
    answer: str
    sources: list


# =====================================================
# STEP 6: CREATE ROOT CAUSE ANALYSIS PROMPT
# =====================================================

root_cause_prompt = ChatPromptTemplate.from_template("""

You are an expert Banking Infrastructure Incident Analyst.

Analyze the incident using the retrieved historical incidents
and technical documentation.

NEW INCIDENT:
{incident}

RETRIEVED KNOWLEDGE:
{context}

Generate a detailed Root Cause Analysis Report with the
following sections:

1. INCIDENT SUMMARY
   - Summarize the incident.
   - Highlight abnormal infrastructure metrics.
   - Explain the potential impact on banking services.

2. POTENTIAL ROOT CAUSES
   - Identify possible root causes.
   - Explain why each cause could contribute to the incident.
   - Prioritize causes as High, Medium, or Low.
   - Do not claim a cause is confirmed without evidence.

3. SUPPORTING EVIDENCE
   - Reference relevant historical incidents.
   - Reference relevant technical documentation.
   - Explain the evidence supporting each potential cause.

4. RECOMMENDED INVESTIGATION STEPS
   - List practical troubleshooting activities.
   - Identify infrastructure metrics and logs to examine.
   - Suggest checks to confirm the root cause.

5. RECOMMENDED RESOLUTIONS
   - Provide immediate corrective actions.
   - Provide longer-term preventive measures.
   - Prioritize recommendations based on impact.

6. BUSINESS IMPACT
   - Explain potential effects on NEFT, IMPS, RTGS,
     authentication, and transaction processing where relevant.

7. CONCLUSION
   - Summarize the most likely explanation.
   - Clearly identify what still requires investigation.

IMPORTANT RULES:

- Use only the supplied retrieved information.
- Distinguish observed facts from possible explanations.
- Never invent historical incidents or incident IDs.
- Do not present potential causes as confirmed causes.
- If the retrieved information is insufficient, state that.
- Do not claim that corrective actions have been performed.

""")


# =====================================================
# STEP 7: FORMAT RETRIEVED DOCUMENTS
# =====================================================

def format_docs(docs):

    formatted_docs = []

    for doc in docs:

        metadata = doc.metadata

        source_name = metadata.get(
            "title",
            metadata.get("incident_id", "Unknown source")
        )

        formatted_docs.append(
            f"""
            Source: {source_name}

            Source Type: {metadata.get('source', 'Unknown')}

            Severity: {metadata.get('severity', 'N/A')}

            Content:
            {doc.page_content}
            """
        )

    return "\n\n".join(formatted_docs)


# =====================================================
# STEP 8: RETRIEVE RELEVANT KNOWLEDGE
# =====================================================

def retrieve_knowledge(state: IncidentState):

    question = state["question"]

    retrieved_docs = retriever.invoke(question)

    context = format_docs(retrieved_docs)

    return {
        "context": context,
        "sources": retrieved_docs
    }


# =====================================================
# STEP 9: GENERATE ROOT CAUSE ANALYSIS
# =====================================================

def generate_root_cause_report(state: IncidentState):

    messages = root_cause_prompt.format_messages(
        incident=state["question"],
        context=state["context"]
    )

    response = llm.invoke(messages)

    return {
        "answer": response.content
    }


# =====================================================
# STEP 10: BUILD LANGGRAPH WORKFLOW
# =====================================================

workflow = StateGraph(IncidentState)

workflow.add_node(
    "retrieve_knowledge",
    retrieve_knowledge
)

workflow.add_node(
    "root_cause_analysis",
    generate_root_cause_report
)

workflow.add_edge(
    START,
    "retrieve_knowledge"
)

workflow.add_edge(
    "retrieve_knowledge",
    "root_cause_analysis"
)

workflow.add_edge(
    "root_cause_analysis",
    END
)

app = workflow.compile()

print("Root Cause Analysis Assistant initialized!")


# =====================================================
# STEP 11: CREATE INCIDENT INVESTIGATION FUNCTION
# =====================================================

def investigate_incident(incident_description):

    result = app.invoke({
        "question": incident_description
    })

    print("\n========== ROOT CAUSE ANALYSIS REPORT ==========\n")

    print(result["answer"])

    print("\n========== SUPPORTING SOURCES ==========\n")

    for i, doc in enumerate(result["sources"], start=1):

        print(f"\nSource {i}")

        print(
            "Title:",
            doc.metadata.get("title", "N/A")
        )

        print(
            "Incident ID:",
            doc.metadata.get("incident_id", "N/A")
        )

        print(
            "Severity:",
            doc.metadata.get("severity", "N/A")
        )

        print(
            "Source Type:",
            doc.metadata.get("source", "N/A")
        )

        print("----------------------------------------")

    return result


# =====================================================
# STEP 12: TEST THE ROOT CAUSE ANALYSIS
# =====================================================

if __name__ == "__main__":

    new_incident = """

    Incident Type: Infrastructure Performance Degradation

    CPU Utilization: 92%
    Memory Usage: 88%
    Database Connection Pool Usage: 96%
    Network Throughput: 220 Mbps
    Application Error Rate: 8%

    NEFT Success Rate: 92%
    IMPS Success Rate: 94%
    RTGS Success Rate: 98%

    Average Transaction Processing Time: 1800 ms

    Incident Description:

    The banking application is experiencing slow transaction
    processing. CPU utilization and database connection pool
    usage have increased.

    NEFT and IMPS transaction success rates have decreased.

    Investigate the possible root causes and recommend
    corrective actions.

    """

    investigate_incident(new_incident)

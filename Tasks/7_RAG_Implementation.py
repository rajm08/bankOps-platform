import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Load OpenAI API key from environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not configured.")


# Initialize embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Connect to existing ChromaDB collection
vector_store = Chroma(
    collection_name="banking_incident_knowledge",
    persist_directory="./banking_knowledge_db",
    embedding_function=embeddings
)

print("Connected to ChromaDB successfully!")


retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


prompt = ChatPromptTemplate.from_template(
    """
        You are an experienced banking infrastructure incident analyst.

        Your task is to investigate a new incident using the
        historical incidents and technical documentation provided below.

        NEW INCIDENT:
        {incident}

        RETRIEVED KNOWLEDGE:
        {context}

        Prepare an incident investigation report using the following sections:

        1. Incident Summary
        Summarize the observed infrastructure metrics and symptoms.

        2. Potential Root Causes
        Identify possible causes based on the retrieved information.

        3. Supporting Evidence
        Explain which retrieved documents or historical incidents support
        each potential cause.

        4. Recommended Investigation Steps
        Provide practical steps to investigate the incident.

        5. Recommended Resolution
        Suggest corrective actions based on the documentation.

        6. Historical References
        Mention the incident IDs and documentation titles used.

        IMPORTANT RULES:
        - Use only the supplied retrieved information.
        - Distinguish potential causes from confirmed root causes.
        - Do not invent historical incidents, incident IDs, or resolutions.
        - If retrieved information is insufficient, explicitly mention it.
        - Do not claim that any recommended action has already been performed.
""")


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


def investigate_incident(incident_description):

    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.invoke(
        incident_description
    )

    # Step 2: Format retrieved documents
    context = format_docs(retrieved_docs)

    # Step 3: Prepare the prompt
    messages = prompt.format_messages(
        incident=incident_description,
        context=context
    )

    # Step 4: Generate response using LLM
    response = llm.invoke(messages)

    # Step 5: Display investigation report
    print("\n========== INCIDENT INVESTIGATION REPORT ==========\n")

    print(response.content)

    return response, retrieved_docs



new_incident = """

A banking application is experiencing slow transaction processing.

Current infrastructure metrics:

CPU utilization: 92%
Memory usage: 80%
Database connection pool usage: 95%
Application error rate: 8%
NEFT success rate: 92%

Investigate the possible causes and recommend corrective actions.

"""

response, retrieved_docs = investigate_incident(
    new_incident
)
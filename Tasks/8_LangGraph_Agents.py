import os
from dotenv import load_dotenv

from typing import TypedDict

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="banking_incident_knowledge",
    persist_directory="./banking_knowledge_db",
    embedding_function=embeddings
)

print("Connected to banking knowledge base!")

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

class AssistantState(TypedDict):

    question: str
    context: str
    answer: str
    sources: list


def format_docs(docs):

    formatted_docs = []

    for doc in docs:

        source = doc.metadata.get(
            "title",
            doc.metadata.get("incident_id", "Unknown source")
        )

        formatted_docs.append(
            f"""
            Source: {source}

            Metadata: {doc.metadata}

            Content:
            {doc.page_content}
            """
        )

    return "\n\n".join(formatted_docs)


prompt = ChatPromptTemplate.from_template("""

You are an AI-powered Banking Infrastructure Incident Assistant.

Answer the user's question using the retrieved historical incidents
and technical documentation.

USER QUESTION:
{question}

RETRIEVED KNOWLEDGE:
{context}

Instructions:

1. Provide clear and beginner-friendly explanations.
2. Identify potential root causes when relevant.
3. Recommend practical troubleshooting steps.
4. Suggest resolutions based on the retrieved documentation.
5. Reference relevant historical incidents and document titles.
6. Do not invent incident IDs, facts, or resolutions.
7. Distinguish possible causes from confirmed root causes.
8. If the knowledge base does not contain sufficient information,
   clearly mention that.

Provide a structured and helpful answer.

""")

def retrieve_knowledge(state: AssistantState):

    question = state["question"]

    # Retrieve relevant documents from ChromaDB
    retrieved_docs = retriever.invoke(question)

    # Convert documents into readable context
    context = format_docs(retrieved_docs)

    return {
        "context": context,
        "sources": retrieved_docs
    }


def generate_answer(state: AssistantState):

    # Prepare prompt with user question and retrieved context
    messages = prompt.format_messages(
        question=state["question"],
        context=state["context"]
    )

    # Generate response using LLM
    response = llm.invoke(messages)

    return {
        "answer": response.content
    }


# Initialize StateGraph
workflow = StateGraph(AssistantState)

# Register nodes
workflow.add_node("retrieve_knowledge", retrieve_knowledge)
workflow.add_node("generate_answer", generate_answer)

# Define workflow sequence
workflow.add_edge(START, "retrieve_knowledge")

workflow.add_edge("retrieve_knowledge", "generate_answer")

workflow.add_edge("generate_answer", END)

# Compile the graph
app = workflow.compile()

print("Banking AI Assistant created successfully!")



question = """
CPU utilization in our banking application has reached 92%.

What could be the possible reasons for this?
How can we investigate and resolve this issue?
"""

result = app.invoke({
    "question": question
})

print("\n========== BANKING AI ASSISTANT ==========\n")

print(result["answer"])


while True:

    question = input("\nAsk your banking assistant: ")

    if question.lower() in ["exit", "quit"]:
        print("Thank you for using Banking AI Assistant!")
        break

    # Execute LangGraph workflow
    result = app.invoke({
        "question": question
    })

    # Display answer
    print("\n========== AI ASSISTANT RESPONSE ==========\n")

    print(result["answer"])

    # Display sources
    print("\n========== SOURCES ==========\n")

    for doc in result["sources"]:

        print(
            doc.metadata.get(
                "title",
                doc.metadata.get("incident_id", "Unknown source")
            )
        )
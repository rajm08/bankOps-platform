import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path="/Users/raj/Desktop/CapStone/bankOps-platform/DataSets/incident_knowledge_base.txt",
        chunkin_strategy="basic",
        max_characters=1000000)
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings= OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Ingesting....")

    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name="banking_incident_knowledge",
        persist_directory="./banking_knowledge_db"
    )

    print("FINISH....")


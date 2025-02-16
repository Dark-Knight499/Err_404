import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Configure Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def create_rag_chain(pdf_path):
    """Create a RAG chain from PDF document"""
    # Load and split the document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    # Create LLM
    llm = GoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1
    )
    
    # Create and return the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
    )
    
    return qa_chain

def main():
    # Example usage
    pdf_path = "sample.pdf"  # Replace with your PDF path
    
    # Create RAG chain
    qa_chain = create_rag_chain(pdf_path)
    
    # Example query
    query = "What is the main topic of the document?"
    response = qa_chain.invoke({"query": query})
    print(f"Query: {query}\nResponse: {response['result']}")

if __name__ == "__main__":
    main()

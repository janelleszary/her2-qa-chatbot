import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "development"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging 
logging.getLogger("langchain").setLevel(logging.ERROR)

from rag_generator import get_qa_chain
from rag_retriever import get_vectorstore, ingest_doc, get_documents

def main():
    """
    Runs an interactive command-line QA chatbot using a RAG pipeline.
    
    Loads document embeddings, sets up a retriever and LLM-based generator,
    and answers user questions based only on the referenced HER2 publication.
    """
    
    print("\n== Clinical QA Chatbot (HER2 Publication) ==")
        
    embed_model_name="e5"
    top_k=3
    threshold=0.7

    print("Loading vectorstore...")
    
    # Import raw text
    text = ingest_doc()
    
    # Get LangChain documents (chunks)
    documents = get_documents(text, embed_model_name)

    # Get vectorstore
    vectorstore = get_vectorstore(embed_model_name, documents, collection_name="e5_embed_collection")

    # Get retriever
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": top_k, "score_threshold": threshold})

    print("Loading generator...")
    
    # Get qa_chain (generator)
    qa_chain = get_qa_chain(retriever)
    
    print("Ready! \n")
    print("Type your question below, or 'q' to quit.\n")

    while True:
        question = input("Q: ")
        if question.strip().lower() in ["q", "quit", "exit"]:
            print("Goodbye!")
            break
        try:
            answer = qa_chain(question)
            print("A:", answer["result"], "\n")
        except Exception as e:
            print("⚠️ Error generating answer:", e, "\n")

if __name__ == "__main__":
    main()

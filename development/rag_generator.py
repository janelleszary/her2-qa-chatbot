# Standard library
import contextlib
import io
import sys
from difflib import SequenceMatcher

# Third-party libraries
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import torch
from IPython.display import display, Markdown
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# LangChain and related libraries
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Local imports
from rag_retriever import get_eval_set

#################################################################
# Constants
#################################################################

PROMPT = """
[INST]
You are a clinical assistant answering questions based only on a specific scientific document. If you are not provided with a question related to the context from that document, respond by saying "Hm, I'm only able to answer questions related to the HER2 study. Please ask me specific questions about that document."

If the context does not contain relevant information, respond exactly with:
"The study does not provide that information."

Be concise. Do not make up answers. Do not guess based on prior knowledge. Only use the abstract, title, and authors for general orientation or if the question is general or about those things.

Title: Human Breast Cancer: Correlation of Relapse and Survival with Amplification of the HER-2/neu Oncogene.
Authors: Dennis J. Slamon, Gary M. Clark, Steven G. Wong, Wendy J. Levin, Axel Ullrich, William L. McGuire
Abstract: The HER-2/neu oncogene is a member of the erbB-like oncogene family, and is related to, but distinct from, the epidermal growth factor receptor. This gene has been shown to be amplified in human breast cancer cell lines. In the current study, alterations of the gene in 189 primary human breast cancers were investigated. HER-2/neu was found to be amplified from 2- to greater than 20-fold in 30% of the tumors. Correlation of gene amplification with several disease parameters was evaluated. Amplification of the HER-2/neu gene was a significant predictor of both overall survival and time to relapse in patients with breast cancer. It retained its significance even when adjustments were made for other known prognostic factors. Moreover, HER-2/neu amplification had greater prognostic value than most currently used prognostic factors, including hormonal-receptor status, in lymph node-positive disease. These data indicate that this gene may play a role in the biologic behavior and/or pathogenesis of human breast cancer.

Note: Figures and tables were not included in the ingested context.

Context:
{context}

Question:
{question}
[/INST]
"""

#################################################################
# Functions
#################################################################

def is_ollama_running() -> bool:
    """Check if the Ollama server is running locally and responding."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
        
def get_llm(model_name: str = "mistral"):
    """
    Load an Ollama-hosted LLM (e.g., Mistral) for use in a LangChain-compatible interface.

    Exits the script with a message if Ollama is not running.
    """
    
    if not is_ollama_running():
        print("\n⚠️  Ollama is not running.\n"
      "Please open a new terminal window and start it with:\n\n"
      "    ollama run mistral\n")
        sys.exit(1)
    
    print(f"Using Ollama model: {model_name_ollama}")
    return OllamaLLM(model=model_name_ollama, temperature=temperature, num_predict=max_new_tokens)

def get_qa_chain(retriever, llm=None, strict=False):
    """
    Build a custom QA chain that answers questions using retrieved document context.

    Args:
        retriever: LangChain-compatible retriever object.
        llm: Optional custom LLM (default loads from Ollama).
        strict: If True, suppress output for questions with no retrieved context.

    Returns:
        Function that accepts a question and returns a generated answer.
    """

    if llm is None: llm = get_llm()

    prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT)
    chain = prompt | llm

    def custom_qa(question, strict=strict):
        # Suppress stdout (ChromaRetriever uses `print`, not logging)
        with contextlib.redirect_stdout(io.StringIO()):
            docs = retriever.invoke(question) 

        if strict and not docs:
            return {"result": "Hm, I can't find anything in this document to answer your question. "
                          "Please make sure you're asking a question about this document, and I'll be happy to help!"}

        context = "\n\n".join(doc.page_content for doc in docs)
        result = chain.invoke({"context": context, "question": question})
        return {"result": result}

    return custom_qa

def evaluate_answer(pred, truth, embedding_model=None):
    """
    Compare a predicted answer to a ground truth using token F1, character similarity,
    and optional embedding-based cosine similarity.

    Args:
        pred: Predicted answer string.
        truth: Ground truth answer string.
        embedding_model: Optional SentenceTransformer for cosine similarity.

    Returns:
        Dict with token_f1, char_similarity, and cosine_similarity.
    """
    
    def normalize(text):
        return text.strip().lower()

    pred_norm = normalize(pred)
    truth_norm = normalize(truth)

    # Token-level F1
    pred_tokens = pred_norm.split()
    truth_tokens = truth_norm.split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        token_f1 = 0.0
    else:
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        token_f1 = 2 * precision * recall / (precision + recall)

    # Character-level similarity
    char_similarity = SequenceMatcher(None, pred_norm, truth_norm).ratio()

    if embedding_model:
        # Cosine similarity between sentence embeddings
        embedding_pred = embedding_model.encode(pred, convert_to_tensor=True)
        embedding_truth = embedding_model.encode(truth, convert_to_tensor=True)
        cosine_similarity = float(util.pytorch_cos_sim(embedding_pred, embedding_truth))
    else: cosine_similarity = -99

    return {
        "token_f1": token_f1,
        "char_similarity": char_similarity,
        "cosine_similarity": cosine_similarity
    }

def evaluate_generator(qa_chain, eval_set=None):
    """
    Run all questions in an evaluation set through the QA chain and score each answer.

    Args:
        qa_chain: Callable QA chain that returns {"result": answer}.
        eval_set: Optional list of QA pairs (dicts with 'question' and 'answer').

    Returns:
        List of dicts with question, generated answer, ground truth, and similarity scores.
    """
    
    if eval_set is None: eval_set = get_eval_set()

    # Lightweight embedder for cos similarity
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    results = []
    
    for item in tqdm(eval_set, desc="Evaluating QA chain"):
        question = item["question"]
        ground_truth = item["answer"]

        given_answer = qa_chain(question)["result"].strip()

        scores = evaluate_answer(given_answer, ground_truth, embedding_model)

        results.append({
            "question": question,
            "given_answer": given_answer,
            "correct_answer": ground_truth, 
            **scores
        })

    return results

def review_answers(df, mark_column="manual_label"):
    """
    Manually label each QA pair in a DataFrame as 'hit', 'miss', or 'partial'.

    Args:
        df: DataFrame with columns including 'question', 'given_answer', and 'correct_answer'.
        mark_column: Column to store manual labels.

    Returns:
        Updated DataFrame with manual labels.
    """

    print("Starting manual review...\n")
    for idx, row in df.iterrows():
        if pd.notnull(row.get(mark_column)):  # skip if already labeled
            continue
        print("\n")
        display(Markdown(f"**Question {idx+1}: {row['question']}**"))
        display(Markdown(f"**Correct Answer:** {row['correct_answer']}"))
        display(Markdown(f"**Given Answer:** {row['given_answer']}"))

        while True:
            decision = input("Mark as (h)it, (m)iss, (p)artial, (s)kip, (q)uit: ").strip().lower()
            if decision in ["h", "m", "p"]:
                label = {"h": "hit", "m": "miss", "p": "partial"}[decision]
                df.at[idx, mark_column] = label
                break
            elif decision == "s":
                break
            elif decision == "q":
                print("Stopping review.")
                return df
            else:
                print("Invalid input. Try h/m/p/s/q.")

    print("Manual review complete.")
    return df

def visualize_generator_metrics(df, title="Generator Evaluation Metrics"):
    """
    Visualize generator performance using manual labels and similarity metrics.

    Args:
        df: DataFrame containing generator outputs and evaluation scores.
        title: Title for the generated plot.

    Returns:
        None. Displays plots inline.
    """

    # Create 1x2 subplot grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set(style="whitegrid")

    # Plot 1: Manual label distribution
    sns.countplot(x="manual_label", data=df, ax=axes[0])
    axes[0].set_title("Fig 1A: Manual Evaluation Labels")
    axes[0].set_xlabel("Manual Label")
    axes[0].set_ylabel("Count")

    # Plot 2: Cosine, F1, and Char similarity across sorted answers
    sorted_df = df.sort_values("cosine_similarity").reset_index(drop=True)
    axes[1].plot(sorted_df.index, sorted_df["cosine_similarity"], marker="o", linestyle="-", label="Cosine Similarity")
    axes[1].plot(sorted_df.index, sorted_df["token_f1"], marker="o", linestyle="-", label="Token F1")
    axes[1].plot(sorted_df.index, sorted_df["char_similarity"], marker="o", linestyle="-", label="Char Similarity")
    axes[1].set_xlabel("Question Index (sorted by cosine similarity)")
    axes[1].set_title("Fig 1B: Generator Metric Trends")
    axes[1].legend()
    print(f"(Partial hit sorted index: {sorted_df[sorted_df.manual_label=="partial"].index[0]})") 

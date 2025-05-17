# Standard library
import json
import os

# Third-party libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# LangChain and Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb import Client
from chromadb.config import Settings

#################################################################
# Functions
#################################################################

def ingest_doc(path=None):
    """Load and return the clean transcript of the HER2 study from file."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "../Slamon_etal_1987_CleanTranscript.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_eval_set():
    """Load the evaluation dataset from a JSONL file."""
    with open("eval_dataset.jsonl", "r", encoding="utf-8") as f:
        eval_set = [json.loads(line) for line in f]
    return eval_set

def model_name_lookup(model_name="e5"):
    """Map a short model name to its full HuggingFace model identifier."""
    mapping = {
        "e5": "intfloat/e5-base-v2",
        "sbert": "sentence-transformers/all-MiniLM-L6-v2",
        "pubmedbert": "pritamdeka/S-PubMedBERT-MS-MARCO",
        "biobert-sim": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "biobert-raw": "dmis-lab/biobert-base-cased-v1.1"
    }
    return mapping.get(model_name)
    
def get_documents(text, model_name="e5", max_tokens=100, overlap=20, save_filename=None):
    """
    Split text into overlapping chunks using model-specific token limits,
    then wrap each chunk as a LangChain Document.

    Args:
        text: Full document text.
        model_name: Embedding model name (used to load appropriate tokenizer).
        max_tokens: Max tokens per chunk.
        overlap: Overlap between adjacent chunks.
        save_filename: Optional path to save chunks as CSV.

    Returns:
        List of LangChain Document objects.
    """
    def _get_chunks(paragraph, max_tokens=max_tokens, overlap=overlap, model_name=model_name):
    
        from transformers import AutoTokenizer, AutoModel
        from nltk.tokenize import sent_tokenize
        import nltk
        nltk.download('punkt', quiet=True)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_lookup(model_name))
        
        # Split text into sentences
        sentences = sent_tokenize(paragraph)
    
        # Group into overlapping chunks
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            token_count = len(tokenizer.tokenize(sentence))
            if current_tokens + token_count > max_tokens:
                chunks.append(" ".join(current_chunk))
                # Start new chunk with overlap
                overlap_tokens = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    overlap_tokens += len(tokenizer.tokenize(s))
                    overlap_chunk.insert(0, s)
                    if overlap_tokens >= overlap:
                        break
                current_chunk = overlap_chunk[:]
                current_tokens = sum(len(tokenizer.tokenize(s)) for s in current_chunk)
        
            current_chunk.append(sentence)
            current_tokens += token_count
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    
    # Split doc into paragraphs 
    paragraphs = text.split('\n\n')

    # Split paragraphs into overlapping chunks
    all_chunks = []
    for p in paragraphs:
        p_chunks = _get_chunks(p, max_tokens=max_tokens, overlap=overlap, model_name=model_name)
        all_chunks.extend(p_chunks)

    all_chunks = all_chunks[2:]

    if save_filename:
        name = model_name.split("/")[-1]
        with open(save_filename.split(".csv")[0] + f"-{name}.csv", "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(chunk + "\n\n")

    # Wrap in LangChain Documents
    documents = [Document(page_content=chunk) for chunk in all_chunks]

    return documents

def get_embedder(model_name):
    """Return an embedding wrapper appropriate for the specified model."""
    
    # models to try:
    # intfloat/e5-base-v2	Great general QA performance, trained for retrieval
    # sentence-transformers/all-MiniLM-L6-v2	Smaller, very fast, still solid
    # pritamdeka/S-PubMedBERT-MS-MARCO	Biomedical-specific, tuned for similarity tasks
    # pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb (BioBERT fine-tuned for similarity)
    # dmis-lab/biobert-base-cased-v1.1
    
    if model_name == "e5":
        return SentenceTransformerEmbedding(model_name=model_name_lookup(model_name))
    elif model_name == "sbert":
        return HuggingFaceEmbeddings(model_name=model_name_lookup(model_name), encode_kwargs={"normalize_embeddings": True})
    elif model_name == "pubmedbert":
        return HuggingFaceEmbeddings(model_name=model_name_lookup(model_name), encode_kwargs={"normalize_embeddings": True})
    elif model_name == "biobert-sim":
        return HuggingFaceEmbeddings(model_name=model_name_lookup(model_name), encode_kwargs={"normalize_embeddings": True})
    # Custom wrapper with mean pooling
    elif model_name == "biobert-raw": return BioBERTMeanEmbedding(model_name=model_name_lookup(model_name))
    else: raise ValueError(f"Unknown embedder: {model_name}")

def get_vectorstore(model_name, documents, collection_name="temp"):
    """
    Create and return a Chroma vectorstore using the specified embedding model.

    Args:
        model_name: Name of the embedding model to use.
        documents: List of LangChain Documents to index.
        collection_name: Chroma collection name.

    Returns:
        A LangChain-compatible Chroma vectorstore.
    """

    embedder = get_embedder(model_name)
    client = Client(Settings(anonymized_telemetry=False))
    
    # # Always delete a temp collection and start fresh
    # try: client.delete_collection(name="temp")
    # except: pass

    # Always start fresh
    try: client.delete_collection(name=collection_name)
    except: pass
        
    return Chroma.from_documents(documents, embedder, collection_name=collection_name, client=client)

def retrieve_context(query, vectorstore, top_k=3, threshold=None):
    """
    Retrieve top-k most similar chunks from the vectorstore using a similarity threshold.

    Args:
        query: Input question string.
        vectorstore: Chroma vectorstore object.
        top_k: Maximum number of chunks to return.
        threshold: Optional similarity threshold (0-1).

    Returns:
        List of filtered document chunks.
    """
    results = vectorstore.similarity_search_with_score(query, k=top_k * 3)
    filtered = []
    for doc, score in results:
        similarity = 1 - score
        if threshold is None or similarity >= threshold:
            doc.metadata["score"] = similarity
            filtered.append(doc)
        if len(filtered) == top_k:
            break
    return filtered
    
def evaluate_retriever(retriever, eval_set=None, top_k=3, threshold=0.8, corrected=False):
    """
    Evaluate retrieval quality using a labeled evaluation set.

    Args:
        retriever: LangChain retriever object.
        eval_set: Optional list of QA pairs with ground-truth context.
        top_k: Number of chunks to retrieve per query.
        threshold: Similarity threshold for retrieval.
        corrected: If True, adjusts recall to only count unique ground-truth hits.

    Returns:
        Tuple of (detailed results, summary metrics).
    """
    def is_match(retrieved, ground_truths):
        return any(gt in retrieved or retrieved in gt for gt in ground_truths)

    def match_gt_index(retrieved, ground_truths):
        for i, gt in enumerate(ground_truths):
            if gt in retrieved or retrieved in gt:
                return i
        return None

    results = []
    vectorstore = retriever.vectorstore

    total_recall = total_precision = total_f1 = total_hits = 0

    if eval_set is None: eval_set = get_eval_set()
    for item in eval_set:
        query = item["question"]
        ground_truth = item["ground_truth_context"]

        retrieved_docs = retrieve_context(query, vectorstore, top_k=top_k, threshold=threshold)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]

        # standard recall
        # this option counts how many retrieved chunks match something in the gt set (which is subsets of chunks)
        # bc of overlapping, some chunks match the same gt. this may inflate metrics
        if corrected:
            matched_gt_indices = set()
            for chunk in retrieved_texts:
                matched_idx = match_gt_index(chunk, ground_truth)
                if matched_idx is not None:
                    matched_gt_indices.add(matched_idx)
            recall = len(matched_gt_indices) / len(ground_truth) if ground_truth else 1
            precision = len(matched_gt_indices) / len(retrieved_texts) if retrieved_texts else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
            hit = len(matched_gt_indices) > 0
                    
        else: 
            match_count = sum(is_match(chunk, ground_truth) for chunk in retrieved_texts)
            recall = match_count / len(ground_truth) if ground_truth else 1
            precision = match_count / len(retrieved_texts) if retrieved_texts else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
            hit = match_count > 0

        total_recall += recall
        total_precision += precision
        total_f1 += f1
        total_hits += hit

        results.append({
            "question": query,
            "hit": hit,
            "recall@k": recall,
            "precision@k": precision,
            "f1": f1,
            "retrieved": retrieved_texts,
            "expected": ground_truth,
        })

    num_samples = len(eval_set)
    metrics = {
        "accuracy": total_hits / num_samples,
        "avg_recall@k": total_recall / num_samples,
        "avg_precision@k": total_precision / num_samples,
        "avg_f1": total_f1 / num_samples,
    }

    return results, metrics


def evaluation_loop(documents,
                    eval_set=None,
                    embedding_models=["e5", "sbert"],
                    thresholds=[None, 0.75, 0.8, 0.85],
                    ks=[1,2,3],
                    corrected=False):
    """
    Run evaluate_retriever across multiple embedding models, k values, and thresholds.

    Args:
        documents: List of LangChain Documents to index.
        eval_set: Optional evaluation dataset.
        embedding_models: List of embedding model keys to compare.
        thresholds: List of thresholds to test.
        ks: List of top_k values to test.
        corrected: Whether to use corrected recall scoring.

    Returns:
        A DataFrame of retrieval metrics.
    """
    
    if eval_set is None: eval_set = get_eval_set()
    
    results = []

    # for model_name in embedding_models:
    for model_name in tqdm(embedding_models, desc="Embedding Models", leave=True):

        # Create a fresh vectorstore and retriever 
        vectorstore = get_vectorstore(model_name, documents, collection_name="temp")        
        retriever = vectorstore.as_retriever(search_type="similarity")

        for k in ks:
            for t in thresholds:
                res, metrics = evaluate_retriever(retriever, eval_set, top_k=k, threshold=t, corrected=corrected)
                metrics["model"] = model_name
                metrics["top_k"] = k
                metrics["threshold"] = t if t is not None else "None"
                results.append(metrics)

            results.append(metrics)
    
    df = pd.DataFrame(results)
    
    return df

def visualize_retrieval_metrics(df, x="top_k", title=""):
    """
    Generate 2x2 plots showing F1, accuracy, recall, and precision by model and retrieval setting.

    Args:
        df: DataFrame with evaluation results.
        x: Variable to plot on the x-axis (e.g., "top_k" or "threshold").
        title: Optional plot title.
    """    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10)) 
    
    # Plot 1: F1 Score vs Threshold
    sns.lineplot(data=df, x=x, y="avg_f1", hue="model", marker="o", ax=axes[0, 0])
    axes[0, 0].set_title(f"Fig 1A: F1 vs {x} (No Threshold)")
    
    # Plot 2: F1 Score vs Top K (No Threshold)
    sns.lineplot(data=df, x=x, y="accuracy", hue="model", marker="o", ax=axes[0, 1])
    axes[0, 1].set_title(f"Fig 1B: Accuracy vs {x}")
    
    # Plot 3: Recall vs Top K (No Threshold)
    sns.lineplot(data=df, x=x, y="avg_recall@k", hue="model", marker="o", ax=axes[1, 0])
    axes[1, 0].set_title(f"Fig 1C: Recall vs {x}")
    
    # Plot 4: Precision vs Top K (No Threshold)
    sns.lineplot(data=df, x=x, y="avg_precision@k", hue="model", marker="o", ax=axes[1, 1])
    axes[1, 1].set_title(f"Fig 1D: Precision vs {x}")
    
    # Tight layout and display
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


#################################################################
# Classes
#################################################################

class SentenceTransformerEmbedding(Embeddings):
    """
    Embedding wrapper for E5 models with expected query/document prefixes.
    """
    def __init__(self, model_name="intfloat/e5-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # E5 expects prefix "passage: " for documents
        texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        # E5 expects prefix "query: " for queries
        return self.model.encode(f"query: {text}", convert_to_numpy=True).tolist()

class BioBERTMeanEmbedding(Embeddings):
    """
    Custom embedding wrapper for BioBERT using mean pooling over token embeddings.
    """
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, 1)
        counted = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = (summed / counted).squeeze()
        return mean_pooled.cpu().numpy()

    def embed_documents(self, texts):
        return [self._embed(t).tolist() for t in texts]

    def embed_query(self, text):
        return self._embed(text).tolist()

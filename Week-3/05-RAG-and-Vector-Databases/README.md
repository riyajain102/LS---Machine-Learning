# RAG and Vector Databases

[Home](../../README.md) > [Week 3](../README.md) > RAG & Vector Databases

| Week 3 · Topic 5 of 5 · Prerequisites: [Modern LLMs & Prompting](../04-Modern-LLMs-and-Prompting/README.md) |
|---|

---

## The Problem RAG Solves

LLMs have a knowledge cutoff — they know nothing about events after their training ended. They also know nothing about *your* documents: your company's internal wiki, a research paper published last week, a specific PDF you want to query.

You could fine-tune the model on your documents. But fine-tuning is expensive, slow, and the model can still hallucinate facts it "learned" during fine-tuning. And every time your documents update, you'd need to fine-tune again.

**Retrieval-Augmented Generation (RAG)** solves this differently: instead of baking knowledge into the weights, *retrieve* the relevant documents at inference time and inject them directly into the prompt. The model never needs to memorise your data — it reads the relevant pieces on demand.

---

## The RAG Pipeline

```
               ┌──────────────────────────────────────┐
               │         OFFLINE (done once)           │
               │                                      │
               │  Your Documents                      │
               │       │                              │
               │       ▼                              │
               │  Chunk into passages                 │
               │       │                              │
               │       ▼                              │
               │  Embed each chunk  ──►  Vector DB    │
               │  (embedding model)                   │
               └──────────────────────────────────────┘

               ┌──────────────────────────────────────┐
               │         ONLINE (every query)          │
               │                                      │
               │  User Question                       │
               │       │                              │
               │       ▼                              │
               │  Embed question  ──►  Vector DB      │
               │                           │          │
               │                    Nearest chunks    │
               │                           │          │
               │                           ▼          │
               │             Prompt = System +        │
               │                     Retrieved chunks │
               │                     + User Question  │
               │                           │          │
               │                           ▼          │
               │                          LLM         │
               │                           │          │
               │                           ▼          │
               │                        Answer        │
               └──────────────────────────────────────┘
```

---

## Step 1 — Chunking

You cannot pass an entire 200-page PDF into a prompt — context windows have limits, and long contexts dilute the signal. Instead, split documents into *chunks* of roughly 200–500 tokens, ideally at natural boundaries (paragraphs, sections).

**Chunking strategies:**
- **Fixed-size:** every chunk is exactly N tokens. Simple but can cut sentences mid-thought.
- **Sentence-based:** chunk by sentences or paragraphs. Preserves semantics.
- **Sliding window:** chunks overlap by K tokens. Prevents information loss at chunk boundaries.

---

## Step 2 — Embedding

Each chunk is converted to a dense vector using an **embedding model** — a Transformer encoder trained specifically to map semantically similar text to nearby vectors in the embedding space.

```python
from transformers import pipeline

embedder = pipeline(
    "feature-extraction",
    model="sentence-transformers/all-MiniLM-L6-v2",
    tokenizer="sentence-transformers/all-MiniLM-L6-v2"
)

def embed_text(text: str) -> list[float]:
    output = embedder(text, return_tensors="pt")
    # Mean-pool over token dimension to get a single vector
    import torch
    vec = torch.tensor(output[0]).mean(dim=0)
    return vec.tolist()

chunk = "The Transformer architecture was introduced in 2017 by Google Brain."
vector = embed_text(chunk)
print(f"Embedding dimension: {len(vector)}")   # 384 for MiniLM
```

This is the same dense vector idea from Week 2 (Word2Vec) — but now applied to *entire passages* instead of single words, and by a much more powerful model.

---

## Step 3 — Vector Database

A **vector database** stores embeddings and supports efficient *nearest-neighbour search* — given a query vector, find the K stored vectors that are most similar (by cosine similarity or dot product).

For the assignment we use **FAISS** (Facebook AI Similarity Search) — a library for fast vector search, no external service needed.

```python
import faiss
import numpy as np

EMBED_DIM = 384

# Build the index
index = faiss.IndexFlatIP(EMBED_DIM)   # IP = inner product (≈ cosine for normalised vectors)

# Add document chunk embeddings (shape: num_chunks x EMBED_DIM)
chunk_embeddings = np.array([embed_text(c) for c in chunks], dtype=np.float32)
faiss.normalize_L2(chunk_embeddings)   # normalise so IP = cosine similarity
index.add(chunk_embeddings)
print(f"Index contains {index.ntotal} vectors")

# Query: find the 3 most similar chunks to a question
query = "When was the Transformer introduced?"
query_vec = np.array([embed_text(query)], dtype=np.float32)
faiss.normalize_L2(query_vec)

scores, indices = index.search(query_vec, k=3)
for score, idx in zip(scores[0], indices[0]):
    print(f"[score: {score:.3f}] {chunks[idx][:120]}")
```

---

## Step 4 — Augmented Generation

The retrieved chunks become part of the prompt:

```python
def build_rag_prompt(question: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(
        f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)
    )
    return f"""You are a helpful assistant. Answer the question using ONLY the
provided context. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
```

This is the key insight: the LLM is not recalling facts from memory — it is *reading* the retrieved passages and synthesising an answer. This dramatically reduces hallucination because the model is grounded in actual retrieved text.

---

## Why Not Just Use a Bigger Context Window?

Modern models (Gemini 1.5, Claude 3) have context windows of 100k–1M tokens. Could you just dump all your documents in there and skip RAG?

Sometimes yes, but:
- **Cost:** every token in the context costs money per API call
- **Latency:** larger contexts = slower inference
- **Attention dilution:** models struggle to attend to relevant information when buried in a sea of irrelevant text (the "lost in the middle" problem)
- **Dynamic data:** if your data updates frequently, retrieval always reflects the latest version without re-running anything

RAG is still the production-standard approach for knowledge-intensive applications.

---

## The Complete RAG System (Conceptual)

```python
class SimpleRAG:
    def __init__(self, documents: list[str]) -> None:
        self.chunks = self._chunk(documents)
        self.index, self.embeddings = self._build_index(self.chunks)

    def _chunk(self, docs: list[str], chunk_size: int = 300) -> list[str]:
        # Split each document into chunks of ~chunk_size characters
        chunks = []
        for doc in docs:
            for i in range(0, len(doc), chunk_size):
                chunks.append(doc[i : i + chunk_size])
        return chunks

    def _build_index(self, chunks):
        # Embed all chunks and build FAISS index
        ...

    def retrieve(self, question: str, k: int = 3) -> list[str]:
        # Embed question, search index, return top-k chunks
        ...

    def answer(self, question: str) -> str:
        context_chunks = self.retrieve(question)
        prompt = build_rag_prompt(question, context_chunks)
        # Pass to LLM pipeline and return response
        ...
```

You will implement this in the assignment.

---

## Resources

**Read:**
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — Lewis et al., 2020](https://arxiv.org/abs/2005.11401) — The original RAG paper. Abstract and Section 2 give you the full picture.
- [FAISS Documentation](https://faiss.ai/) — Reference for the library you'll use in the assignment.
- [Sentence Transformers](https://www.sbert.net/) — The embedding models used for RAG. Browse the pretrained models page.

---

## Before You Move On

- Why does RAG reduce hallucination compared to a vanilla LLM?
- What is the role of the embedding model in the RAG pipeline?
- Why do we chunk documents instead of embedding them whole?
- If your company's internal knowledge base updates daily, why is RAG a better fit than fine-tuning?

---

[← Modern LLMs & Prompting](../04-Modern-LLMs-and-Prompting/README.md) | [Week 3 Overview](../README.md) | [Assignment →](../Week_3_Assignment.md)

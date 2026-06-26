# The Transformer Architecture

[Home](../../README.md) > [Week 3](../README.md) > Transformer Architecture

| Week 3 · Topic 1 of 5 · Prerequisites: [Embeddings & Word2Vec](../../Week-2/03-Embeddings-and-Word2Vec/README.md) |
|---|

---

## Why This Topic

In Week 2 you learned that Word2Vec gives every word one fixed vector regardless of context. "Bank" always had the same representation whether you were talking about a river bank or a financial bank. That was the fundamental ceiling of pre-2017 NLP.

The Transformer breaks through that ceiling. It computes a *different* representation for every word *depending on every other word in the sentence* — simultaneously, in parallel. This single idea is responsible for every major AI system you use today.

> 💡 This is the most technically dense topic of the entire course. Budget time for it. Read slowly. The payoff is that once you understand this, every modern model — BERT, GPT, T5, DALL-E, Whisper — becomes readable.

---

## What You Will Learn

- The **problem with RNNs** that Transformers solved
- **Self-Attention** — the core mechanism, and the intuition behind it
- **Query, Key, Value** — what these matrices actually mean
- **Scaled Dot-Product Attention** — the formula and why each piece exists
- **Multi-Head Attention** — attending to multiple relationship types simultaneously
- **Positional Encoding** — how the model knows word order without recurrence
- **The full Transformer block** — putting it all together

---

## The Problem Transformers Solved

Before Transformers, sequence models (RNNs, LSTMs) processed text *word by word*, left to right. To understand the word at position 50, the model had to pass information through 49 intermediate steps — and information from early in the sequence would get diluted or forgotten by the time it was needed.

```
"The trophy didn't fit in the suitcase because it was too big."
                                                    ↑
                          What does "it" refer to? (trophy or suitcase?)
```

An RNN has to carry the meaning of "trophy" through every word between "trophy" and "it". For long sentences, this is where they break.

**The Transformer's solution:** look at all words *simultaneously* and compute relationships between *any* two words directly, regardless of how far apart they are. No recurrence. Pure attention.

---

## Self-Attention — The Core Idea

The intuition: for each word in a sentence, self-attention asks *"which other words in this sentence are most relevant to understanding me?"* and creates a weighted mixture of those words' representations.

For the word "it" in the sentence above, a well-trained attention mechanism learns to assign high weight to "trophy" and low weight to "suitcase" — and the resulting representation of "it" will be enriched with information about trophies.

---

## Query, Key, Value

Self-attention is implemented using three learned linear projections of the input embeddings: **Query (Q)**, **Key (K)**, and **Value (V)**.

Think of it like a soft search engine:
- **Query** = what this word is *looking for* ("what is 'it' referring to?")
- **Key** = what each word *advertises about itself* ("I am a trophy", "I am a suitcase")
- **Value** = the actual *content* each word contributes if selected

For an input sequence of $n$ tokens, each with embedding dimension $d_{model}$:

$$X \in \mathbb{R}^{n \times d_{model}}$$

We project with three learned weight matrices:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d_{model} \times d_k}$ and $d_k$ is the dimension of each head.

---

## Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Let's unpack each part:

**Step 1 — Dot products:** $Q K^T \in \mathbb{R}^{n \times n}$

Each entry $(i, j)$ in this matrix is the dot product between the query of token $i$ and the key of token $j$. High dot product = token $i$ finds token $j$ highly relevant.

**Step 2 — Scaling by $\frac{1}{\sqrt{d_k}}$**

Without this, dot products grow large as $d_k$ increases (their variance scales with $d_k$). Large values push the softmax into regions of near-zero gradient — the vanishing gradient problem again. Dividing by $\sqrt{d_k}$ keeps the values in a stable range.

**Step 3 — Softmax:** converts raw scores into a probability distribution over positions. Each row sums to 1.

$$\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{m=1}^{n} \exp(q_i \cdot k_m / \sqrt{d_k})}$$

$\alpha_{ij}$ is the *attention weight* — how much token $i$ should attend to token $j$.

**Step 4 — Weighted sum of Values:** $\text{softmax}(\cdot) \cdot V$

The output for each position is a weighted sum of all value vectors. Positions with high attention weight contribute more to the output representation.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Args:
        Q: (batch, heads, seq_len, d_k)
        K: (batch, heads, seq_len, d_k)
        V: (batch, heads, seq_len, d_k)
        mask: optional (batch, 1, seq_len, seq_len) — True where attention is forbidden
    Returns:
        (batch, heads, seq_len, d_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, H, T, T)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)   # attention weights, sum to 1 across last dim
    return torch.matmul(weights, V)        # (B, H, T, d_k)
```

---

## Multi-Head Attention

Running attention once lets the model discover one type of relationship between tokens. **Multi-head attention** runs $h$ attention operations *in parallel*, each with its own learned $W^Q$, $W^K$, $W^V$ projections.

Why? Different heads can specialise. In a coreference task, one head might learn to track pronoun-noun relationships, another might learn syntactic dependencies, another might learn semantic similarity. None of this is hand-coded — it emerges from training.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W^O$$

$$\text{where} \quad \text{head}_i = \text{Attention}(Q W^Q_i,\; K W^K_i,\; V W^V_i)$$

The concatenated output is projected back to $d_{model}$ via $W^O \in \mathbb{R}^{hd_k \times d_{model}}$.

Typically: $d_{model} = 512$, $h = 8$, so $d_k = 64$ per head. Total compute is similar to one large attention, but expressiveness is much higher.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h   = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Project and split into heads: (B, T, d_model) -> (B, H, T, d_k)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.h, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(x))
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        attn = scaled_dot_product_attention(Q, K, V, mask)  # (B, H, T, d_k)

        # Concatenate heads and project: (B, T, d_model)
        out = attn.transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)
        return self.W_o(out)
```

---

## Positional Encoding

Self-attention has no notion of order — if you shuffle all the words in a sentence, the attention scores change (because different words are now at each position), but the mechanism itself is *permutation-equivariant*. The model needs to be told explicitly where each token sits in the sequence.

The original Transformer uses fixed sinusoidal encodings added to the input embeddings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

where $pos$ is the position index and $i$ is the dimension index.

**Why sinusoids?**
- They're bounded (no exploding values)
- Different frequencies encode different scales (nearby vs. distant positions)
- The model can learn to attend by relative position, because $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$
- They generalise to sequence lengths unseen during training

Modern models (BERT, GPT) use *learned* positional embeddings instead — a simple lookup table that is trained alongside the model. Even more modern models (RoPE, ALiBi) encode relative rather than absolute position. But sinusoids remain the foundation.

```python
import torch
import math

def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Returns positional encoding matrix of shape (seq_len, d_model)."""
    pe  = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len).unsqueeze(1).float()          # (T, 1)
    div = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )                                                              # (d_model/2,)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe   # (T, d_model)
```

---

## The Full Transformer Block

Each Transformer *layer* (the original paper stacks 6 of them for the encoder and 6 for the decoder) contains:

```
Input
  │
  ├─► LayerNorm ─► Multi-Head Self-Attention ─► + (residual) ─► LayerNorm
  │                                                                   │
  └───────────────────────────────────────────────────────────────────┘
                                                                       │
  ┌────────────────────────────────────────────────────────────────────┘
  │
  ├─► LayerNorm ─► Feed-Forward Network ─► + (residual) ─► LayerNorm
  │                (Linear → ReLU → Linear)
  └─────────────────────────────────────────────────────────────────────► Output
```

**Residual connections** (the `+` shortcuts) are critical. They let gradients flow directly from deep layers back to early layers, making training stable. Without them, a 12-layer Transformer would be nearly untrainable.

**Layer Norm** normalises the activations within each token's representation (across the feature dimension, not the batch). This stabilises training and is more suited to sequential data than Batch Norm.

**Feed-Forward Network (FFN):** applied independently to each position. Acts as a position-wise memory bank — research suggests this is where much of the model's factual knowledge is stored.

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

Typically $d_{ff} = 4 \times d_{model} = 2048$.

---

## Resources

**Watch (in this order — the best visual explanations ever made):**
- [Attention is All You Need — Yannic Kilcher](https://www.youtube.com/watch?v=iDulhoQ2pro) — Paper walkthrough by one of the best ML explainers. Dense but worth every minute.
- [The Illustrated Transformer — Jay Alammar (video)](https://www.youtube.com/watch?v=4Bdc55j80l8) — Visual and accessible. Great complement.
- [Let's build GPT from scratch — Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 2 hours. Implements a Transformer live. **The single best resource for making this concrete.**

**Read:**
- [The Illustrated Transformer — Jay Alammar (blog)](https://jalammar.github.io/illustrated-transformer/) — The most-read Transformer explainer on the internet. Bookmark it.
- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) — The original paper. Read the Abstract, Introduction, and Section 3. The rest is for when you're ready.

---

## Before You Move On

- Can you explain what Q, K, and V represent in your own words — without using the words "query", "key", or "value"?
- Why do we divide by $\sqrt{d_k}$? What goes wrong if we don't?
- What does one "head" in multi-head attention learn to do? Why do we need multiple heads?
- How does the model know the order of words if self-attention is permutation-equivariant?
- What do the residual connections do, and why would training fail without them?

---

[Week 3 Overview](../README.md) | [Next: Encoder vs Decoder →](../02-Encoder-vs-Decoder/README.md)

# Embeddings & Word2Vec

[Home](../../README.md) > [Week 2](../README.md) > Embeddings & Word2Vec

| Week 2 · Topic 3 of 3 · Prerequisites: [Introduction to Deep Learning](../02-Deep-Learning-Fundamentals/README.md) |
|---|

---

## Why This Topic

This is the topic that changed NLP forever.

Before embeddings, words were symbols — arbitrary tokens with no relationship to each other. "King" and "Queen" were as similar as "King" and "Potato". TF-IDF gave you a bag of unrelated numbers. Neural networks operating on those numbers had to learn relationships from scratch, every single time, for every single task.

Word2Vec (2013, Google) introduced the idea that words can be represented as **dense vectors in a continuous space** — and that the *geometry* of that space encodes meaning. Words with similar meanings cluster together. Relationships become arithmetic:

```
king - man + woman ≈ queen
paris - france + italy ≈ rome
```

This insight — that meaning can be represented as position in a learned vector space — is the foundation of every modern NLP model, including the transformers that power ChatGPT and BERT.

---

## What You Will Learn

- **Bigrams** — moving beyond single words; the beginning of capturing context
- **One-Hot Encoding** — the naive vector representation of words, and exactly why it fails
- **Dense Vector Embeddings** — the solution; low-dimensional, continuous, meaningful representations
- **Word2Vec** — how to learn embeddings from raw text using neural networks (Skip-gram & CBOW)

---

## Why It Matters

Every modern NLP system begins with an embedding layer. Whether it is Word2Vec, GloVe, FastText, or the learned embeddings inside BERT and GPT — the core idea is the same: map words to dense vectors where geometry = meaning.

Understanding this is non-negotiable for understanding transformers, attention, and large language models in Week 3 and 4.

---

## Concepts in Depth

---

### 📝 Bigrams — A Step Toward Context

A **unigram** model looks at words one at a time: `["I", "love", "this"]`

A **bigram** model looks at pairs of consecutive words: `["I love", "love this"]`

This captures some local context. "New York" as a bigram is meaningful; "New" and "York" separately are not. Bigrams (and n-grams generally) were a classical NLP technique used to give BoW models a sense of word order.

**Why bigrams aren't enough:**
- The vocabulary of bigrams is `V²` — if you have 50,000 words, you have up to 2.5 billion possible bigrams
- Still sparse, still symbolic — no notion of meaning
- Can't capture long-range dependencies ("The food, which I had been waiting for since morning, was disappointing" — the sentiment links words far apart)

Bigrams are a patch on a broken system. The real solution is embeddings.

---

### 🔢 One-Hot Encoding

The most straightforward way to represent a word as a vector: create a vector of length `|V|` (vocabulary size), put a `1` in the position corresponding to the word, and `0` everywhere else.

**Example with vocabulary:** `["cat", "dog", "fish", "bird"]` (|V| = 4)

```
"cat"  → [1, 0, 0, 0]
"dog"  → [0, 1, 0, 0]
"fish" → [0, 0, 1, 0]
"bird" → [0, 0, 0, 1]
```

Simple. Unambiguous. But fundamentally broken for representing language.

---

#### ❌ Why One-Hot Encoding Fails

**Problem 1: No semantic similarity**
```
cosine_similarity("cat", "dog")  = 0.0
cosine_similarity("cat", "fish") = 0.0
cosine_similarity("cat", "car")  = 0.0
```
Every word is equally dissimilar to every other word. The model has no way of knowing that "cat" and "dog" are both animals, that they're both pets, that they often appear in similar contexts.

**Problem 2: Dimensionality is huge and sparse**
A real vocabulary has 50,000–100,000 words. Every one-hot vector is 50,000 dimensions, with exactly one non-zero entry. You're storing almost no information at enormous cost.

**Problem 3: No generalisation across similar words**
If the model learns something about "happy", it cannot transfer that knowledge to "joyful". They are represented by completely orthogonal vectors. Every word has to be learned from scratch.

**Problem 4: Out-of-vocabulary words**
A word not in the training vocabulary has no representation at all.

---

### 🌌 Dense Vector Embeddings — The Solution

Instead of a sparse vector of size 50,000 with one `1`, represent each word as a **dense vector of size 100–300**, where every dimension has a meaningful value.

```
One-hot:   "cat" → [0, 0, 1, 0, 0, 0, ..., 0]  ← size 50,000, one non-zero
Embedding: "cat" → [0.23, -0.71, 0.44, 0.12, ..., -0.08]  ← size 300, all non-zero
```

In a good embedding space:
- Similar words are **close together** (small distance)
- Dissimilar words are **far apart**
- Relationships are encoded as **directions** in space

```
Vector("King")   - Vector("Man")  + Vector("Woman")  ≈  Vector("Queen")
Vector("Paris")  - Vector("France") + Vector("Italy") ≈  Vector("Rome")
```

This is not magic — it's a consequence of learning from the statistical patterns of how words appear together in text. Words that appear in similar contexts end up in similar regions of the space.

---

### 🧬 Word2Vec

Word2Vec (Mikolov et al., Google, 2013) is the algorithm that made dense word embeddings practical and famous. It trains a shallow neural network on a simple self-supervised task: **predict context from a word, or predict a word from its context**. The actual prediction is discarded; the learned weights become the embeddings.

There are two architectures:

---

#### CBOW — Continuous Bag of Words

**Task:** Given the surrounding context words, predict the centre word.

```
Context:  ["The", "cat", ___, "on", "the", "mat"]
Target:   "sat"
```

The model takes the average of the context word embeddings and tries to predict the missing word. CBOW is faster and works well for frequent words.

---

#### Skip-gram

**Task:** Given a single centre word, predict each of its surrounding context words.

```
Centre:   "sat"
Predict:  ["The", "cat", "on", "the", "mat"]
```

Skip-gram is slower but performs better on rare words and smaller datasets. It is the more commonly used Word2Vec variant.

---

#### How Training Works

```
1. Slide a context window (size W) over the entire corpus
2. For each (centre, context) pair:
   a. Look up the embedding of the centre word
   b. Run it through a shallow network
   c. Try to predict which words appeared in the context
   d. Compare prediction to reality → compute loss
   e. Backpropagate → update the embedding vectors
3. After training on millions of sentences, the embeddings encode
   the statistical context patterns of each word
```

The network never explicitly learns "cat and dog are both animals". It implicitly learns this because "cat" and "dog" appear next to similar words ("my ___ is cute", "___ food", "I have a ___"). Similar contexts → similar embeddings.

---

#### Negative Sampling

Training Word2Vec naively requires computing a softmax over the entire vocabulary for every training step — extremely expensive for 50,000+ words. **Negative sampling** is the trick that makes it fast:

Instead of predicting the full distribution, for each real (centre, context) pair, sample a few random "negative" words that didn't appear in the context. Train the network to distinguish real context words from random ones. Much faster, same quality.

---

#### What Word2Vec Embeddings Capture

| Relationship | Example |
|---|---|
| Synonyms | "happy" ≈ "joyful" |
| Antonyms | close in space but different direction |
| Is-a | "cat" is closer to "animal" |
| Country-capital | "France" → "Paris" :: "Germany" → "Berlin" |
| Gender | "king" → "queen" :: "actor" → "actress" |
| Tense | "run" → "ran" :: "walk" → "walked" |

---

#### Limitations of Word2Vec (Leading into Week 3)

Word2Vec gives every word **one fixed vector**, regardless of context.

```
"I went to the bank to deposit money."    → "bank" = [0.2, -0.5, ...]
"I sat by the river bank."                → "bank" = [0.2, -0.5, ...]
```

Same vector. But the word means something completely different. This is the **polysemy problem**.

The solution is **contextual embeddings** — the representation of a word depends on the sentence it appears in. This is exactly what BERT, GPT, and all transformer models compute. Transformers are the natural next step from Word2Vec.

---

## Resources

**Watch:**
- [Word2Vec, Clearly Explained — StatQuest](https://www.youtube.com/watch?v=viZrOnJclY0) — Best visual intro to Word2Vec. Start here.
- [Word Embeddings — 3Blue1Brown / Anthropic](https://www.youtube.com/watch?v=wjZofJX0v4M) — The visual intuition of what embedding spaces represent
- [Illustrated Word2Vec — Jay Alammar](https://jalammar.github.io/illustrated-word2vec/) — The most complete visual walkthrough of Skip-gram and CBOW

**Read:**
- [The Original Word2Vec Paper — Mikolov et al., 2013](https://arxiv.org/abs/1301.3781) — You don't need to understand all the math, but skim the abstract and introduction to appreciate the insight
- [Jay Alammar: The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — Read the article version too; it goes deeper than the video

**Interactive:**
- [Embedding Projector — Google](https://projector.tensorflow.org/) — Explore actual Word2Vec/GloVe embeddings in 3D. Type a word and see its nearest neighbours. **5 minutes here is worth an hour of reading.**

**Deep Dive:**
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) — An alternative to Word2Vec with slightly different training. Widely used as a baseline.
- [fastText — Facebook AI](https://fasttext.cc/) — Word2Vec at the character/subword level. Handles out-of-vocabulary words elegantly.

---

## Before You Move On

- Can you explain in your own words why one-hot encoding is a bad representation for words?
- Can you describe the training task of Word2Vec Skip-gram? What is the input, what is the target, what is being learned?
- Can you explain why words with similar contexts end up with similar embedding vectors?
- Can you give an example of a relationship that Word2Vec embeddings capture as arithmetic on vectors?
- Can you explain the polysemy problem and why it limits Word2Vec?

---

[Previous: Introduction to Deep Learning](../02-Deep-Learning-Fundamentals/README.md) | [Week 2 Overview](../README.md) | [Next: Week 2 Assignment](../Week_2_Assignment.md)

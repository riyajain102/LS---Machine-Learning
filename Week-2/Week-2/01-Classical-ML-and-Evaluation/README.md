# Classical ML & Evaluation

[Home](../../README.md) > [Week 2](../README.md) > Classical ML & Evaluation

| Week 2 · Topic 1 of 3 · Prerequisites: [Data Pre-processing & NLP](../../Week-1/04-Data-Preprocessing-NLP/README.md) |
|---|

---

## Why This Topic

Before neural networks took over, NLP was built on hand-crafted features and classical statistical models. You need to understand this era — not to use these tools as your primary approach, but because **(a)** they still appear in industry baselines, **(b)** they reveal exactly *why* deep learning was needed, and **(c)** the evaluation metrics introduced here (F1, confusion matrix, loss) are universal — you will use them to evaluate every model you ever build, including transformers.

> ⚠️ **Priority note:** This topic is **lower priority** compared to Topics 2 and 3. Read it for intuition and context. Do not spend more than 2–3 hours here.

---

## What You Will Learn

**Classical Text Representations:**
- **Bag-of-Words (BoW)** — representing text as word count vectors, ignoring order entirely
- **TF-IDF** — weighting words by how informative they are across a corpus
- **Logistic Regression** — using these vectors to classify sentiment; how it works at a high level

**Why These Approaches Fall Short (The Key Insight):**
- BoW loses all word order ("dog bites man" = "man bites dog")
- TF-IDF still treats every word as an isolated, unrelated symbol
- No understanding of synonyms, context, or meaning
- Dimensionality explodes with vocabulary size (sparse, high-dimensional vectors)
- Logistic Regression is a linear model — it cannot capture non-linear patterns in language

**Evaluation Metrics (Universal — memorise these):**
- **Loss** — the number the model minimises during training (cross-entropy for classification)
- **Accuracy** — fraction of correct predictions; misleading on imbalanced datasets
- **Precision, Recall, F1 Score** — the trio that gives a complete picture of classification quality
- **Confusion Matrix** — the ground truth table that shows exactly where a model gets confused

---

## Why It Matters

Understanding the *failures* of classical ML is what makes deep learning intuitive. Every limitation listed above has a direct solution in modern architectures:

| Classical Limitation | Deep Learning Solution |
|---|---|
| BoW loses word order | RNNs, Transformers model sequence |
| Words are isolated symbols | Embeddings (Word2Vec, BERT) encode meaning |
| Linear decision boundary | Multi-layer networks learn non-linear functions |
| Sparse, high-dimensional input | Dense, low-dimensional learned representations |

The evaluation metrics, however, are timeless. Accuracy, F1, and confusion matrices are how you measure *any* model.

---

## Concepts in Depth

### 📦 Bag-of-Words (BoW)

You have a vocabulary of `V` unique words. Every sentence becomes a vector of length `V`, where each position counts how many times that word appears.

```
Vocabulary: ["I", "love", "hate", "this", "movie"]

"I love this movie"  → [1, 1, 0, 1, 1]
"I hate this movie"  → [1, 0, 1, 1, 1]
```

These two sentences have almost identical vectors. The model *can* distinguish them, but notice:
- `"movie this love I"` gives the **exact same vector** — word order is completely gone
- Vocabulary of 50,000 words → vectors of size 50,000, mostly zeros (sparse)

**Disadvantages:**
- No word order → no grammar, no syntactic meaning
- No semantic meaning → "good" and "great" are as unrelated as "good" and "table"
- Extremely high-dimensional and sparse
- Common words ("the", "a", "is") dominate and add noise

---

### 📊 TF-IDF (Term Frequency – Inverse Document Frequency)

An improvement over raw counts. Instead of just counting, it *weights* words by how informative they are.

```
TF(word, doc)  = (count of word in doc) / (total words in doc)
IDF(word)      = log(total docs / docs containing word)
TF-IDF         = TF × IDF
```

Words like "the" appear in every document → low IDF → low weight.  
Words like "transformer" appear in few documents → high IDF → high weight.

**Disadvantages:**
- Still a bag (no order, no meaning)
- Still sparse and high-dimensional
- Cannot generalise: a word unseen in training has no TF-IDF score (out-of-vocabulary problem)

---

### 📈 Logistic Regression for Sentiment

Take a TF-IDF vector → pass it through a single linear layer → sigmoid/softmax → class probabilities.

It works surprisingly well for simple sentiment tasks because positive/negative reviews often differ in specific, frequent words ("great", "terrible"). But:

**Disadvantages:**
- Linear decision boundary only — cannot learn that "not bad" is positive
- Completely dependent on feature quality (garbage in, garbage out)
- Cannot generalise to unseen word combinations
- Falls apart on nuanced, sarcastic, or context-dependent language

---

### 📐 Evaluation Metrics

#### Confusion Matrix
The foundation of all classification evaluation:

```
                   Predicted Positive    Predicted Negative
Actual Positive  |  True Positive (TP)  | False Negative (FN) |
Actual Negative  |  False Positive (FP) | True Negative (TN)  |
```

- **TP** — correctly predicted positive
- **TN** — correctly predicted negative
- **FP** — predicted positive, was actually negative (Type I error)
- **FN** — predicted negative, was actually positive (Type II error)

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
**Problem:** If 95% of your dataset is class A, a model that always predicts A gets 95% accuracy — while being completely useless.

#### Precision & Recall
```
Precision = TP / (TP + FP)   ← Of all positives I predicted, how many were right?
Recall    = TP / (TP + FN)   ← Of all actual positives, how many did I find?
```

- High Precision = when you say "positive", you're usually right (few false alarms)
- High Recall = you find most of the actual positives (few misses)
- There's always a tradeoff — increasing one often decreases the other

#### F1 Score
The harmonic mean of Precision and Recall. Balances both:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Use F1 when you care equally about false positives and false negatives, and especially on **imbalanced datasets**.

#### Loss (Cross-Entropy)
For classification, the model minimises binary cross-entropy:
```
Loss = -[y·log(p) + (1-y)·log(1-p)]
```
Where `y` is the true label (0 or 1) and `p` is the predicted probability. Loss measures *how confident and correct* the model is — not just whether it got the answer right.

---

## Resources

**Watch:**
- [Bag of Words, Clearly Explained — StatQuest](https://www.youtube.com/watch?v=UFtXy0KRxVI) — Clear, visual, intuitive. Watch this first.
- [TF-IDF, Clearly Explained — StatQuest](https://www.youtube.com/watch?v=OymqCnh-uqc) — Direct follow-up, same quality
- [Confusion Matrix & Metrics — StatQuest](https://www.youtube.com/watch?v=Kdsp6soqA7o) — Covers all the metrics with real examples

**Read:**
- [Scikit-learn: Working with Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) — Practical walkthrough of BoW → TF-IDF → Logistic Regression pipeline
- [The Illustrated Machine Learning Glossary — Jay Alammar](https://jalammar.github.io/) — Excellent visual breakdowns

**Deep Dive (optional):**
- [Stanford CS224N Lecture Notes on BoW and Naive Bayes](https://web.stanford.edu/class/cs224n/) — If you want to see the math rigorously

---

## Before You Move On

- Can you explain Bag-of-Words to someone with no ML background in 2 minutes?
- Can you name 3 concrete reasons BoW is insufficient for understanding language?
- Can you explain what F1 score captures that accuracy doesn't?
- Given a confusion matrix, can you calculate Precision, Recall, and F1 by hand?
- Can you explain intuitively why cross-entropy is a better training signal than accuracy?

---

[Previous: Week 1 Overview](../../Week-1/README.md) | [Week 2 Overview](../README.md) | [Next: Introduction to Deep Learning](../02-Deep-Learning-Fundamentals/README.md)

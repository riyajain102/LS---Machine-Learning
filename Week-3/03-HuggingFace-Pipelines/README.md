# HuggingFace Pipelines

[Home](../../README.md) > [Week 3](../README.md) > HuggingFace Pipelines

| Week 3 · Topic 3 of 5 · Prerequisites: [Encoder vs Decoder](../02-Encoder-vs-Decoder/README.md) |
|---|

---

## Why This Topic

You now understand how Transformers work. You could, in theory, implement one from scratch. But in practice, the models that matter — the ones with real-world performance — have been trained on hundreds of billions of tokens using thousands of GPUs. You are not going to retrain BERT.

What you *will* do is use these models. HuggingFace is the platform that makes this possible. It hosts over 500,000 pre-trained models and provides a `pipeline` API that wraps any of them in a consistent, 5-line interface.

> ⚠️ **Setup required before running any code in this section.** See the setup block below.

---

## Setup

```python
# Run once in your Colab notebook
!pip install transformers torch sentencepiece -q

# You will need a free HuggingFace API token for some models.
# Create one at: https://huggingface.co/settings/tokens
# Then run:
from huggingface_hub import login
login("YOUR_HF_TOKEN_HERE")  # paste your token
```

---

## The Pipeline API

`pipeline()` is HuggingFace's high-level abstraction. You specify a *task*, optionally a *model*, and get back a callable that handles tokenisation, inference, and decoding automatically.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("This movie was absolutely brilliant!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

That's it. Behind the scenes, HuggingFace:
1. Downloaded a pre-trained `distilbert-base-uncased-finetuned-sst-2-english` model
2. Tokenised your input using the model's tokeniser
3. Ran a forward pass
4. Mapped the output logit to a human-readable label

---

## 1. Sentiment Analysis

```python
from transformers import pipeline

# Default model: DistilBERT fine-tuned on SST-2
sentiment = pipeline("sentiment-analysis")

reviews = [
    "The food was absolutely incredible, best meal I've had in years.",
    "Waited 45 minutes and the order was still wrong. Terrible service.",
    "It was okay, nothing special but not bad either."
]

results = sentiment(reviews)
for review, result in zip(reviews, results):
    print(f"{result['label']:10s} ({result['score']:.2%})  |  {review[:60]}")
```

**Under the hood:** This is an Encoder model (DistilBERT) with a classification head. The `[CLS]` token's output embedding is passed through a linear layer → softmax → label.

---

## 2. Zero-Shot Classification

Classify text into categories *you define at inference time* — no fine-tuning needed.

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

text = "The new iPhone has a faster chip and improved camera system."
labels = ["technology", "sports", "politics", "entertainment", "science"]

result = classifier(text, candidate_labels=labels)
print(result['labels'])   # ['technology', 'science', 'entertainment', ...]
print(result['scores'])   # [0.87, 0.06, 0.04, ...]
```

**How this works:** The model is trained on Natural Language Inference (NLI) — given a premise and hypothesis, predict entailment/contradiction/neutral. Zero-shot classification frames each label as a hypothesis: *"This text is about technology."* High entailment score → high classification confidence.

---

## 3. Text Generation

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The most important thing about machine learning is"
output = generator(
    prompt,
    max_new_tokens=80,
    num_return_sequences=2,
    temperature=0.8,      # > 1 = more random, < 1 = more focused
    do_sample=True,
)

for i, seq in enumerate(output):
    print(f"\n--- Sequence {i+1} ---")
    print(seq['generated_text'])
```

**Key generation parameters:**

| Parameter | What it controls |
|---|---|
| `max_new_tokens` | Maximum tokens to generate (not counting the prompt) |
| `temperature` | Randomness — 0.0 = deterministic greedy, 1.0 = full sampling distribution |
| `top_p` | Nucleus sampling — only sample from the top tokens summing to probability p |
| `do_sample` | If False, uses greedy decoding (always picks the most likely next token) |

---

## 4. Translation

```python
from transformers import pipeline

# Helsinki-NLP has 1000+ translation models
translator = pipeline("translation_en_to_fr",
                      model="Helsinki-NLP/opus-mt-en-fr")

texts = [
    "Machine learning is transforming every field of science.",
    "The attention mechanism allows models to focus on relevant context."
]

for text in texts:
    result = translator(text, max_length=200)
    print(f"EN: {text}")
    print(f"FR: {result[0]['translation_text']}\n")
```

---

## 5. Question Answering (Extractive)

Given a context passage and a question, find the answer *within* the passage.

```python
from transformers import pipeline

qa = pipeline("question-answering",
              model="deepset/roberta-base-squad2")

context = """
    The Transformer architecture was introduced in the paper 'Attention Is All You Need'
    published in 2017 by researchers at Google Brain. It replaced recurrent neural networks
    with self-attention mechanisms, enabling parallelisation and dramatically improving
    performance on sequence-to-sequence tasks. BERT, introduced in 2018, adapted the
    Transformer encoder for language understanding tasks.
"""

questions = [
    "When was the Transformer introduced?",
    "What did the Transformer replace?",
    "What is BERT used for?"
]

for q in questions:
    result = qa(question=q, context=context)
    print(f"Q: {q}")
    print(f"A: {result['answer']}  (confidence: {result['score']:.2%})\n")
```

---

## 6. Summarisation

```python
from transformers import pipeline

summariser = pipeline("summarization",
                      model="facebook/bart-large-cnn")

long_text = """
    Artificial intelligence researchers have developed a new technique that allows
    language models to generate more factually accurate responses. The method,
    called Retrieval-Augmented Generation (RAG), combines a language model with
    a search engine over a document database. When a question is asked, the system
    first retrieves relevant documents, then uses the language model to synthesise
    an answer from those documents. This approach dramatically reduces hallucination,
    the tendency of language models to generate plausible-sounding but false information.
    The technique has been adopted by major AI companies including Google, Microsoft,
    and Anthropic in their production AI assistants.
"""

result = summariser(long_text, max_length=80, min_length=30, do_sample=False)
print(result[0]['summary_text'])
```

---

## Choosing the Right Model

When a pipeline's default model isn't good enough, search [huggingface.co/models](https://huggingface.co/models) and filter by task. Key things to check:
- **Downloads/month** — proxy for community trust
- **Model card** — what data was it trained on? What tasks?
- **Size** — larger models are more capable but slower; `distilbert` variants are fast, `large` variants are more accurate

---

## Resources

**Read/Explore:**
- [HuggingFace Pipeline Documentation](https://huggingface.co/docs/transformers/main_classes/pipelines) — All supported tasks and parameters
- [HuggingFace Model Hub](https://huggingface.co/models) — Browse and filter 500k+ models
- [The Illustrated BERT — Jay Alammar](https://jalammar.github.io/illustrated-bert/) — Understand what happens inside pipeline("sentiment-analysis")

---

## Before You Move On

- Can you write a pipeline call for zero-shot classification in under 10 lines?
- What is the difference between `temperature=0.1` and `temperature=1.5` in text generation?
- Why does extractive QA return a span from the context, rather than generating a new answer?
- If you wanted to classify customer support tickets into 20 custom categories without labelled training data, which pipeline would you use?

---

[← Encoder vs Decoder](../02-Encoder-vs-Decoder/README.md) | [Week 3 Overview](../README.md) | [Next: Modern LLMs & Prompting →](../04-Modern-LLMs-and-Prompting/README.md)

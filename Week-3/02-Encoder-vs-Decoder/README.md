# Encoder vs Decoder Transformers

[Home](../../README.md) > [Week 3](../README.md) > Encoder vs Decoder

| Week 3 · Topic 2 of 5 · Prerequisites: [Transformer Architecture](../01-Transformer-Architecture/README.md) |
|---|

---

## Why This Topic

"Transformer" is not one model — it's an architecture. Different tasks need different versions of it. BERT and GPT are both Transformers. They look nearly identical on paper. But they are used for completely opposite things, and understanding *why* will make every paper you read and every model you use make immediate sense.

---

## The Two Halves of the Original Transformer

The 2017 "Attention Is All You Need" paper introduced a **full Encoder-Decoder** architecture for machine translation:

```
[Encoder Stack]         [Decoder Stack]
  Input text     →→→→→→  Output text (generated one token at a time)
  (full sequence)        (uses encoder output + its own previous tokens)
```

The **Encoder** reads the full input simultaneously. Its job is to build rich contextual representations of every token.

The **Decoder** generates output sequentially, one token at a time. It attends to its own previous outputs *and* to the encoder's representations.

Modern models took this architecture and kept only one half.

---

## Encoder-Only: BERT

**BERT** (Bidirectional Encoder Representations from Transformers, Google 2018) keeps only the encoder stack and discards the decoder.

**Key property — Bidirectional attention:** Every token attends to every other token, in both directions simultaneously. To understand the word at position 5, BERT looks at positions 1–4 *and* 6–end at the same time.

```
"The cat sat on the mat"
         ↑
      "sat" attends to: The, cat, on, the, mat  ← all directions, simultaneously
```

**How BERT was trained — Masked Language Modelling (MLM):**
15% of tokens are randomly masked. The model must predict the masked tokens using the surrounding context. Because it can see both left and right context, it learns deep bidirectional representations.

```
Input:  "The cat [MASK] on the mat"
Target: "sat"
```

**What BERT is good for (understanding tasks):**
- Text classification (sentiment, topic)
- Named entity recognition
- Question answering (finding an answer span in a passage)
- Sentence similarity

**What BERT cannot do:** generate text. Its bidirectional attention means it sees the future during training — you cannot use it to predict the next word autoregressively.

---

## Decoder-Only: GPT

**GPT** (Generative Pre-trained Transformer, OpenAI 2018–present) keeps only the decoder stack and discards the encoder.

**Key property — Causal (masked) attention:** Each token can only attend to *previous* tokens. The future is masked out. This is enforced with a causal mask — an upper triangular matrix of `-inf` values applied before the softmax.

```
"The cat sat on the mat"
         ↑
      "sat" attends to: The, cat  ← only past, never future
```

This is why it's called *autoregressive* — the model generates one token, appends it to the sequence, then generates the next, using all previous tokens as context.

**How GPT was trained — Next Token Prediction:**
Given a sequence of tokens, predict the next one. Applied to billions of tokens of internet text. Simple objective, extraordinary results at scale.

**What GPT is good for (generation tasks):**
- Text generation / completion
- Summarisation
- Translation
- Chatbots (with fine-tuning)
- Code generation

**What GPT is less suited for:** deep bidirectional understanding tasks. It never sees the full sentence at once during training.

---

## Side-by-Side Comparison

| | BERT (Encoder) | GPT (Decoder) |
|---|---|---|
| Attention direction | Bidirectional | Causal (left-to-right only) |
| Training objective | Masked Language Model | Next Token Prediction |
| Output | Contextual embeddings | Next token probabilities |
| Primary use | Understanding / classification | Generation |
| Examples | BERT, RoBERTa, DistilBERT | GPT-2, GPT-4, LLaMA, Mistral |

---

## Encoder-Decoder: T5, BART

Some models keep both halves — these are best for **sequence-to-sequence** tasks where you have a structured input and a structured output:
- **Translation:** English sentence → French sentence
- **Summarisation:** long document → short summary
- **Question answering:** question + context → answer

Examples: T5, BART, mT5.

---

## The Practical Rule of Thumb

> **Classifying or understanding something?** → Encoder (BERT family)
> **Generating something?** → Decoder (GPT family)
> **Transforming one sequence into another?** → Encoder-Decoder (T5 family)

When you open HuggingFace next session and choose a model, this distinction determines which one you reach for.

---

## Resources

**Read:**
- [BERT Paper — Devlin et al., 2018](https://arxiv.org/abs/1810.04805) — Abstract and Section 3 (Model Architecture). The rest is experiments.
- [The Illustrated BERT — Jay Alammar](https://jalammar.github.io/illustrated-bert/) — Essential companion to the paper.
- [The Illustrated GPT-2 — Jay Alammar](https://jalammar.github.io/illustrated-gpt2/) — Same quality, decoder side.

---

## Before You Move On

- Why can't you use BERT to generate a story?
- Why can't you use GPT as a drop-in replacement for BERT on a text classification task?
- What is the causal mask, and why does GPT need it during training?
- If you were building a document summariser, which architecture family would you choose and why?

---

[← Transformer Architecture](../01-Transformer-Architecture/README.md) | [Week 3 Overview](../README.md) | [Next: HuggingFace Pipelines →](../03-HuggingFace-Pipelines/README.md)

# Week 3 - Machine Learning LS 2026

[Home](../README.md) > Week 3

---

## The Big Picture

Week 1 gave you the fundamentals. Week 2 gave you the tools to represent language as numbers. This week, you meet the architecture that changed everything.

The Transformer (2017) didn't just improve NLP — it *replaced* almost every prior approach within two years. BERT, GPT, ChatGPT, Gemini, Stable Diffusion, AlphaFold — all Transformers. Understanding this architecture is not optional if you want to work in modern ML.

By the end of this week you will understand how Transformers work, be able to use state-of-the-art models in 5 lines of code, understand how ChatGPT was trained, and build a working RAG-powered chatbot.

---

## Learning Path

```
01 - Transformer Architecture
  └── The attention mechanism, Q/K/V, multi-head attention, positional encoding.
      This is the core. Take your time here.

02 - Encoder vs Decoder Transformers
  └── Why BERT and GPT look the same but are used completely differently.

03 - HuggingFace Pipelines
  └── Using state-of-the-art models in 5 lines of code.

04 - Modern LLMs and Prompting
  └── How ChatGPT was built, how to talk to it effectively.

05 - RAG and Vector Databases
  └── How to give a language model access to your own documents.

Assignment - Console Chatbot
  └── Build a retrieval-augmented chatbot from scratch.
```

---

## Topics

| # | Topic | What You Will Learn | Est. Time |
|---|-------|---------------------|-----------|
| 1 | [Transformer Architecture](./01-Transformer-Architecture/README.md) | Self-attention, Q/K/V, multi-head, positional encoding | 4–5 hrs |
| 2 | [Encoder vs Decoder](./02-Encoder-vs-Decoder/README.md) | BERT vs GPT, masked attention, use cases | 1–2 hrs |
| 3 | [HuggingFace Pipelines](./03-HuggingFace-Pipelines/README.md) | Sentiment, translation, generation, zero-shot pipelines | 2–3 hrs |
| 4 | [Modern LLMs & Prompting](./04-Modern-LLMs-and-Prompting/README.md) | Few-shot, CoT, RLHF, SFT | 2–3 hrs |
| 5 | [RAG & Vector Databases](./05-RAG-and-Vector-Databases/README.md) | Embeddings, vector search, retrieval-augmented generation | 2–3 hrs |

---

## Before You Move to Week 4

- Can you explain what the Query, Key, and Value matrices represent in your own words?
- Can you explain why BERT is used for classification but GPT is used for generation?
- Can you write a HuggingFace pipeline for sentiment analysis in under 10 lines?
- Can you explain what Chain-of-Thought prompting does and why it works?
- Can you describe the RAG pipeline end-to-end: from a user's question to the final answer?

---

[Home](../README.md) | [← Week 2](../Week-2/README.md) | [Next: Week 4 →]()

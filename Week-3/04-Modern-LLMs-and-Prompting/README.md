# Modern LLMs and Prompting

[Home](../../README.md) > [Week 3](../README.md) > Modern LLMs and Prompting

| Week 3 · Topic 4 of 5 · Prerequisites: [HuggingFace Pipelines](../03-HuggingFace-Pipelines/README.md) |
|---|

---

## Why This Topic

GPT-2 (2019) was a Transformer decoder trained on next-token prediction. GPT-4 (2023) is a Transformer decoder trained on next-token prediction. The architecture is largely the same. The difference is scale, data quality, and — critically — the training pipeline that happens *after* the base model is trained.

This section covers how a raw language model becomes ChatGPT, and how to talk to it effectively.

---

## From Pre-training to ChatGPT: The Three Stages

### Stage 1 — Pre-training (Base LLM)

Train a large Transformer decoder on next-token prediction over a massive text corpus (Common Crawl, Wikipedia, books, code, etc.). The model learns grammar, facts, reasoning patterns, and world knowledge implicitly.

**Result:** A model that is excellent at *completing* text, but has no concept of being an assistant. Ask it "What is the capital of France?" and it might complete the question with more questions, or continue as if writing a geography textbook.

---

### Stage 2 — Supervised Fine-Tuning (SFT)

Take the base model and fine-tune it on a curated dataset of **(instruction, ideal response)** pairs, written or curated by humans.

```
Instruction:  "Explain gradient descent to a 10-year-old."
Response:     "Imagine you're blindfolded on a hilly landscape and you want
               to reach the lowest point. You feel which direction slopes
               downward and take a small step that way. Repeat. That's
               gradient descent — the model takes small steps in the direction
               that reduces its mistakes."
```

**Result:** A model that learns the *format* of being helpful — answering questions, following instructions, completing tasks. But it can still be inconsistent, verbose, or occasionally harmful.

---

### Stage 3 — RLHF (Reinforcement Learning from Human Feedback)

This is the step that made ChatGPT feel *aligned* with what users actually want.

**Step 3a — Train a Reward Model:**
Show human raters pairs of model responses to the same prompt. They rank which response is better. Train a separate neural network (the *reward model*) to predict these human preference scores.

**Step 3b — Fine-tune with RL (PPO):**
Use the reward model as a signal to further fine-tune the LLM via Proximal Policy Optimisation (PPO). The LLM generates responses, the reward model scores them, and the LLM's weights are updated to produce higher-scoring responses.

```
Prompt → LLM generates response → Reward Model scores it → PPO updates LLM weights
         ↑___________________________________________________|
```

**Why RL and not just more SFT?**
SFT requires humans to *write* ideal responses — expensive and limited. RLHF only requires humans to *compare* responses — much faster and scalable to preferences that are hard to articulate but easy to judge.

**Result:** A model that is not just capable, but *helpful, harmless, and honest* (the "3H" goal, originally from Anthropic).

---

## Prompting Techniques

The base model is fixed at inference time — you cannot change its weights. But you can dramatically change its behaviour through *how you phrase your input*. This is **prompt engineering**.

---

### Zero-Shot Prompting

Ask the model to perform a task with no examples.

```
Prompt:   "Classify the sentiment of this review as Positive or Negative:
           'The battery life is terrible and the screen cracked after a week.'"

Response: "Negative"
```

Works well for tasks the model has seen plenty of during training.

---

### Few-Shot Prompting

Provide 2–5 examples of the task *inside the prompt*. The model infers the pattern and applies it to the new input.

```
Prompt:
"Classify sentiment. Examples:

Review: Great product, highly recommend!  → Positive
Review: Waste of money, broke after a day → Negative
Review: Decent quality, nothing special   → Neutral

Review: Absolutely love this, buying again! → "

Response: "Positive"
```

**Why it works:** LLMs are trained on text that contains many examples of tasks being demonstrated, so "learning from examples in context" is something they've seen at massive scale. This is called *in-context learning* — the model adapts to the task within a single forward pass, without weight updates.

---

### Chain-of-Thought (CoT) Prompting

Force the model to *reason step by step* before giving its final answer. Dramatically improves performance on arithmetic, logic, and multi-step reasoning tasks.

```
❌ Without CoT:
Prompt:   "Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. How many does he have?"
Response: "11"  ← correct, but fragile

✅ With CoT (just add "Let's think step by step"):
Prompt:   "Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
           How many does he have? Let's think step by step."
Response: "Roger starts with 5 balls. He buys 2 cans × 3 balls = 6 balls.
           Total: 5 + 6 = 11 balls."
```

**Why it works:** The model generates its "thinking" as text tokens before the answer. Each reasoning step conditions the next one, making it more likely the final answer is correct. You're essentially giving the model scratch paper.

**Zero-Shot CoT:** Just appending *"Let's think step by step"* to any prompt enables chain-of-thought reasoning without any examples.

---

### System Prompts

Modern chat APIs accept a *system prompt* — instructions that define the model's persona, constraints, and behaviour, separate from the user's message.

```python
messages = [
    {
        "role": "system",
        "content": "You are a concise ML tutor. Explain concepts clearly using analogies.
                    Never write more than 3 sentences per response."
    },
    {
        "role": "user",
        "content": "What is attention in a Transformer?"
    }
]
```

The model will adhere to the system prompt throughout the conversation.

---

### Prompt Engineering Best Practices

| Do | Don't |
|---|---|
| Be specific about format: "respond in bullet points" | Leave format ambiguous |
| Give examples (few-shot) for unusual tasks | Assume the model knows your exact intent |
| Use CoT for reasoning: "think step by step" | Ask for multi-step answers in one shot without guidance |
| Set a persona in the system prompt | Put everything in one giant unstructured message |
| Specify length: "in 2 sentences" | Hope the model chooses the right length |

---

## The Scale Question

| Model | Parameters | Training tokens | Notable capability |
|---|---|---|---|
| GPT-2 | 1.5B | 40B | Coherent paragraphs |
| GPT-3 | 175B | 300B | Few-shot learning, most tasks |
| LLaMA 2 (7B) | 7B | 2T | Runs on a consumer laptop |
| GPT-4 | ~1T (estimated) | Unknown | Near-human on most benchmarks |

Scale matters, but it's not everything. LLaMA 3 (8B) with good fine-tuning outperforms GPT-3 (175B) on many tasks. Data quality and alignment training matter as much as raw parameter count.

---

## Resources

**Read:**
- [Chain-of-Thought Prompting Elicits Reasoning — Wei et al., 2022](https://arxiv.org/abs/2201.11903) — The paper that introduced CoT. Short and readable.
- [InstructGPT Paper — Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) — How RLHF was used to build the model behind ChatGPT. Section 3 (Methods) is the key part.
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — Comprehensive, practical, well-maintained reference.

---

## Before You Move On

- What is the difference between a base LLM and an instruction-tuned LLM?
- Why does RLHF use human *comparisons* rather than human-written ideal responses?
- What does Chain-of-Thought prompting actually change about how the model produces its output?
- In few-shot prompting, the model's weights are not updated. How is it "learning" from the examples?

---

[← HuggingFace Pipelines](../03-HuggingFace-Pipelines/README.md) | [Week 3 Overview](../README.md) | [Next: RAG & Vector Databases →](../05-RAG-and-Vector-Databases/README.md)

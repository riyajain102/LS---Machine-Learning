# Agentic AI & Tool Calling

[Home](../../README.md) > [Week 4](../README.md) > Agentic AI & Tool Calling

| Week 4 · Topic 3 of 6· Prerequisites: [Reinforcement Learning](../01-Reinforcement-Learning/README.md) (helpful, not required) |
|---|

---

## Why This Topic

This is the most directly applicable topic in the week — it's the
mechanism underneath the IITB Assistant project. RAG, in particular, is
just one specific tool call in the broader pattern described here.
Understanding the loop, not just the API syntax, is what lets you debug an
agent when it does something unexpected.

---

## What You Will Learn

- What actually makes something "agentic" vs. a single LLM call
- The tool-calling / function-calling mechanics behind hosted APIs
- The ReAct pattern (Reason + Act) and why writing out reasoning helps
- Planning strategies beyond single-step ReAct: plan-and-execute,
  reflection, multi-agent orchestration
- The failure modes to design around: hallucinated tool calls, infinite
  loops, compounding errors, context bloat

---

## Why It Matters

| Failure Mode | Mitigation |
|---|---|
| Hallucinated tool calls | Strict schema validation, clear tool descriptions/examples |
| Infinite loops | Max-iteration budget, force a final answer past it |
| Compounding errors | Verification steps, human-in-the-loop checkpoints for high-stakes actions |
| Context bloat from long tool outputs | Summarize old steps, or retrieve instead of keeping everything verbatim |

---

## Concepts in Depth

### 🤖 What "Agentic" Actually Means

An LLM by itself just maps text to text. An **agent** is an LLM wrapped in
a loop that lets it observe the world (via tool outputs), decide on an
action (call a tool, or respond), take that action, and repeat — using its
own prior outputs as context for the next decision. This is the same
state → action → observation loop from [Topic 1](../01-Reinforcement-Learning/README.md),
except the "policy" is a frozen pretrained LLM being steered by prompting
rather than gradient updates.

### 🔧 Tool Calling (Function Calling) Mechanics

Modern hosted models are fine-tuned to support structured tool use:
1. You provide a list of tool definitions (name, description, JSON schema
   for arguments).
2. The model, instead of (or in addition to) a text reply, can emit a
   structured **tool call**: a tool name plus arguments matching the schema.
3. Your code executes the actual tool — the model never runs anything
   itself.
4. You feed the tool's result back into the conversation as a new message
   and call the model again.
5. Repeat until the model responds with a final answer instead of a tool call.

The model is just predicting "the next reasonable thing to emit" — the
schema and stop conditions are what turn free text generation into
something you can safely execute against.

### 🧠 The ReAct Pattern

**ReAct** (Reason + Act, Yao et al. 2022) interleaves explicit reasoning
text with actions:
```
Thought: I need the current weather in Mumbai to answer this.
Action: get_weather(city="Mumbai")
Observation: 31°C, humid
Thought: I can now answer.
Final Answer: ...
```
Writing the "Thought" out loud before acting measurably improves tool-use
accuracy — it gives the model a place to plan before committing to an
action, and gives you an inspectable trace of *why* it chose that tool.

### 📋 Planning Strategies Beyond Single-Step ReAct

- **Plan-and-execute** — generate a full multi-step plan up front, then
  execute steps one at a time, replanning only if something goes wrong.
  Cheaper (fewer full-context LLM calls), less adaptive.
- **Reflection / self-critique** — after producing an answer or taking an
  action, the agent (or a second LLM call) critiques the result and
  decides whether to retry.
- **Multi-agent orchestration** — split responsibilities across
  specialized agents (planner, researcher, coder, critic) communicating
  through structured messages. Reduces the context/attention burden per
  call, at the cost of orchestration complexity.

### 🔗 Where RAG Fits In

Retrieval-Augmented Generation is really just one specific, simple tool: a
"search my documents" tool usually called once, near the start, to ground
the model's answer in retrieved context rather than parametric memory. The
IITB Assistant applies exactly this pattern — everything above (loop
structure, tool schemas, failure modes) still applies; RAG is just one
tool in the toolbox.

---

## Resources

**Watch:**
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course) — free, hands-on course covering ReAct loops, tool use, and multi-agent patterns with runnable code.

**Read:**
- [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — practical patterns (prompt chaining, routing, orchestrator-workers, evaluator-optimizer) with clear tradeoffs, written from production experience.

**Deep Dive (papers):**
- [Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629)
- [Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*](https://arxiv.org/abs/2303.11366) — self-critique/retry loops.

---

## Before You Move On

- What's the actual difference between an LLM call and an agent — one
  sentence?
- Walk through the 5-step tool-calling mechanics without looking back.
- Why does writing a "Thought" before an "Action" measurably help, rather
  than just being extra tokens?
- Name two ways an agent loop can fail silently, and how you'd guard
  against each.
- Where does RAG fit into everything you just learned — is it a separate
  paradigm or a special case?

---

[Previous: DPO / PPO](../02-DPO-PPO/README.md) | [Week 4 Overview](../README.md) | [Next: Stable Diffusion](../04-Stable-Diffusion/README.md)

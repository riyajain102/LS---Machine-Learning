# DPO / PPO — Aligning Language Models

[Home](../../README.md) > [Week 4](../README.md) > DPO / PPO

| Week 4 · Topic 2 of 6· Prerequisites: [Reinforcement Learning](../01-Reinforcement-Learning/README.md) |
|---|

---

## Why This Topic

Pretraining teaches a model to predict the next token over internet text.
That's not the same as being a helpful, honest, instruction-following
assistant. **Alignment** is the extra training stage that closes that gap,
usually using human preference data. PPO and DPO are two different ways to
do it, and understanding both is what lets you reason about *why* a hosted
model behaves the way it does when you prompt it.

---

## What You Will Learn

- The full RLHF pipeline: SFT → reward modeling → PPO fine-tuning
- Why a KL penalty against the original model is essential, not optional
- DPO's key insight: how to skip the reward model and RL loop entirely
- When to reach for PPO vs. DPO in practice

---

## Why It Matters

| | PPO (RLHF) | DPO |
|---|---|---|
| Needs a separate reward model | Yes | No |
| Needs on-policy rollouts/sampling during training | Yes | No |
| Training stability | Harder to tune (RL) | Much more stable (supervised-style) |
| Compute/engineering cost | High (3+ models in memory) | Low (2 models: policy + frozen ref) |
| Ceiling / flexibility | Can optimize any reward signal, including online/changing rewards | Limited to a fixed offline preference dataset |

In practice, DPO (and variants — IPO, KTO, ORPO, SimPO) is the default for
most open-model alignment today because it's dramatically simpler to
implement correctly. PPO/RLHF is still preferred when you want to optimize
against a reward signal that can be sampled online — e.g. a verifiable
reward like "did the code pass tests," which is also the foundation of
RLVR-style reasoning-model training.

---

## Concepts in Depth

### 🏗️ The RLHF Pipeline (the PPO route)

This is the pipeline behind the original InstructGPT/ChatGPT recipe:

**Step 1 — Supervised Fine-Tuning (SFT).** Fine-tune the base model on
high-quality demonstrations of the target behavior.

**Step 2 — Reward Modeling.** Collect pairs of responses to the same
prompt, have humans rank which is better, and train a **reward model** to
predict that preference:
```
L = −log σ(r(x, y_chosen) − r(x, y_rejected))
```
The reward model only scores text — it doesn't generate it.

**Step 3 — RL fine-tuning with PPO.** Use the reward model as the reward
signal and fine-tune the SFT model with PPO (see [Topic 1](../01-Reinforcement-Learning/README.md)
for the general mechanics). Crucially, a **KL penalty** against the
original SFT model is added to the reward:
```
reward = r_θ(x,y) − β · KL(π_RL(y|x) ‖ π_SFT(y|x))
```
Without this, the policy drifts arbitrarily far to exploit quirks in the
reward model ("reward hacking") and produces degenerate, high-reward but
low-quality text.

**Why this pipeline is painful:** you need to train and host *three*
models simultaneously during RL fine-tuning (policy, reward model, and a
frozen reference model for the KL term), plus generate rollouts on-policy
at every step. Expensive, finicky, and prone to reward hacking.

### 🎯 DPO — The Simplification

**Direct Preference Optimization** (Rafailov et al., 2023) asks: do we
actually need the RL loop and a separate reward model at all?

The key insight: for the specific reward-plus-KL-penalty objective RLHF
optimizes, there's a closed-form relationship between the optimal policy
and the reward function. That lets you substitute the reward model's
preference loss *directly* in terms of the policy's own log-probabilities —
no reward model, no PPO rollouts, no RL loop:

```
L_DPO = −log σ( β·[log π_θ(y_w|x) − log π_ref(y_w|x)]
                − β·[log π_θ(y_l|x) − log π_ref(y_l|x)] )
```
where `y_w` is the preferred (winning) response, `y_l` the dispreferred
one, and `π_ref` is the frozen SFT model. This is just a supervised
classification-style loss over preference pairs — train it the same way
you'd fine-tune with cross-entropy, no environment or rollouts required.

---

## Resources

**Watch:**
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model — DPO paper explained](https://www.youtube.com/watch?v=XZLc09hkMwA) — walks through the derivation from the reward objective to the final DPO loss.
- StatQuest's *Reinforcement Learning with Human Feedback (RLHF), Clearly Explained* — search for it on the [StatQuest video index](https://statquest.org/video_index.html) for the intuitive, non-mathy version of the RLHF pipeline.

**Read:**
- [Hugging Face TRL documentation](https://huggingface.co/docs/trl) — has working PPO and DPO trainers; reading the code alongside the papers makes the abstractions concrete.

**Deep Dive (papers):**
- [Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Schulman et al., *Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347)
- [Rafailov et al., *Direct Preference Optimization*](https://arxiv.org/abs/2305.18290)

---

## Before You Move On

- Can you list the three stages of the RLHF pipeline in order, and what
  each one produces?
- What specifically goes wrong if you drop the KL penalty from PPO
  fine-tuning?
- In one sentence, what closed-form relationship does DPO exploit to skip
  the reward model?
- Give one concrete scenario where you'd still prefer PPO over DPO.

---

[Previous: Reinforcement Learning](../01-Reinforcement-Learning/README.md) | [Week 4 Overview](../README.md) | [Next: Agentic AI & Tool Calling](../03-Agentic-AI-Tool-Calling/README.md)

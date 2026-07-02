
# Week 4 - Machine Learning LS 2026 
[Home](../README.md) > Week 4
 
---
 
## The Big Picture
 
Week 3 gave you the Transformer — the architecture underneath every model
you've prompted so far. But knowing the architecture only explains how a
model turns tokens into tokens. It doesn't explain how that model was
*taught to behave* (RLHF, DPO), how it *acts* rather than just talks
(agentic AI, tool calling), or how the exact same ideas show up outside
text — generating images (diffusion), understanding pixels and sound
(vision, audio), and what's coming after attention itself (SSMs, JEPA).
 
This week is the "everything else" a working ML practitioner needs to be
fluent in. None of it is optional trivia — RLHF is why ChatGPT behaves
like an assistant instead of a raw completion engine, tool calling is the
mechanism your own RAG chatbot runs on, and diffusion/vision/audio are
where the same transformer ideas keep resurfacing in new clothes.
 
By the end of this week you will understand how a base model becomes an
aligned assistant, be able to explain what an "agent" actually is beyond
the marketing term, know how Stable Diffusion turns noise into an image,
and be able to place vision, audio, and next-generation architectures
(Mamba, JEPA) in context relative to everything you already know.
 
---
 
## Learning Path
 
```
01 - Reinforcement Learning
  └── The agent/action/reward loop, Q-learning, policy gradients, PPO.
      Foundational for everything alignment-related this week.
02 - DPO / PPO
  └── How a base model is actually aligned: SFT → reward model → PPO,
      and the simpler DPO alternative that skips the RL loop.
03 - Agentic AI & Tool Calling
  └── What makes an LLM call "agentic," the ReAct loop, and how your
      own RAG chatbot fits into this pattern.
04 - Stable Diffusion
  └── Forward/reverse diffusion, latent space, classifier-free guidance.
05 - Vision Models
  └── ViT, CLIP, and how vision-language models reuse the transformer.
06 - Audio Models
  └── Spectrograms, Whisper, and text-to-speech.
```
 
---
 
## Topics
 
| # | Topic | What You Will Learn 
|---|-------|---------------------
| 1 | [Reinforcement Learning](./01-Reinforcement-Learning/README.md) | MDPs, Q-learning, policy gradients, PPO 
| 2 | [DPO / PPO](./02-DPO-PPO/README.md) | RLHF pipeline, reward modeling, PPO vs. DPO 
| 3 | [Agentic AI & Tool Calling](./03-Agentic-AI-Tool-Calling/README.md) | Tool-calling mechanics, ReAct, planning, failure modes 
| 4 | [Stable Diffusion](./04-Stable-Diffusion/README.md) | Diffusion process, latent diffusion, guidance 
| 5 | [Vision Models](./05-Vision-Models/README.md) | ViT, CLIP, vision-language models 
| 6 | [Audio Models](./06-Audio-Models/README.md) | Spectrograms, Whisper, TTS 
 
---
 

 
[Home](../README.md) | ← Week 3: Transformer Architecture (previous week) | Next: IITB Assistant Project (tracked separately)

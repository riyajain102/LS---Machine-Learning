# Stable Diffusion

[Home](../../README.md) > [Week 4](../README.md) > Stable Diffusion

| Week 4 · Topic 4 of 6 · Prerequisites: None |
|---|

---

## Why This Topic

Diffusion is the dominant paradigm for image (and increasingly video/audio)
generation, and it's a genuinely different training philosophy from
next-token prediction. It's also the clearest example this week of how
much of modern generative AI is one core idea (learn to reverse a noising
process) wrapped in a few clever engineering choices to make it fast and
controllable.

---

## What You Will Learn

- The forward (fixed) and reverse (learned) diffusion processes
- Why the training objective is just a mean-squared-error on predicted noise
- The three additions that turn a diffusion model into *Stable* Diffusion:
  latent diffusion, cross-attention conditioning, classifier-free guidance
- How diffusion generalizes beyond images

---

## Why It Matters

| Design Choice | Problem it Solves |
|---|---|
| Latent diffusion (denoise in VAE latent space, not pixels) | Makes diffusion cheap enough to run on a consumer GPU |
| Cross-attention conditioning | Lets a text prompt actually steer the image, at every resolution |
| Classifier-free guidance | Lets you trade off prompt-faithfulness vs. diversity at sampling time, with one knob |
| Fast samplers (DDIM, DPM-Solver) | Cuts sampling from ~1000 steps to ~20–50 with similar quality |

---

## Concepts in Depth

### ❄️ Forward Process (Fixed, Not Learned)

Take a real image, add a little Gaussian noise, repeat many times until
it's pure noise:
```
x_t = √(1−β_t)·x_{t-1} + √β_t · ε,   ε ~ N(0, I)
```
This has a closed form letting you jump straight from `x_0` to any `x_t`
in one step — useful for training, since you don't need to simulate every
intermediate step.

### 🔄 Reverse Process (What the Network Learns)

A neural network `ε_θ(x_t, t)` is trained to predict the noise `ε` that was
added to produce `x_t`. The training objective is a plain MSE:
```
L = E‖ε − ε_θ(x_t, t)‖²
```
This simplicity — compared to a GAN's adversarial min-max game — is a big
part of why diffusion models train so much more stably.

### 🎲 Sampling

Start from `x_T ~ N(0, I)` (pure noise) and iteratively apply the learned
denoiser, subtracting predicted noise, until you reach `x_0` — a sample
from the data distribution. Smarter samplers (DDIM, DPM-Solver) get this
down to ~20–50 steps from the original ~1000.

### 🏛️ What Makes It "Stable" Diffusion Specifically

- **Latent diffusion** — a pretrained autoencoder (VAE) compresses images
  into a much smaller latent space; diffusion happens there, and the
  decoder maps back to pixels. This is the single biggest reason Stable
  Diffusion is cheap enough to run locally.
- **Conditioning via cross-attention** — a CLIP text encoder turns a
  prompt into embeddings injected into the U-Net denoiser via
  cross-attention at every resolution.
- **Classifier-free guidance (CFG)** — the model is trained to denoise
  both with and without text conditioning (dropping the prompt randomly
  during training). At sampling time, extrapolate *away* from the
  unconditional prediction toward the conditional one:
  ```
  ε_guided = ε_uncond + w · (ε_cond − ε_uncond)
  ```
  Higher `w` = more prompt-faithful but less diverse, can look over-saturated.

**Architecture at a glance:**
`Text prompt → CLIP text encoder → embeddings → cross-attention into U-Net`
`U-Net (self-attention + cross-attention blocks) denoises in latent space`
`→ VAE decoder → final image`

### 🌐 Where This Generalizes

The same forward/reverse noising framework underlies video generation
(Sora-style models) and audio generation — diffusion isn't image-specific,
it's a general framework for learning to sample from a complex
distribution.

---

## Resources

**Watch:**
- [How AI Image Generators Work (Stable Diffusion / DALL·E) — Computerphile](https://www.youtube.com/watch?v=1CIpzeNxIhU) — Dr. Mike Pound's clear, visual walkthrough. Start here.
- [But how do AI videos actually work? — 3Blue1Brown](https://www.3blue1brown.com/lessons/diffusion) — goes deeper into the math than the typical "model learns to remove noise" summary, with excellent visualizations.

**Read:**
- [Jay Alammar — The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) — the best diagram-first walkthrough of the full pipeline.

**Deep Dive (papers):**
- [Ho et al., *Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239)
- [Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752) — the Stable Diffusion paper.
- [Ho & Salimans, *Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598)

---

## Before You Move On

- Can you describe the forward and reverse processes without using the
  word "diffusion"?
- Why is the training loss just an MSE, and why does that matter compared
  to a GAN's loss?
- What specific problem does moving to latent space solve?
- Explain classifier-free guidance's `w` knob — what happens at `w=0` vs.
  a very high `w`?

---

[Previous: Agentic AI & Tool Calling](../03-Agentic-AI-Tool-Calling/README.md) | [Week 4 Overview](../README.md) | [Next: Vision Models](../05-Vision-Models/README.md)

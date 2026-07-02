# Vision Models

[Home](../../README.md) > [Week 4](../README.md) > Vision Models

| Week 4 · Topic 5 of 6 · Prerequisites: Week 3 Transformer Architecture |
|---|

---

## Why This Topic

Once you know the transformer architecture, vision models are mostly "the
same idea, different tokenization." Understanding that connection is what
makes CLIP, Stable Diffusion's text conditioning ([Topic 4](../04-Stable-Diffusion/README.md)),
and modern vision-language chat models all click as variations on one
theme rather than separate things to memorize.

---

## What You Will Learn

- Why CNNs' locality assumption is both their strength and their ceiling
- How Vision Transformer (ViT) applies the exact same self-attention
  machinery from NLP to image patches
- How CLIP connects vision and language via contrastive learning
- How modern vision-language models (VLMs) are built on top of an existing LLM

---

## Why It Matters

| Classical CNN Limitation | Transformer-era Solution |
|---|---|
| Built-in locality bias limits long-range relationships | Self-attention has no locality assumption — any patch can attend to any other |
| Vision and language live in separate architectures | ViT uses the same transformer block as text models, so they compose naturally |
| No native way to connect an image to a text description | CLIP's contrastive loss places matching (image, caption) pairs close in a shared embedding space |
| Vision-language systems needed training from scratch | A VLM can reuse a pretrained LLM + a lightweight projection layer instead |

---

## Concepts in Depth

### 🖼️ The CNN Era (for Context)

CNNs exploit two properties of images: **locality** (nearby pixels are
related) and **translation invariance** (a cat is a cat wherever it
appears). A small filter slides across the image, sharing weights across
positions, and stacking layers builds edges → textures → parts → objects.
This inductive bias makes CNNs data-efficient — but it's also a ceiling:
built-in locality limits how well they capture long-range relationships
across an image.

### 🧩 Vision Transformer (ViT)

ViT (Dosovitskiy et al., 2020) drops convolution's inductive bias entirely
and applies transformer self-attention to images:
1. Split the image into fixed-size patches (e.g. 16×16 pixels).
2. Flatten and linearly project each patch into an embedding — exactly
   like a token embedding in text.
3. Add positional embeddings (patches have 2D position, unlike text's
   linear order — this matters even more here).
4. Feed the sequence of patch embeddings through a standard transformer
   encoder.
5. A special `[CLS]` token's output embedding is used for classification.

With less built-in structure than a CNN, ViT needs *more* data to reach
the same performance — but with enough pretraining data it matches or
beats CNNs, and unifies vision architecture with the rest of the
transformer ecosystem.

### 🔗 CLIP — Connecting Vision and Language

CLIP (Radford et al., 2021) trains an image encoder and text encoder
*jointly* — not to classify into fixed categories, but to place matching
(image, caption) pairs close together in a shared embedding space, via a
**contrastive loss**: for a batch of N pairs, compute similarity between
every image and every text, push true pairs up and mismatched pairs down.

The result is zero-shot classification (compare an image embedding to
candidate text-label embeddings, no fine-tuning) — and, more importantly,
CLIP's image encoder became the standard backbone feeding vision into
Stable Diffusion's text conditioning and most vision-language models.

### 👁️ Vision-Language Models (VLMs)

Modern multimodal chat models generally follow this pattern:
1. A vision encoder (often ViT/CLIP-style) turns an image into a sequence
   of embeddings — "image tokens."
2. A lightweight projection layer maps these into the LLM's text-token
   embedding space.
3. Image and text tokens are concatenated and fed into the LLM together,
   so the model attends across both modalities with the same transformer
   machinery used for text.

This is why VLMs can be built relatively cheaply on top of an existing
LLM — you're mainly training the projection layer, not learning vision and
language from scratch together.

### 🔬 Other Things Worth Knowing

- **Object detection/segmentation** (YOLO, Segment Anything) — same
  encoder backbones, different task framing (boxes/masks vs. a label).
- **Self-supervised vision pretraining** (DINO, MAE) — learn good visual
  representations without labels, by predicting masked patches or matching
  augmented views. MAE is a direct conceptual cousin of JEPA
  ([Topic 7](../07-Bleeding-Edge-SSMs-JEPA/README.md)).

---

## Resources

**Watch:**
- [OpenAI CLIP: Connecting Text and Images (Paper Explained) — Yannic Kilcher](https://www.youtube.com/watch?v=T9XSU0pKX2E) — clear walkthrough of the contrastive objective and why it works.

**Read:**
- [AI Summer — Transformers in Computer Vision](https://theaisummer.com/transformers-computer-vision/) — good overview of ViT and its variants (Swin, DINO) with diagrams.

**Deep Dive (papers):**
- [Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT)](https://arxiv.org/abs/2010.11929)
- [Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)](https://arxiv.org/abs/2103.00020)
- [Liu et al., *Visual Instruction Tuning* (LLaVA)](https://arxiv.org/abs/2304.08485) — a clean, readable VLM architecture/training recipe.

---

## Before You Move On

- What specifically does ViT give up by dropping convolution's inductive
  bias, and what does it gain?
- Describe CLIP's contrastive loss in your own words — what's being
  pushed together, what's being pushed apart?
- How does a VLM reuse a pretrained LLM rather than training vision and
  language from scratch together?
- Can you name one self-supervised vision method and say, at a high
  level, what it's predicting?

---

[Previous: Stable Diffusion](../04-Stable-Diffusion/README.md) | [Week 4 Overview](../README.md) | [Next: Audio Models](../06-Audio-Models/README.md)

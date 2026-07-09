# Audio Models

[Home](../../README.md) > [Week 4](../README.md) > Audio Models

| Week 4 · Topic 6 of 6 · Prerequisites: [Vision Models](../05-Vision-Models/README.md) (helpful) |
|---|

> ⚠️ **Priority note:** This topic is **lower priority** relative to the
> others this week. Read it for breadth and to see how the spectrogram
> trick and transformer decoder ideas transfer to a new modality. Don't
> spend more than 2–3 hours here.

---

## Why This Topic

Audio models are a good test of whether the ideas from earlier topics
actually generalize. If you understand "turn the input into something
that looks like an image" (from Vision Models) and "predict the next token
autoregressively" (from the transformer architecture), most of ASR and TTS
is just those two ideas applied to sound.

---

## What You Will Learn

- How raw audio becomes a spectrogram a model can actually consume
- Whisper's encoder-decoder architecture for speech recognition
- Two dominant approaches to text-to-speech: diffusion-based and
  token-based
- How self-supervised audio representation learning mirrors masked
  language modeling

---

## Why It Matters

| Idea From Earlier This Week | Audio Application |
|---|---|
| Spectrogram-as-image ([Topic 5](../05-Vision-Models/README.md)) | Audio is converted to a 2D log-mel spectrogram and processed with the same conv/transformer toolkit used for vision |
| Diffusion ([Topic 4](../04-Stable-Diffusion/README.md)) | Diffusion/flow-based TTS applies the same denoising framework directly to audio, conditioned on text |
| Autoregressive next-token prediction | Token-based TTS treats speech generation as next-audio-token prediction, reusing the exact transformer decoder machinery behind text LLMs |

---

## Concepts in Depth

### 🔊 Turning Audio Into Something a Model Can Consume

Raw audio is a very high sample-rate 1D waveform (16,000–48,000 samples/
second) — too dense to feed directly into a transformer at reasonable
sequence lengths. The near-universal fix: convert audio into a **log-mel
spectrogram**, a 2D representation of frequency content over time,
produced via short-time Fourier transforms binned onto a mel scale
(weighted to match human hearing). This turns audio into something that
looks like an image.

### 🎙️ Automatic Speech Recognition — Whisper

Whisper (Radford et al., 2022) is a fairly standard encoder-decoder
transformer:
- **Encoder** — consumes the log-mel spectrogram of a 30-second audio chunk.
- **Decoder** — autoregressively generates text tokens, cross-attending to
  the encoder's output — the same cross-attention mechanism used in the
  original transformer and in Stable Diffusion's text conditioning.
- Special task tokens in the decoder's input let one model handle multiple
  tasks: transcription, translation-to-English, language ID, timestamps —
  all switched by the prompt.

What made Whisper notable wasn't architectural novelty — it's a fairly
standard transformer — but training at large scale on ~680k hours of
weakly-labeled, diverse multilingual audio, which gave it strong
robustness to accents and noise without task-specific fine-tuning.

### 🗣️ Text-to-Speech

Two dominant modern approaches:
- **Diffusion/flow-based TTS** — apply the same denoising framework from
  [Topic 4](../04-Stable-Diffusion/README.md) directly to audio (or an
  intermediate spectrogram/latent), conditioned on text.
- **Neural codec / token-based TTS** (e.g. VALL-E) — use a neural audio
  codec to compress audio into discrete tokens, then train a standard
  autoregressive transformer to predict audio tokens conditioned on text —
  treating speech generation as "just" another next-token-prediction problem.

### 🧠 Audio Representation Learning

Models like **wav2vec 2.0** and **HuBERT** learn general audio
representations via self-supervised objectives — predicting masked
portions of the audio's own learned discrete units, conceptually similar
to BERT's masked language modeling and to MAE/JEPA in vision. These
representations transfer well to speaker ID, emotion recognition, and
low-resource ASR.

---

## Resources

**Read:**
- [Hugging Face Audio Course](https://huggingface.co/learn/audio-course) — free, well-sequenced practical intro covering spectrograms through ASR/TTS pipelines.

**Deep Dive (papers):**
- [Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper)](https://arxiv.org/abs/2212.04356)
- [Baevski et al., *wav2vec 2.0*](https://arxiv.org/abs/2006.11477)
- [Wang et al., *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers* (VALL-E)](https://arxiv.org/abs/2301.02111)

---

## Before You Move On

- Why can't you just feed raw waveform samples directly into a
  transformer at a reasonable sequence length?
- What role does cross-attention play in Whisper's decoder?
- Name the two dominant TTS approaches and one key difference between them.
- How does wav2vec 2.0's pretraining objective echo something you learned
  earlier this week?

---

[Previous: Vision Models](../05-Vision-Models/README.md) | [Week 4 Overview](../README.md) 


# Introduction to Deep Learning

[Home](../../README.md) > [Week 2](../README.md) > Introduction to Deep Learning

| Week 2 · Topic 2 of 3 · Prerequisites: [Classical ML & Evaluation](../01-Classical-ML-and-Evaluation/README.md) |
|---|

---

## Why This Topic

This is the most important topic of Week 2. Everything in modern ML — GPT, BERT, Stable Diffusion, AlphaFold — is built on the ideas introduced here. A neuron. A layer. An activation function. These are the atoms of every model you will ever build or use.

The goal of this section is not to make you implement backpropagation from scratch (yet) — it is to give you a crystal-clear mental model of **what a neural network is, how it processes information, and why it can learn things that classical ML cannot**.

> 💡 Take your time here. Every hour you invest in understanding this deeply will save you 10 hours of confusion later.

---

## What You Will Learn

- **A Neuron** — the fundamental computational unit; what it computes and why
- **A Perceptron** — the original single-neuron model; what it can and cannot do
- **Multi-Layer Perceptron (MLP)** — stacking layers to learn complex functions
- **Hidden Layers** — what they represent and why depth matters
- **Activation Functions** — tanh, ReLU, and Softmax; what they do and when to use each
- **The forward pass** — how data flows from input to output through a network

---

## Why It Matters

Classical ML drew hard lines — literally. Logistic Regression can only draw a straight boundary between classes. Real data almost never separates that cleanly.

Neural networks solve this by composing many simple operations into complex, non-linear transformations. Add enough layers and enough neurons, and you can approximate **any function** — this is the Universal Approximation Theorem.

The intuition: each layer learns to look at the data from a higher, more abstract perspective. In an image model, early layers detect edges, middle layers detect shapes, later layers detect faces. In a language model, early layers detect word patterns, later layers detect meaning and context.

---

## Concepts in Depth

---

### 🧠 A Biological Neuron → An Artificial Neuron

A biological neuron:
1. Receives signals from other neurons via **dendrites**
2. If the total signal is strong enough, it **fires** a signal down its **axon**
3. That signal reaches the next neuron

An artificial neuron does the exact same thing mathematically:

```
Inputs:   x₁, x₂, x₃  (numbers — e.g., word counts, pixel values)
Weights:  w₁, w₂, w₃  (learned — how much to care about each input)
Bias:     b            (learned — shifts the activation threshold)

Step 1 — Weighted Sum:   z = w₁x₁ + w₂x₂ + w₃x₃ + b
Step 2 — Activation:     output = f(z)   where f is an activation function
```

The **weight** controls how strongly a neuron responds to an input.  
The **bias** shifts the threshold at which the neuron activates.  
The **activation function** adds non-linearity — without it, stacking neurons is pointless (a stack of linear functions is still a linear function).

---

### ⚡ The Perceptron

The perceptron (Rosenblatt, 1958) is the simplest possible network: **one neuron**.

```
Input layer → [Single Neuron] → Output
```

It computes: `output = step(w·x + b)`

Where `step(z) = 1 if z > 0 else 0` — it fires or it doesn't.

**What it can learn:** Any problem where a **single straight line** separates the classes (linearly separable). Classic example: AND, OR gates.

**What it cannot learn:** XOR — the famous example that shows a single neuron is fundamentally limited. You cannot draw one line to separate XOR inputs. This caused the first "AI winter" in the 1970s.

**The fix:** Stack multiple neurons. Use multiple layers. This is the MLP.

---

### 🏗️ Multi-Layer Perceptron (MLP)

An MLP stacks neurons into **layers**:

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer
```

Example: classifying movie reviews as Positive / Negative

```
Input:         TF-IDF vector of size 5000
               [0.0, 0.3, 0.0, 0.7, ...]
                         ↓
Hidden Layer 1: 256 neurons (each connected to all 5000 inputs)
                         ↓
Hidden Layer 2: 128 neurons
                         ↓
Output Layer:   2 neurons  → [P(positive), P(negative)]
```

Every neuron in one layer is connected to every neuron in the next — this is called a **fully connected** or **dense** layer.

**Total parameters in this example:**
- Layer 1: 5000 × 256 weights + 256 biases = 1,280,256
- Layer 2: 256 × 128 weights + 128 biases = 32,896
- Output: 128 × 2 + 2 = 258
- **Total: ~1.3 million learnable parameters**

All of these are adjusted during training via backpropagation to minimise the loss.

---

### 🔍 Hidden Layers — What Are They Learning?

The input layer is your raw data. The output layer is your prediction. The hidden layers in between are **learned feature detectors** — the network decides what to look for.

You cannot directly interpret what a hidden neuron "means" — unlike hand-crafted features in classical ML. But you can observe that:

- Early hidden layers tend to detect low-level patterns (word co-occurrences, simple phrases)
- Later hidden layers combine these into higher-level concepts (sentiment, topic, intent)

This **hierarchical feature learning** is why deep networks outperform shallow ones on complex tasks — depth allows the model to build increasingly abstract representations.

**Why not just make the network very wide instead of deep?**
Depth is more parameter-efficient than width. A deep, narrow network can represent functions that would require an exponentially wider shallow network.

---

### 🔋 Activation Functions

Without activation functions, an MLP collapses into a single linear transformation — no matter how many layers you add. Activation functions inject **non-linearity**, allowing the network to learn complex, curved decision boundaries.

---

#### tanh (Hyperbolic Tangent)

```
tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)

Output range: [-1, 1]
Shape: S-curve, centred at 0
```

**Properties:**
- Zero-centred output (unlike sigmoid) — helps gradient flow in early layers
- Saturates at both ends: for very large or very small z, the gradient → 0
- This causes the **vanishing gradient problem** in very deep networks

**When used:** Hidden layers in RNNs, some older architectures. Rarely used in modern feedforward nets (ReLU is preferred).

---

#### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)

Output range: [0, ∞)
Shape: flat at 0 for negative inputs, linear for positive
```

**Properties:**
- Computationally trivial (just a threshold)
- Does **not** saturate for positive values → gradients flow freely during backprop
- Sparse activation: roughly half of neurons are "off" at any time → efficient
- **Dead ReLU problem:** neurons with z < 0 output 0 and have gradient 0 — if this happens consistently, the neuron stops learning forever

**When used:** Default activation for hidden layers in almost every modern feedforward and convolutional network. If in doubt, use ReLU.

**Variants:** LeakyReLU (small negative slope instead of 0), GELU (used in transformers like BERT and GPT).

---

#### Softmax (Output Layer for Classification)

```
softmax(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ

Output range: (0, 1) for each class, and all outputs sum to 1
```

Softmax converts raw output scores (logits) into a **probability distribution** over classes.

Example with 3 classes (Positive, Neutral, Negative):
```
Raw logits:   [2.0,  0.5,  -1.0]
After softmax: [0.71, 0.24,  0.05]   ← sums to 1.0
```

The model predicts "Positive" with 71% confidence.

**Why softmax and not just argmax?**
- Softmax is differentiable — you can backpropagate through it
- It gives calibrated probabilities, not just a winner
- Combined with cross-entropy loss, it provides strong gradients for learning

**Important:** Softmax is used at the **output layer only**. Never use it in hidden layers.

---

### 🔄 Putting It All Together: The Forward Pass

Here's the complete flow for a 2-hidden-layer MLP classifying text sentiment:

```
1. INPUT
   Raw TF-IDF vector x ∈ ℝ⁵⁰⁰⁰

2. HIDDEN LAYER 1
   z₁ = W₁·x + b₁          (linear transformation)
   a₁ = ReLU(z₁)            (activation)

3. HIDDEN LAYER 2
   z₂ = W₂·a₁ + b₂          (linear transformation)
   a₂ = ReLU(z₂)            (activation)

4. OUTPUT LAYER
   z₃ = W₃·a₂ + b₃          (linear transformation)
   ŷ  = Softmax(z₃)         (convert to probabilities)

5. LOSS
   L = CrossEntropy(ŷ, y_true)

6. BACKPROPAGATION (training only)
   Compute gradients of L w.r.t. all W and b
   Update: W ← W - η·∂L/∂W  (η = learning rate)
```

The forward pass computes a prediction. The backward pass computes how to improve. Repeat for millions of examples. This is training.

---

## Resources

**Watch (in order):**
- [But what is a neural network? — 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) — The single best visual introduction to MLPs ever made. **Watch this first, no exceptions.**
- [Gradient descent, how neural networks learn — 3Blue1Brown](https://www.youtube.com/watch?v=IHZwWFHWa-w) — Part 2 of the series. Covers the learning process.
- [What is backpropagation really doing? — 3Blue1Brown](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — Part 3. How gradients flow backward.
- [Activation Functions Explained — Andrej Karpathy (CS231n)](https://www.youtube.com/watch?v=gYpoJMlgyXA) — Deep dive into ReLU, tanh, their tradeoffs

**Read:**
- [Neural Networks and Deep Learning — Michael Nielsen (Ch. 1)](http://neuralnetworksanddeeplearning.com/chap1.html) — Free online book. Chapter 1 is the clearest written introduction to MLPs.
- [The Illustrated Neural Network — Jay Alammar](https://jalammar.github.io/) — Beautiful visual walkthroughs

**Interactive:**
- [TensorFlow Playground](https://playground.tensorflow.org/) — Build and train neural networks in your browser. **Spend 30 minutes here — it is the best way to build intuition for hidden layers and activations.**

**Deep Dive:**
- [CS231n Lecture Notes — Karpathy](https://cs231n.github.io/neural-networks-1/) — Stanford's canonical deep learning course. Covers everything in this section rigorously.

---

## Before You Move On

- Can you draw a 3-layer MLP from scratch, labelling inputs, weights, activations, and outputs?
- Can you explain why activation functions are necessary? What happens without them?
- Can you describe the difference between ReLU and tanh, and name one advantage of each?
- Can you explain what Softmax does and why it's used at the output layer?
- Can you trace a single input through the forward pass and describe what happens at each step?

---

[Previous: Classical ML & Evaluation](../01-Classical-ML-and-Evaluation/README.md) | [Week 2 Overview](../README.md) | [Next: Embeddings & Word2Vec](../03-Embeddings-and-Word2Vec/README.md)

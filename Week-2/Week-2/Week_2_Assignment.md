# Week 2 Assignment

Welcome to your Week 2 assignment for the WnCC Machine Learning Learner Space! This week you move from theory into building real NLP systems. We have prepared Google Colab notebooks for you — no local setup needed.

**How to start:**
1. Click the Colab link for your chosen track
2. Go to `File` → `Save a copy in Drive`
3. Start coding in your own copy
4. Run the "Sanity Check" cells to verify your implementation is correct

---

## The Main Mission: Sentiment Analyser (Mandatory)

**Objective:** Build a complete text sentiment analysis pipeline from scratch — from raw text to a trained neural network that can tell you whether a review is positive or negative.

**What you will build:**
- A text preprocessing pipeline (cleaning, tokenisation)
- A Bag-of-Words and TF-IDF vectoriser (implemented manually, not just `sklearn.fit()`)
- Evaluation: compute Accuracy, Precision, Recall, F1 Score, and plot a Confusion Matrix
- A fully connected MLP (using PyTorch or NumPy) trained on the vectorised text
- Compare: does your MLP beat your Logistic Regression baseline? Why or why not?

**Dataset:** IMDb Movie Reviews (50,000 reviews, binary sentiment labels — provided in the notebook)

**What to submit:**
- Your completed Colab notebook with all cells run and outputs visible
- A short written reflection (inside the notebook, in a Markdown cell) answering:
  - What was the accuracy of your BoW + Logistic Regression baseline?
  - What was the accuracy of your MLP?
  - What surprised you? What didn't work as expected?

📎 [Open in Colab — Sentiment Analyser Starter]([https://colab.research.google.com/drive/PLACEHOLDER_MAIN_LINK?usp=sharing](https://colab.research.google.com/drive/1skoJLLBET9wiQMjxhV9Ur4gII-Cd0t-V#scrollTo=oQl6NyDI17fY))

---

## The Hacker Track: Name Completer with Embeddings (Optional / Advanced)

**Objective:** Build a character-level name completion model — given the first few characters of a name, predict the next character. This is a miniature version of how language models work, implemented from scratch using NumPy.

**What you will build:**
- A character-level tokeniser and one-hot encoder
- A lookup embedding table (learned, not pre-trained)
- A simple MLP that takes an embedding as input and predicts the next character
- Training loop with cross-entropy loss
- Interactive demo: type `"Dhr"` → model completes → `"Dhruv"`, `"Dhriti"`, etc.

**Dataset:** A list of Indian first names (provided in the notebook)

**Bonus Challenge:** Swap out the one-hot input for a learned embedding layer. Does the model perform better? Can you visualise the character embeddings using PCA?

📎 [Open in Colab — Name Completer Starter]([https://colab.research.google.com/drive/PLACEHOLDER_HACKER_LINK?usp=sharing](https://colab.research.google.com/drive/1Zw-nG6kCzLc0XeCb3meV7NUPZly9i0a_#scrollTo=pzwA_0r034IR))

> 💡 **This track is directly inspired by Andrej Karpathy's makemore project** — one of the best pedagogical ML projects ever built. If you complete this, you will have a real intuition for how character-level language models work.

---

## Submission

Submit your completed notebook link (make sure sharing is set to "Anyone with the link can view") to the WnCC submission form by the end of Week 2.

Questions? Reach out to the course moderators on the WhatsApp group or open an issue in this repository.

# Week 1: Introduction to Machine Learning & Prerequisites

Welcome to Week 1! This week is all about building a solid foundation. We will start by understanding the high-level concepts of Machine Learning (how algorithms learn, gradient descent, and neural networks) before rolling up our sleeves to master the fundamental programming tools and libraries required for the rest of the course.

## Table of Contents
1. [Introduction to Machine Learning (Theory)](#1-introduction-to-machine-learning-theory)
2. [How Machines Learn: Gradient Descent & Backpropagation](#2-how-machines-learn-gradient-descent--backpropagation)
3. [Environment & Programming Prerequisites (Practical)](#3-environment--programming-prerequisites-practical)
4. [Data Pre-processing & Introduction to NLP](#4-data-pre-processing--introduction-to-nlp)
5. [Bonus / Deep Dive Material](#5-bonus--deep-dive-material)

---

## 1. Introduction to Machine Learning (Theory)
Before writing any code, we need to understand what Machine Learning is, the difference between supervised and unsupervised learning, and the general ML lifecycle (data collection, training, evaluation, inference).

**Watch:**
-  📺 [A Gentle Introduction to Machine Learning - StatQuest](https://www.youtube.com/watch?v=Gv9_4yMHFhI) *(Great high-level overview)*
-  📺 [But what is a neural network? | Deep learning chapter 1 - 3b1b](https://www.youtube.com/watch?v=aircAruvnKk) *(Excellent visual explanation of network architecture)*

**Interactive Practice:**
-  💻 [Kaggle Learn: Intro to Machine Learning (Chapters 1-5)](https://www.kaggle.com/learn/intro-to-machine-learning) *(Interactive introduction covering ML models, data exploration, and model validation)*

---

## 2. How Machines Learn: Gradient Descent & Backpropagation
Here we dive into the core engines of machine learning. You will learn about Loss Functions (how we measure error) and Gradient Descent (how we minimize that error to "learn"). 

**Watch:**
-  📺 [Gradient Descent, Step-by-Step - StatQuest](https://www.youtube.com/watch?v=sDv4f4s2SB8) *(The intuition behind minimizing loss)*
-  📺 [Stochastic Gradient Descent, Clearly Explained!!! - StatQuest](https://www.youtube.com/watch?v=vMh0zPT0tLI) *(How to make gradient descent computationally efficient)*
-  📺 [Gradient descent, how neural networks learn | Deep Learning Chapter 2 - 3b1b](https://www.youtube.com/watch?v=IHZwWFHWa-w) *(The math and visuals behind network learning)*
-  📺 [Backpropagation, intuitively | Deep Learning Chapter 3 - 3b1b](https://www.youtube.com/watch?v=Ilg3gGewQ5U) *(How error is pushed back through the network to update weights)*

---

## 3. Environment & Programming Prerequisites (Practical)
Machine Learning requires a specific set of tools. Python is our language, Google Colab is our workspace, and libraries like NumPy, Pandas, and Matplotlib are our tools for handling and visualizing data. 

**Environment Setup:**
-  📝 [Welcome to Colaboratory - Google](https://colab.research.google.com/notebooks/intro.ipynb#scrollTo=5fCEDCU_qrC0) *(Learn how to run Python code in the browser using Jupyter-style notebooks)*

**Python Basics:**
-  💻 [Kaggle Learn: Python](https://www.kaggle.com/learn/python)

**Data Science Libraries (The "Holy Trinity"):**
-  📖 **NumPy** (Math & Arrays): [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
-  📖 **Matplotlib** (Plotting/Visuals): [Matplotlib: Pyplot Tutorial](https://matplotlib.org/stable/tutorials/pyplot.html)
-  💻 **Pandas (Data Manipulation):** [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas)

> **Tip:** It is recommended to practice these yourselves in a colab notebook.

---

## 4. Data Pre-processing & Introduction to NLP
Data is rarely ready for ML models right out of the box. In this section, we cover practical data splits (Training vs. Validation vs. Test sets), basic text cleaning, and the theory behind Tokenization (how text is converted into numbers for models to process).

**Intro  to NLP:**
- https://youtu.be/fLvJ8VdHLA0?si=3J-O1uQLccL0dg4I
**Practical Text Processing & Splits:**
-  💻 [Kaggle Code: Getting Started with NLP for absolute beginners](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners) *(Hands-on guide by Jeremy Howard covering basic text processing and creating training/validation splits)*

**Tokenization Theory:**
-  📺 [Let's build the GPT Tokenizer - Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE) **(Only upto 27:02)**

---

## 5. Bonus / Deep Dive Material
If you have extra time or want a much deeper understanding of the math and library mechanics, check out these advanced resources.

**Intermediate Deep Learning:**
-  📺 [The spelled-out intro to neural networks and backpropagation: building micrograd - Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0) *(Recommended if you want to understand exactly how backpropagation is coded from scratch in Python)*

**Detailed Chapters (by Pandas Creator Wes McKinney):**
-  📖 [Python for Data Analysis: NumPy Basics](https://wesmckinney.com/book/numpy-basics)
-  📖 [Python for Data Analysis: Pandas Basics](https://wesmckinney.com/book/pandas-basics)

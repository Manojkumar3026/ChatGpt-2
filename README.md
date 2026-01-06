**🧠 ChatGPT-2 (From Scratch in Python)**

A minimal yet powerful Transformer-based language model built entirely from scratch using pure Python.
This project demonstrates the core architecture behind modern LLMs like GPT—without using deep learning frameworks such as PyTorch or TensorFlow.

**🚀 Project Overview**

This project is an educational implementation of a GPT-style language model, designed to deeply understand:

Self-Attention mechanics

Transformer blocks

Layer Normalization

Feedforward (Dense) networks

Token-based text generation

The model is trained to predict the next token in a sequence, enabling basic text generation similar to early GPT models.

**🏗️ Architecture**

The model follows the classic Decoder-only Transformer architecture:

Input Tokens
   ↓
Token Embedding + Positional Encoding
   ↓
[ Transformer Block × N ]
   ├─ Multi-Head Self Attention
   ├─ Add & Layer Normalization
   ├─ Feed Forward (Dense Layers)
   └─ Add & Layer Normalization
   ↓
Linear Projection
   ↓
Softmax → Next Token Prediction

🧩 Core Components Implemented
✅ Self-Attention Module

Scaled dot-product attention

Causal masking (prevents looking into the future)

Multi-head attention support

✅ Layer Normalization

Implemented manually (mean, variance, epsilon handling)

Applied after attention and feedforward layers

✅ Feedforward Network

Two dense layers with activation

Expands and compresses embedding dimensions

✅ Transformer Block

Residual connections

Attention → Norm → FFN → Norm pipeline

✅ Language Modeling Head

Linear projection from embeddings to vocabulary size

Softmax-based probability distribution

**🛠️ Tech Stack**

Language: Python

Libraries:

numpy (matrix operations)

math (scaling & stability)

No frameworks used (No PyTorch / TensorFlow)

**📁 Project Structure**
chatgpt2-from-scratch/
│
├── tokenizer.py        # Tokenization logic
├── attention.py        # Self-attention implementation
├── layer_norm.py       # Layer normalization
├── dense.py            # Feedforward layers
├── transformer.py      # Transformer block
├── model.py            # GPT-style model assembly
├── train.py            # Training loop
├── generate.py         # Text generation
└── README.md

**⚙️ How It Works**

Text is tokenized into integer IDs

Tokens are embedded and positionally encoded

Data flows through stacked Transformer blocks

Model predicts the probability of the next token

Tokens are sampled iteratively to generate text

**▶️ Usage**
Train the Model
python train.py

Generate Text
python generate.py

**🎯 Learning Objectives**

This project helped achieve:

Deep understanding of Transformer internals

Hands-on experience with attention math

Clarity on how LLMs work without abstractions

Confidence to build models beyond frameworks

**⚠️ Limitations**

Not optimized for large-scale training

Slower compared to GPU-based frameworks

Intended for learning & experimentation, not production

**🌱 Future Improvements**

Byte Pair Encoding (BPE) tokenizer

Better sampling (Top-k, Top-p)

Weight saving/loading

Mini-batch training

GPU acceleration support

**🧑‍💻 Author**

Built with passion for deep learning fundamentals and LLM architecture exploration.

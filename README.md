# GPT From Scratch

A character-level GPT language model built from scratch in PyTorch, trained on Shakespeare's text. Implements the full transformer architecture including multi-head self-attention, feed-forward layers, residual connections, and layer normalization.

---

## What It Does

The model learns to generate Shakespeare-style text one character at a time. Given a prompt, it continues the text in the style it was trained on.

**Example output:**
```
Triumpens lady as this like asister,
To regal inveness of me; ware person,
So two much mirclip in justice: with such ron,
The worlds and is larged in presence.

Second Murderer:
Here it that; and for our coming with himself.

Servant:
It is that all forning it, that would be done.
Fool, no more to that he rest.

CLARENCE:
To the Clarence intended every Baptiss so here,
Lest his successful way, till he is to resist
A danger of direfulness at reason.
```

---

## Architecture

Built following Andrej Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial, implementing a decoder-only transformer:

- **Token + Positional Embeddings** — maps characters to vectors and encodes position
- **Multi-Head Self-Attention** — 6 heads, each learning different relationships between tokens
- **Feed-Forward Network** — per-token processing after attention
- **Residual Connections** — helps with training deep networks
- **Layer Normalization** — stabilizes training
- **6 Transformer Blocks** stacked sequentially

```
Input (characters)
      ↓
Token Embedding + Position Embedding
      ↓
[Block x6]
  └── LayerNorm → Multi-Head Attention → Residual
  └── LayerNorm → FeedForward → Residual
      ↓
LayerNorm
      ↓
Linear → Logits → Softmax → Next character
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Block size (context length) | 256 |
| Embedding size | 384 |
| Number of heads | 6 |
| Number of layers | 6 |
| Dropout | 0.2 |
| Learning rate | 3e-4 |
| Training steps | 5000 |
| Parameters | ~10.8M |

---

## Files

| File | Description |
|------|-------------|
| `bigram.py` | Full model definition and training loop |
| `generate.py` | Load saved model and generate text interactively |

---

## Setup

**Clone the repo:**
```bash
git clone https://github.com/sainitish1609/gpt-from-scratch.git
cd gpt-from-scratch
```

**Install dependencies:**
```bash
pip install torch
```

**Download the dataset:**
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

---

## Training

```bash
python3 bigram.py
```

Training prints loss every 500 steps. A well-trained model reaches around `1.4-1.6` validation loss:
```
Device: mps
10.788929 M parameters
step 0:    train loss 4.2221, val loss 4.2306
step 500:  train loss 1.7453, val loss 1.9068
step 1000: train loss 1.3902, val loss 1.5966
step 1500: train loss 1.2653, val loss 1.5245
step 2000: train loss 1.1863, val loss 1.4978
step 2500: train loss 1.1223, val loss 1.4985
step 3000: train loss 1.0723, val loss 1.4794
step 3500: train loss 1.0184, val loss 1.5027
step 4000: train loss 0.9610, val loss 1.5156
step 4500: train loss 0.9103, val loss 1.5475
step 4999: train loss 0.8619, val loss 1.5596
Model saved!
```

The trained model is saved as `gpt_model.pth`.

**Recommended hardware:**
- GPU with at least 4GB VRAM (NVIDIA or Apple Silicon MPS)
- CPU works but is significantly slower

---

## Generating Text

```bash
python3 generate.py
```

Interactive prompt:
```
Enter your prompt (or 'quit' to exit): To be or not to be
How many tokens to generate? (default=200): 300

-----Output-----
To be or not to be, that is the question...
--------End-------
```

---

## How It Works

**Training:**
1. Text is encoded character by character into integers
2. Random chunks of 256 characters are sampled as training batches
3. The model predicts the next character at every position
4. Cross-entropy loss is minimized using AdamW optimizer

**Generation:**
1. Your prompt is encoded into integers
2. The model predicts the next character probabilities
3. A character is sampled from those probabilities
4. Appended to the context and repeated until desired length

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)
- MPS (optional, for Apple Silicon)

---

## Credits

- Based on Andrej Karpathy's [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Dataset: [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- Original paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
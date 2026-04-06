# Project 1 — Transformer from Scratch
### "Attention Is All You Need" (Vaswani et al., NeurIPS 2017)

Full encoder-decoder Transformer implemented in PyTorch, trained on English→German translation.

---

## Architecture

Implemented every component from the paper:

| Component | Details |
|-----------|---------|
| Multi-head attention | Scaled dot-product, Q/K/V projections, output projection |
| Positional encoding | Sinusoidal (fixed), sin for even dims, cos for odd dims |
| Encoder layer | Self-attention → Add & Norm → FFN → Add & Norm |
| Decoder layer | Masked self-attention → Cross-attention → FFN (each with Add & Norm) |
| Causal masking | Upper-triangular mask to prevent decoder from seeing future tokens |
| Greedy decoding | Autoregressive token generation at inference |

**Config:** d_model=256, num_heads=8, num_layers=3, d_ff=512, dropout=0.1, max_seq_len=60

---

## Dataset

**Multi30k** (EN→DE) — 29,000 training pairs, 1,014 validation pairs.

- EN vocabulary: 7,704 tokens (min_freq=2)
- DE vocabulary: 9,597 tokens (min_freq=2)
- Special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`

---

## Training

- Optimizer: Adam (lr=1e-4, β₁=0.9, β₂=0.98, ε=1e-9)
- Loss: CrossEntropyLoss (ignoring PAD index)
- Gradient clipping: max_norm=1.0
- Epochs: 30, batch size: 64

---

## Results

| Metric | Value |
|--------|-------|
| Initial validation loss | 9.32 |
| Final validation loss | **2.01** |

### Sample translations (greedy decoding)

| English | German (predicted) |
|---------|-------------------|
| A man is walking a dog. | ein mann geht einen hund. |
| Two children are playing in the park. | zwei kinder spielen im park. |
| A woman is sitting on a bench near the water. | eine frau sitzt auf einer bank in der nähe des wassers. |

---

## How to run

Open `transformer.ipynb` in Kaggle or Colab (T4 GPU recommended). All dependencies install automatically. Training takes ~9 minutes on T4.

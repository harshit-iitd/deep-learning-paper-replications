# Project 2 — Vision Transformer from Scratch
### "An Image is Worth 16×16 Words" (Dosovitskiy et al., ICLR 2021)

Full ViT implementation in PyTorch, trained on CIFAR-10 from random initialization.

---

## Architecture

| Component | Implementation |
|-----------|---------------|
| Patch embedding | `Conv2D(in=3, out=768, kernel=16, stride=16)` → flatten → transpose |
| CLS token | Learnable `nn.Parameter(randn(1, 1, embed_dim))`, prepended before patches |
| Positional encoding | Learnable `nn.Parameter(randn(1, seq_len+1, embed_dim))` |
| Transformer encoder block | Pre-norm: LayerNorm → MultiHeadAttention → residual; LayerNorm → MLP (GELU) → residual |
| MLP head | Single `nn.Linear(embed_dim, num_classes)` on CLS token output |

**Config:** img_size=224, patch_size=16 → 196 patches, embed_dim=768, num_heads=8, depth=6, mlp_dim=3072

---

## Dataset

**CIFAR-10** — 50,000 training images, 10,000 test images, 10 classes. Images resized from 32×32 to 224×224.

---

## Training

- Optimizer: Adam (lr=1e-3)
- Loss: CrossEntropyLoss
- Epochs: 5 (intentionally limited)
- Batch size: 32, Hardware: Tesla T4

---

## Results

| Epoch | Train Accuracy |
|-------|---------------|
| 1 | 29.66% |
| 2 | 29.99% |
| 3 | 25.47% |
| 4 | 27.09% |
| 5 | ~26% |

### Why is accuracy low? — This is expected.

The original ViT paper explicitly states that ViT trained from scratch on small/medium datasets underperforms CNNs. The model lacks the inductive biases (locality, translation equivariance) that ConvNets have built in. ViT only outperforms CNNs when pretrained on large datasets (JFT-300M, ImageNet-21k) and then fine-tuned.

This experiment reproduces that exact finding. See Project 3 for what happens when you start from a pretrained checkpoint instead.

---

## How to run

Open `vit_scratch.ipynb` in Kaggle or Colab (T4 GPU recommended). Training takes ~50 minutes for 5 epochs.

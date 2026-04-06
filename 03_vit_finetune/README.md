# Project 3 — ViT-Base Fine-tuning on CIFAR-10
### "An Image is Worth 16×16 Words" — Section 4: Fine-tuning protocol

Fine-tuning `google/vit-base-patch16-224-in21k` (86M parameters) on CIFAR-10, reproducing the paper's fine-tuning methodology from Table 4.

---

## Setup

- **Base model:** `google/vit-base-patch16-224-in21k` (pretrained on ImageNet-21k, 86,389,248 parameters)
- **Classification head:** Single `Linear(768 → 10)`, zero-initialized (paper: "replace by a single, zero-initialized linear layer")
- **Why zero-init?** Ensures all logits = 0 at step 0 → uniform 1/10 probability per class → initial loss = −log(0.1) = 2.303. First gradient updates are driven entirely by pretrained features, not random head noise.

---

## Hyperparameters (from paper Table 4)

| Parameter | Value | Source |
|-----------|-------|--------|
| Total steps | 10,000 | Paper |
| Warmup steps | 500 | Paper |
| Base LR | 0.01 | Paper grid {0.001, 0.003, 0.01, 0.03} |
| Optimizer | SGD, momentum=0.9 | Paper |
| LR schedule | Cosine decay with linear warmup | Paper |
| Grad clipping | Global norm 1.0 | Paper |
| Image size | 224×224 (paper uses 384, T4 memory constraint) | Adapted |
| Batch size | 64 (paper uses 512) | Adapted |
| Normalization | ImageNet mean/std | HuggingFace preprocessor config |

---

## LR Schedule

```
Step 0–500:   Linear warmup  0 → 0.01
Step 500–10K: Cosine decay   0.01 → 0
```

---

## Results

### Overall

| Metric | Value |
|--------|-------|
| **Best test accuracy** | **98.80%** |
| Final test accuracy | 98.72% |
| Total training time | ~7.2 hours (Tesla T4) |

### Per-class accuracy (final)

| Class | Accuracy |
|-------|----------|
| ship | 99.8% |
| horse | 99.4% |
| deer | 99.2% |
| frog | 99.2% |
| airplane | 99.0% |
| bird | 99.0% |
| automobile | 98.9% |
| truck | 98.2% |
| cat | 97.4% |
| dog | 97.1% |

### Training progression

| Epoch | Train Acc | Test Acc |
|-------|-----------|----------|
| 1 | 96.28% | 98.00% |
| 2 | 98.85% | 98.51% |
| 3 | 99.45% | 98.56% |
| 5 | 99.92% | 98.67% |
| 10 | 100.00% | 98.80% ← best |
| 13 | 100.00% | 98.72% |

---

## Key observation — Scratch vs Fine-tuned

| Setting | Test Accuracy |
|---------|--------------|
| ViT from scratch (Project 2, 5 epochs) | ~27% |
| ViT fine-tuned from ImageNet-21k (this project) | **98.7%** |

The 72-point gap between the two settings is the empirical demonstration of the paper's central claim: ViT requires large-scale pretraining to be competitive.

---

## How to run

Open `vit_finetune.ipynb` in Kaggle (T4 GPU, ~7 hours) or use a Colab Pro A100 (~2 hours). Mixed-precision training (`torch.cuda.amp`) is enabled to fit on T4 memory.

# Deep Learning Paper Replications
Implementations of foundational deep learning papers from scratch, with training runs and verified results. Built as part of independent research preparation.
| # | Paper | Task | Key Result |
|---|-------|------|------------|
| 1 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) | EN→DE Translation (Multi30k) | Val loss 2.01 (from 9.32 at init), correct translations on held out sentences |
| 2 | [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020) | CIFAR 10 Classification | ~27% train acc (ViT from scratch, 5 epochs — data hungry by design) |
| 3 | ViT Base Fine tuning (same paper, Section 4) | CIFAR 10 Classification | **98.7% test accuracy** (pretrained ViT Base/16, 10K steps) |
| 4 | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021) | Image Text Contrastive Learning (Flickr8k) | Architecture verified, poor zero shot results (expected at 50,000× data gap) |
---
## Why these papers?
These four projects together cover the full arc of the modern Transformer paradigm:

**Project 1** establishes the core mechanism — self attention, positional encoding, encoder decoder architecture — in the sequence to sequence setting where Transformers were originally proposed.

**Project 2** shows what happens when you apply the same ideas to vision: patch tokenization, CLS token, learnable positional embeddings. Training from scratch on CIFAR 10 intentionally underperforms (ViT requires large scale pretraining — this is a finding, not a bug).

**Project 3** shows the correct way to use ViT: fine tune a pretrained checkpoint. The jump from ~27% (scratch) to 98.7% (fine tuned) makes the pretraining hypothesis viscerally concrete.

**Project 4** connects vision and language in a shared embedding space. CLIP uses a dual encoder architecture (ViT for images, causal Transformer for text) trained with a contrastive objective. Building it from scratch on Flickr8k demonstrates the architecture faithfully while making it clear why CLIP's results depend on massive scale (400M pairs, batch size 32,768). The poor results on 8K images are the lesson, not a failure.
---
## Setup
```bash
git clone https://github.com/[yourusername]/deep-learning-paper-replications
cd deep-learning-paper-replications
pip install -r requirements.txt
```
Each subfolder has its own README with notebook specific instructions. All notebooks are self contained and runnable on a free Kaggle/Colab T4 GPU.
---
## Hardware
All experiments run on a single **NVIDIA Tesla T4** (free tier on Kaggle/Colab).
| Project | Training time |
|---------|--------------|
| Transformer (30 epochs, Multi30k) | ~9 minutes |
| ViT from scratch (5 epochs, CIFAR 10) | ~50 minutes |
| ViT fine tuning (10K steps, CIFAR 10) | ~7.2 hours |
| Mini CLIP (30 epochs, Flickr8k) | ~2 hours |
---
## References
1. Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.
2. Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale." ICLR 2021.
3. Radford, A., et al. "Learning transferable visual models from natural language supervision." ICML 2021.
4. HuggingFace: `google/vit-base-patch16-224-in21k`

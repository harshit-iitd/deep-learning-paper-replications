# Mini CLIP: Learning Transferable Visual Models From Natural Language Supervision

A from scratch PyTorch implementation of [CLIP (Contrastive Language Image Pre training)](https://arxiv.org/abs/2103.00020) by Radford et al. (2021), scaled down to train on the Flickr8k dataset (~8,000 images).

## Architecture

Every architectural detail follows the original paper (Sections 2.3 to 2.5), with dimensions reduced to fit a small dataset and single GPU training.

### Paper vs Mini CLIP

| Component | Paper (ViT L/14) | Mini CLIP |
|---|---|---|
| Image encoder | ViT L/14, 768 dim, 24 layers, 16 heads | ViT Tiny/16, 256 dim, 6 layers, 4 heads |
| Text encoder | 63M params, 512 dim, 12 layers, 8 heads | ~11M params, 256 dim, 4 layers, 4 heads |
| Embedding dim | 768 | 256 |
| Batch size | 32,768 | 128 |
| Dataset | WIT (400M image text pairs) | Flickr8k (~40K image caption pairs) |
| Max sequence length | 76 | 76 |
| Training GPUs | 256 to 592 V100s | 1 GPU |
| Total parameters | ~428M | ~16M |

### Components implemented from the paper

**Image encoder (Vision Transformer):** Patch embedding (16×16), [CLS] token, positional embeddings, an extra LayerNorm before the transformer (paper §2.4), and post LayerNorm on the [CLS] output.

**Text encoder (Causal Transformer):** GPT 2 style with masked (causal) self attention, BPE tokenization (via HuggingFace `bert base uncased`), [EOS] token extraction as the text feature representation, followed by LayerNorm.

**Linear projections:** No non linear projection head. Only learned linear maps (W_i, W_t) into the shared embedding space, as specified in §2.3.

**Learned temperature:** Log parameterized scalar τ initialized to 0.07, clamped to prevent logit scaling beyond 100 (§2.5).

**Contrastive loss:** Symmetric cross entropy over the cosine similarity matrix of all N×N image text pairings in a batch (Figure 3 pseudocode).

**AdamW optimizer:** Decoupled weight decay (0.2) applied to all weights except biases and LayerNorm parameters (§2.5).

**Cosine LR schedule:** Linear warmup (200 steps) followed by cosine decay to zero (§2.5).

## Dataset

[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) consists of 8,091 images each with 5 human written captions (40,455 image caption pairs total).

```
Flickr8k/
├── Images/          # 8,091 JPEG images
└── captions.txt     # CSV: image_name, caption
```

## Training

```python
from mini_clip import train_clip

model, history = train_clip()
```

Training logs loss, learned temperature (τ), and learning rate at each epoch. The model checkpoint is saved as `mini_clip.pt`.

## Inference

### Image to best caption

```python
model.eval()

image_tensor = transform(Image.open("photo.jpg").convert("RGB")).unsqueeze(0).to(device)

captions = ["a dog in the park", "a car on the road", "a sunset over the ocean"]
tokens = tokenizer(captions, padding='max_length', truncation=True, max_length=76, return_tensors="pt")

with torch.no_grad():
    image_emb = model.encode_image(image_tensor)
    text_emb = model.encode_text(tokens['input_ids'].to(device), tokens['attention_mask'].to(device))
    similarities = (image_emb @ text_emb.T).squeeze(0)
    probs = F.softmax(similarities * torch.exp(model.log_temperature), dim=0)

best = captions[probs.argmax()]
```

### Caption to best image

```python
images = torch.stack([transform(Image.open(p).convert("RGB")) for p in image_paths]).to(device)
tokens = tokenizer("a dog in the park", padding='max_length', truncation=True, max_length=76, return_tensors="pt")

with torch.no_grad():
    image_emb = model.encode_image(images)
    text_emb = model.encode_text(tokens['input_ids'].to(device), tokens['attention_mask'].to(device))
    similarities = (text_emb @ image_emb.T).squeeze(0)
    probs = F.softmax(similarities * torch.exp(model.log_temperature), dim=0)

best = image_paths[probs.argmax()]
```

## Results and Limitations

**The model produces poor zero shot results.** This is expected and not a code bug. The reasons are fundamental to how contrastive learning works at scale:

### 1. Data scale (50,000× gap)

The original CLIP was trained on **400 million** image text pairs (the WIT dataset). Mini CLIP trains on **8,000 images** (~40K captions). Contrastive learning relies on seeing massive diversity of visual concepts paired with natural language. 8K images simply cannot cover the breadth of concepts needed for the model to learn generalizable image text associations.

### 2. Batch size (256× gap)

CLIP uses a batch size of **32,768**. Each image in a batch is contrasted against 32,767 negative texts. This enormous number of negatives is critical for the contrastive objective to produce useful gradients. Mini CLIP uses a batch size of **128**, meaning only 127 negatives per positive pair. With so few negatives, the model can achieve low loss without learning truly discriminative representations. It only needs to distinguish an image from 127 alternatives instead of 32,767.

### 3. Model capacity (27× gap)

Mini CLIP has **16M parameters** versus the original's **~428M**. The smaller model has less capacity to learn the complex multi modal mapping between vision and language. However, this is actually the least important factor. Even a 428M model would fail with only 8K training images.

### 4. Compute (orders of magnitude gap)

The largest CLIP model (RN50x64) trained for **18 days on 592 V100 GPUs**. The ViT L/14 trained for **12 days on 256 V100 GPUs**. Mini CLIP trains for 30 epochs on a single GPU in minutes. The total FLOPS difference is roughly 4 to 5 orders of magnitude.

### 5. Data diversity

Flickr8k images are predominantly outdoor scenes, people, and animals. The model never sees medical images, satellite photos, diagrams, food close ups, or thousands of other visual categories. CLIP's WIT dataset was specifically constructed with 500,000 search queries to cover the broadest possible range of visual concepts.

### Why this project still matters

The purpose of this project is **educational**, to understand CLIP's architecture by building it from scratch, not to replicate its results. Every component (ViT with the extra pre LayerNorm, causal text encoder, linear projections, learned temperature, symmetric contrastive loss, AdamW with selective weight decay, cosine schedule) is implemented exactly as described in the paper. The architecture is correct; only the scale is different.

## Requirements

```
torch
torchvision
transformers
tqdm
Pillow
```

## References

Radford, A., Kim, J.W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.

Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.*

Vaswani, A., et al. (2017). *Attention Is All You Need.*

## License

This project is for educational purposes only.

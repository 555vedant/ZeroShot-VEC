# Visual–Emotion Coherence Modeling

A zero-shot multimodal system that measures how well a painting aligns with a given emotion — returning a coherence score between 0 and 1.

---

## Overview

Traditional emotion classification systems are limited to a fixed set of labels. This system takes a different approach — it maps paintings and emotion words into a shared embedding space, enabling it to score any image–emotion pair and generalize to emotions it has never seen before, without retraining.

---

## Tech Stack

| Component             | Tool                     |
|-----------------------|--------------------------|
| Vision + Text Encoder | OpenAI CLIP (`ViT-B/32`) |
| Framework             | PyTorch                  |
| Language              | Python 3.10+             |
| Datasets              | WikiArt, ArtEmis         |

---

## Data

```
data/
├── wikiart/       # Painting images
└── artemis/       # Emotion annotations
```

- **WikiArt** — painting images across styles and artists
- **ArtEmis** — human-annotated emotional responses to paintings

Together they provide the image–emotion pairs used for training.

---

## How It Works

The system is built on a dual-encoder architecture using a pre-trained CLIP model.

1. An **image encoder** converts a painting into a feature vector
2. A **text encoder** converts an emotion word into a feature vector
3. Both vectors are passed through **projection layers** into a shared embedding space
4. **Cosine similarity** is computed between the two vectors
5. A **coherence score** between 0 and 1 is returned

Higher scores indicate stronger alignment between the painting and the emotion.

---

## Training

The model is trained using **transfer learning**:

- Start with pre-trained CLIP weights
- Freeze the encoder layers initially (linear probing)
- Train only the projection layers
- Optionally unfreeze deeper layers for fine-tuning

This preserves general visual and language knowledge while adapting the model to the emotion-matching task.

**Loss function** — symmetric contrastive loss over batches of image–emotion pairs. Correct pairs are pulled together in the embedding space; incorrect pairs are pushed apart. The final loss is the average of image-to-text and text-to-image losses.

---

## Zero-Shot Capability

During training, the model sees a fixed set of emotions such as joy, sadness, anger, fear, and awe. At inference, it can score unseen emotions like `serenity`, `loneliness`, or `melancholy` by encoding them into the same embedding space — no retraining required.

This works because the model learns the underlying meaning of emotions rather than memorizing fixed labels.

---

## Output

| Output           | Description                                      |
|------------------|--------------------------------------------------|
| Coherence score  | Float between 0 and 1 for any image–emotion pair |
| Ranked retrieval | Top-k paintings sorted by emotional alignment    |
| Zero-shot        | Works on emotions not seen during training       |

---


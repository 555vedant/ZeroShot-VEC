# Zero-Shot Evaluation and Fine-Tuning in CLIP

## 1. Why is it called "Zero-Shot"?
In machine learning, **"zero-shot"** means testing a model on categories or labels it has **never seen during training**. 
In this project, the script splits the dataset's emotion keywords into two groups: *seen* emotions (used to train the model) and *holdout/unseen* emotions. During evaluation, the model is asked to match images to the unseen emotions. Since the fine-tuned CLIP model has had "zero" training examples for these specific emotion categories, it has to rely on its general understanding of text and image relationships to guess right. 

### How `zero-short-evaluation.py` Works (Key Points)
1. **Creates a Strict Holdout Split:** The `_build_zero_shot_eval_dataset` function looks at all available data and separates out specific emotions that weren't used in training. It filters the dataset so that only records containing these new, unseen emotions remain.
2. **Loads the Fine-Tuned Model:** It fetches your fine-tuned CLIP model weights from the saved checkpoint.
3. **Generates Text & Image Embeddings:** It formats the unseen emotions into text prompts (e.g., "An image feeling [emotion]") and converts them into embeddings. It does the same for the test images.
4. **Ranking & Matching (Retrieval):** The script compares one image embedding against *all* the unseen emotion text embeddings. It ranks the emotions based on how closely they match the image to calculate Hit@1, Hit@3, and MRR.
5. **Positive vs Negative Pair Scoring:** The script individually scores a "positive" text pair (true unseen emotion) and a "negative" text pair (wrong emotion) to calculate binary classification metrics like Accuracy, F1-Score, and AUROC.

---

## 2. CLIP Training vs. Our Fine-Tuning Approach
Original CLIP *was* trained on image-text pairs (about 400M from the internet). Here is why the fine-tuning approach works:
* **The Core Mechanism is the Same:** Both use **Contrastive Learning**. They push embeddings of *matching* pairs closer together and *mismatched* pairs further apart.
* **Domain Adaptation:** Base CLIP learned general concepts. Fine-tuning teaches it a specialized vocabulary (art and emotions).
* **Zero-Shot Transfer:** Because CLIP learns general language-visual relationships rather than memorizing exact pairs, it can guess new, unseen emotions.

### Utilizing Explicit Positive/Negative Pairs
Original CLIP relies on "in-batch negatives" (assuming every *other* image in a batch is a negative). Because emotions are subjective and overlapping, random in-batch negatives can be problematic (e.g., "happy" and "joyful" might be treated as negatives).
We use explicit positive and negative pairs:
1. **Explicit Feed:** Feed the model an image and a specific text prompt.
2. **Similarity Score:** Calculate the dot product (similarity score).
3. **Binary Loss Evaluation:** Use Binary Cross Entropy (BCE) loss against an exact label (1 for match, 0 for mismatch).
4. **Weight Update:** If 1, weights update to push embeddings closer. If 0, weights update to push them apart.

---

## 3. How the Batch-Wise Loss Works (`matching_bce_loss`)
```python
def matching_bce_loss(pos_logits, neg_logits):
    pos_targets = torch.ones_like(pos_logits)
    neg_targets = torch.zeros_like(neg_logits)

    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_targets)
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets)  
    return 0.5 * (pos_loss + neg_loss)
```
This operates **batch-wise**:
1. **Inputs:** If batch size is 32, `pos_logits` has 32 scores (image vs correct emotion), and `neg_logits` has 32 scores (image vs wrong emotion).
2. **Targets:** `ones_like` creates 32 ones (`[1, 1...]`). `zeros_like` creates 32 zeros (`[0, 0...]`).
3. **Loss Calculation:** `F.binary_cross_entropy_with_logits` compares the 32 positive scores against the 32 ones and gets the average error for the batch. It does the same for negatives.
4. **Final Output:** It averages the positive and negative loss for a single backpropagation step.

---

## 4. How the Evaluation is Calculated (Mathematically)
The evaluation tests the model in two different ways.

### A. The Ranking Test (Hit@1, Hit@3, MRR)
Testing the model like a multiple-choice question:
* **Step 1:** Get embeddings for ALL unseen emotions.
* **Step 2:** Get the embedding for a test image.
* **Step 3:** Multiply image embedding by all text embeddings to score every emotion.
* **Step 4:** Rank emotions from highest to lowest. (The model is blind to the true answer).
* **Metrics:** 
  * **Hit@1:** Was the true emotion #1?
  * **Hit@3:** Was the true emotion in the Top 3?
  * **MRR:** Mean Reciprocal Rank (1/rank of the true emotion).

### B. The Binary Classification Test (Accuracy, F1, AUROC)
Testing the model like True/False questions. We **do not combine** positive/negative probabilities. They are treated as separate questions.

**Example Setup:**
* **Image:** Dark, rainy painting.
* **Pair A (Positive):** Image + "Sad" ➔ Hidden Label = 1
* **Pair B (Negative):** Image + "Happy" ➔ Hidden Label = 0

**The Process:**
1. **Model Predictions:** The model independently scores Pair A (e.g., 0.85) and Pair B (e.g., 0.20).
2. **Thresholding (e.g., 0.5):** 
   * Pair A: 0.85 > 0.5 ➔ Model guesses 1 (True Positive)
   * Pair B: 0.20 <= 0.5 ➔ Model guesses 0 (True Negative)
3. **Calculating Metrics (Across all pairs in the test set):**
   * **Accuracy:** (TP + TN) / Total Pairs
   * **Precision:** TP / (TP + FP)
   * **Recall:** TP / (TP + FN)
   * **F1-Score:** 2 * (Precision * Recall) / (Precision + Recall)
   * **AUROC:** Sorts all pairs strictly by the model's raw score. Measures if the model consistently gives higher scores to Positive Pairs (1s) than Negative Pairs (0s) without needing a 0.5 threshold.

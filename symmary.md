# CLIPFit Accuracy Investigation and 70%+ Improvement Plan

## Implementation status (2026-11-05)

The repository now uses visual LoRA on the final four vision-block
query/value projections, with the text encoder and base visual weights frozen.
It also uses weighted multi-negative ranking, dynamic valid-label-aware
negatives, cosine learning-rate decay, prompt ensembles, and image-disjoint
zero-shot training/evaluation. Existing
`pairs.json` is preserved and is no longer regenerated automatically.

The improved training and zero-shot commands were attempted, but this
environment does not contain the WikiArt image root referenced by
`pairs.json`, so both stop before model training/evaluation with
`No valid image paths could be resolved from pairs.json`. CUDA is also
unavailable here. No new checkpoint or metrics can honestly be reported
from this environment.

## Short answer

The most likely root cause is **not that CLIPFit is intrinsically bad**. The current pipeline combines:

1. A small, noisy and highly imbalanced affective dataset.
2. A **pairwise BCE objective** that is not the same as CLIP's original batch contrastive objective.
3. A strong CLIPFit knowledge-distillation penalty that keeps the student close to the frozen base model.
4. Random single-negative sampling, which gives weak and unstable supervision.
5. A zero-shot split that is not fully strict because the same image can remain in training with its other emotion label.
6. Accuracy measured at a fixed sigmoid threshold of `0.5`, even though the score is a temperature-scaled cosine similarity and is not calibrated as a probability.

Therefore, the reported ~50% accuracy is likely a mixture of **under-adaptation, noisy labels, weak negatives, class imbalance, and a mismatched metric**. Full/domain fine-tuning can perform better because it has enough capacity to learn the dataset's visual-emotion shortcuts; CLIPFit deliberately restricts that capacity.

> A guaranteed 70% cannot be promised without measuring a fixed, leakage-free test set. The practical target should be 70%+ on a clearly defined validation/test protocol, with AUROC, macro-F1 and ranking metrics reported alongside accuracy.

## 1. What the repository currently does

### Model

- Base model: `openai/clip-vit-base-patch32` (`ViT-B/32`).
- Image and text encoders produce normalized embeddings.
- Similarity is a cosine dot product multiplied by `1 / TEMPERATURE` for training/evaluation pair logits.
- `FINE_TUNING_STRATEGY = "clipfit"`.
- In CLIPFit mode, all parameters are frozen, then only:
  - image-side `LayerNorm` parameters;
  - text FFN `fc2.bias` values;
  - projection heads and `logit_scale` through the existing unfreeze logic.
- A frozen copy of the original model is used as a teacher.

Relevant implementation: [`src/model.py`](./src/model.py), [`utils/config.py`](./utils/config.py).

### Data

The checked-in `webapp/pairs.json` contains:

| Quantity | Current value |
|---|---:|
| Image-text pairs | 39,644 |
| Unique images | 19,822 |
| Labels per image | exactly 2 |
| Emotion labels | 9 |
| Most frequent label | awe: 10,594 (26.7%) |
| Least frequent label | disgust: 447 (1.1%) |
| Second-largest label | anger: 8,370 (21.1%) |
| Two-label images | 19,822 / 19,822 |

The preprocessing script samples the ArtEmis histogram and keeps the top two emotions. Ties at the top-k boundary are retained. The result is a multi-label, subjective dataset, but the training loop treats each selected label as an independent positive pair.

Relevant implementation: [`src/preprocess.py`](./src/preprocess.py), [`src/dataset.py`](./src/dataset.py).

### Current training objective

For each positive image/emotion pair:

```text
positive = BCE(logit, 1)
negative = BCE(logit, 0)
ranking = softplus(0.2 + negative_logit - positive_logit)
loss = 0.5 * (positive + negative) + 0.5 * ranking
```

In CLIPFit mode, the loss also includes:

```text
loss += CLIPFIT_KD_WEIGHT * embedding_distillation_loss
```

The current configuration uses:

| Setting | Value |
|---|---:|
| Batch size | 64 |
| Epochs | 25 |
| Learning rate | 1e-5 |
| Backbone multiplier | 0.2 |
| Effective backbone LR | 2e-6 |
| Temperature | 0.07 |
| Ranking loss weight | 0.5 |
| KD weight | 1.0 |
| Validation split | 20% by image |
| Early stopping patience | 5 |

Relevant implementation: [`src/train.py`](./src/train.py), [`utils/config.py`](./utils/config.py).

### Current evaluation

The evaluator:

- samples one negative emotion for each positive pair;
- converts logits to sigmoid scores;
- reports `Accuracy@0.5`, precision, recall, F1 and AUROC;
- separately reports Hit@1, Hit@3 and MRR.

The zero-shot evaluator holds out two labels using seed 42: `contentment` and `fear`. It then evaluates only those labels.

Relevant implementation: [`src/evaluate.py`](./src/evaluate.py), [`src/zero-short-evaluation.py`](./src/zero-short-evaluation.py).

## 2. Root-cause analysis

### Root cause A — CLIPFit is being asked to learn a larger domain shift than its update budget can support

CLIPFit intentionally updates very few parameters. That is useful for preserving general CLIP knowledge and reducing overfitting, but emotion recognition in paintings requires changing visual features and text semantics together. The current model must learn:

- art-specific visual cues;
- palette, composition and style cues;
- subjective emotion associations;
- a new prompt/task distribution;
- class imbalance corrections.

With the image transformer mostly frozen and the text encoder almost entirely frozen, the model may not have enough capacity to move the two modalities into the right local geometry. The distillation term reinforces this: it penalizes exactly the representation changes needed for domain adaptation.

**Why full fine-tuning can win:** it can learn the dataset-specific mapping. **Why it may overfit:** it can also memorize style, artist, or image-level shortcuts.

### Root cause B — The objective is mismatched with the CLIP architecture

Original CLIP is trained with a symmetric in-batch contrastive loss: every image is compared with every text in the batch, and the correct diagonal is optimized against all other candidates. This project instead computes independent positive and negative pair logits and uses BCE.

Consequences:

- only one sampled negative is used per positive;
- the model does not see a full candidate-label competition;
- the negative distribution changes by random seed and epoch;
- BCE on a temperature-scaled cosine score does not make `sigmoid(score) >= 0.5` a meaningful probability boundary;
- the objective does not directly optimize Hit@1 or multiclass label ranking.

The extra margin term helps separation but does not restore the full contrastive structure.

### Root cause C — One random negative is too weak and sometimes semantically wrong

`sample_negative_emotion()` chooses from the five most frequent eligible labels. This has two problems:

1. It under-samples rare emotions such as disgust and excitement.
2. It can select a semantically plausible emotion that is actually a valid human annotation for the image unless all image labels are correctly excluded from the candidate pool.

Because each image has two positive labels, a negative chosen outside the current record but inside the image's other-label set would be a false negative in a naive implementation. The current dataset-level set normally prevents that, but the objective still sees only one negative and does not model all valid labels jointly.

### Root cause D — Label imbalance changes what “accuracy” rewards

`awe`, `anger`, and `amusement` account for about 62.5% of all pairs; `disgust` accounts for about 1.1%. A model that favors common emotions can look acceptable on micro accuracy while failing minority emotions. Conversely, a strict zero-shot holdout can be dominated by difficult or low-frequency labels.

The current split holds out `contentment` and `fear`, not a balanced or repeated set of holdouts. One two-label split is not enough evidence for general zero-shot performance.

### Root cause E — The labels are subjective and the top-2 reduction loses information

ArtEmis records human affective responses. Multiple emotions for one painting are not necessarily contradictions. Selecting only the top two labels discards:

- the annotation histogram;
- annotator agreement;
- uncertainty;
- the strength difference between the first and second label.

The current BCE loss treats every selected label as equally certain and every sampled negative as definitely false. That creates label noise, especially for visually and semantically related emotions.

### Root cause F — The “strict zero-shot” split has image leakage

Training filters records by `allowed_emotions=seen_emotions`, while evaluation filters records by holdout emotions. If an image has one seen label and one holdout label, its seen-label record enters training and its holdout-label record enters zero-shot evaluation.

Since every current image has two labels, this is a real risk rather than a theoretical edge case. The evaluation is therefore not strict image-level zero-shot. It can either inflate results through visual memorization or create a confusing train/evaluation mismatch.

Fix: assign images to seen/holdout groups first, then remove every image appearing in the holdout group from training.

### Root cause G — Fixed threshold 0.5 is not a reliable accuracy measurement

The score is:

```text
cosine_similarity / 0.07
```

and then `sigmoid()` is applied. A threshold of 0.5 means logit 0, not “50% probability that the pair is a true annotation.” The correct threshold depends on candidate set, class balance, calibration, and the task.

For ranking, compare candidate emotions directly and report Hit@1/MRR. For binary pair scoring, tune a threshold on validation only and report test performance separately. Always include AUROC and macro-F1.

### Root cause H — Checkpoint selection does not optimize the target metric

The best checkpoint is selected using validation BCE. The user target, however, is accuracy/quality and the zero-shot evaluator uses ranking over unseen labels. The lowest BCE checkpoint is not guaranteed to have the best Hit@1, macro-F1, or calibrated accuracy.

## 3. Recommended modification plan

### Phase 0 — Make the experiment trustworthy first

1. **Create image-disjoint splits.**
   - Split by image ID, not pair record.
   - For zero-shot evaluation, remove every holdout image from training.
   - Keep train, validation and test image IDs frozen in a manifest.

2. **Use repeated holdouts.**
   - Run at least five fixed seeds.
   - Report mean and standard deviation for Hit@1, MRR, macro-F1 and AUROC.
   - Do not select the holdout because it gives the best result.

3. **Add per-emotion metrics.**
   - support;
   - precision/recall/F1;
   - one-vs-rest AUROC;
   - confusion matrix for seen-label classification;
   - Hit@1 and MRR for label ranking.

4. **Separate model selection from final test evaluation.**
   - tune threshold/temperature on validation;
   - run the final test exactly once;
   - never tune on the reported test set.

### Phase 1 — Correct the data target

Prefer preserving the histogram as a soft target:

```text
target(image, emotion) = normalized annotation frequency
```

Use one of these alternatives:

- **Soft multi-label BCE** with the histogram probability.
- **Weighted multilabel contrastive loss** where all annotated emotions are positives.
- **Label smoothing** for hard labels, for example positive target 0.9 and negative target 0.1.
- Keep “something_else” as a valid class, but evaluate it separately because it is semantically broad.

Use class-balanced sampling or inverse-frequency loss weights, with a cap so the rarest labels do not dominate. A balanced batch should contain a controlled number of examples per emotion.

### Phase 2 — Replace one-negative BCE with a CLIP-compatible objective

Recommended primary experiment:

1. Encode a batch of images and all candidate text prompts.
2. Compute the image-to-text similarity matrix.
3. Use supervised multi-positive contrastive loss:
   - all annotated emotions for an image are positive;
   - other emotions are negatives;
   - positives are masked rather than incorrectly pushed apart.
4. Add the symmetric text-to-image term if the batch contains unique images.

For a simpler first test, use a multiclass soft-label cross-entropy over the nine emotions. This directly optimizes the ranking problem and is easier to debug than pairwise BCE.

Hard-negative mining can be added after the baseline:

- choose the highest-scoring wrong emotion from the current batch;
- do not mine from the same image's valid label set;
- keep random negatives mixed in to avoid collapse.

### Phase 3 — Compare adaptation strategies systematically

Run the same frozen splits and metric script for these controlled baselines:

| Experiment | Trainable parameters | Purpose |
|---|---|---|
| Zero-shot base CLIP | none | True baseline |
| Linear probe | projection/head only | Tests whether labels are already linearly separable |
| CLIPFit, KD off | CLIPFit parameters | Tests whether KD is blocking adaptation |
| CLIPFit, KD 0.1 | CLIPFit parameters | Lower preservation pressure |
| LoRA adapters | attention/MLP low-rank adapters | More capacity with parameter efficiency |
| Top-layer fine-tuning | last 2–4 vision/text blocks | Controlled domain adaptation |
| Full fine-tuning, low LR | all parameters | Upper-capacity reference |

The most important immediate ablation is **CLIPFit with `CLIPFIT_KD_WEIGHT=0` or 0.1**. If it improves substantially, the frozen teacher is preventing the required domain shift. If it does not, the loss/data/split is the dominant problem.

Suggested starting ranges:

- CLIPFit: `1e-5` to `5e-5`.
- LoRA/top-layer fine-tuning: `1e-6` to `1e-5` for pretrained layers.
- Full fine-tuning: `1e-7` to `2e-6`, with warmup and cosine decay.
- KD weight: `{0, 0.05, 0.1, 0.5, 1.0}`.
- Temperature: learn CLIP's `logit_scale` or tune it on validation; do not assume `0.07` is optimal.

### Phase 4 — Improve image and text handling

- Compare prompts such as:
  - `a painting that evokes {emotion}`;
  - `a painting expressing {emotion}`;
  - `an artwork conveying {emotion}`;
  - `a {emotion} painting`.
- Average text embeddings over several prompt templates.
- Keep the prompt set fixed during evaluation.
- Use CLIP's native preprocessing consistently.
- Avoid aggressive color augmentation; restrained augmentation is appropriate because palette is meaningful, but validate whether mirroring is harmful for composition/subject semantics.
- Consider higher-resolution CLIP or a stronger vision backbone only after the data/loss protocol is corrected.

### Phase 5 — Select the right target metric

For an emotion-ranking product, prioritize:

1. Hit@1;
2. MRR;
3. macro-F1;
4. AUROC;
5. calibrated pair accuracy.

For a pair-coherence score, additionally report:

- reliability diagram / expected calibration error;
- Spearman correlation with annotation strength;
- calibration error per emotion;
- performance by painting style and artist, if metadata is available.

## 4. Recommended first implementation configuration

Do not change everything at once. The first high-value run should be:

```text
image-disjoint split: enabled
strict zero-shot image exclusion: enabled
loss: supervised multi-positive contrastive or soft-label multiclass CE
KD weight: 0.1
negative sampling: batch hard negatives + valid-label masking
class balancing: balanced sampler or capped inverse-frequency weights
prompt ensemble: 4 fixed templates
checkpoint selection: validation Hit@1 + macro-F1
test reporting: mean/std over 5 seeds
```

Then run the ablation matrix:

1. Current CLIPFit + current BCE.
2. CLIPFit + corrected split + corrected loss.
3. CLIPFit + corrected loss + KD 0.
4. Top-layer fine-tuning + corrected loss.
5. Full fine-tuning + corrected loss.
6. Best method + prompt ensemble and calibration.

This identifies whether the 50% result is caused mainly by:

- adaptation capacity;
- the loss;
- the split;
- the negative sampler;
- calibration;
- or label noise.

## 5. What not to do

- Do not claim 70% from a threshold tuned on the test set.
- Do not report only micro accuracy; it hides minority-label failure.
- Do not compare the current BCE accuracy directly with CLIP zero-shot Hit@1.
- Do not train on images that appear in the zero-shot evaluation set.
- Do not increase epochs first; that can make the current noisy objective overfit.
- Do not assume CLIPFit should outperform full fine-tuning on a domain-specific task; its main benefit is parameter efficiency and retention, not universal accuracy superiority.

## 6. Expected outcome

A realistic sequence is:

1. Correcting leakage and metrics may lower the headline number but makes it trustworthy.
2. Replacing one-negative BCE with multi-positive contrastive/soft-label training should improve ranking consistency.
3. Lowering or removing KD should recover adaptation capacity.
4. Class balancing and image-disjoint validation should improve minority emotions and macro-F1.
5. Top-layer or LoRA tuning is the likely best compromise if CLIPFit remains below target.
6. Full fine-tuning is the reference upper bound, but it requires stronger regularization and early stopping.

The highest-probability path to 70%+ is **not “more CLIPFit epochs.”** It is:

> clean image-disjoint evaluation + soft/multi-positive targets + CLIP-compatible ranking loss + weaker KD + controlled top-layer/LoRA adaptation + validation-based calibration.

## 7. References

1. Radford et al., **Learning Transferable Visual Models From Natural Language Supervision** (CLIP), ICML 2021.  
   https://arxiv.org/abs/2103.00020  
   https://proceedings.mlr.press/v139/radford21a.html

2. Li et al., **Vision-Language Model Fine-Tuning via Simple Parameter-Efficient Modification** (CLIPFit), EMNLP 2024.  
   https://arxiv.org/abs/2409.16718  
   https://aclanthology.org/2024.emnlp-main.797/

3. Achlioptas et al., **ArtEmis: Affective Language for Visual Art**, CVPR 2021.  
   https://arxiv.org/abs/2101.07396  
   https://openaccess.thecvf.com/content/CVPR2021/html/Achlioptas_ArtEmis_Affective_Language_for_Visual_Art_CVPR_2021_paper.html

4. Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**, ICLR 2022.  
   https://arxiv.org/abs/2106.09685

## Final diagnosis

The current 50% result is best explained by a **pipeline mismatch**, not a single broken line:

```text
subjective multi-label data
        + hard top-2 labels
        + severe imbalance
        + one random negative
        + BCE on CLIP logits
        + strong frozen-teacher KD
        + very limited CLIPFit capacity
        + fixed uncalibrated threshold
        + non-strict image-level zero-shot split
        = unstable and misleading accuracy
```

Fix the evaluation and target first, then compare CLIPFit, LoRA/top-layer tuning, and full fine-tuning under exactly the same protocol. That is the shortest reliable route to determining what can actually reach 70%+.

# Beyond Accuracy: A Multi-Pillar Clinical Trust Framework for Brain Tumor MRI Classification

Research code accompanying our IEEE submission proposing a four-pillar evaluation framework for medical imaging AI — because a model can be 94%+ accurate while still being dangerously overconfident on the cases it gets wrong.

**Paper status:** Submitted, IEEE, 2026 (not yet accepted/published)
**Authors:** Iqra Safdar, Zeeshan Raza, Malaika Arif — Department of Computer Science, COMSATS University Islamabad, Sahiwal Campus

---

## The Problem

Brain tumor MRI classifiers routinely report 94%+ accuracy. But accuracy alone doesn't tell you whether a model is safe to deploy clinically — it can hide two dangerous failure modes:

1. **Miscalibration** — the model's stated confidence doesn't match its actual reliability
2. **Silent high-confidence errors** — the model is *most* wrong exactly when it claims to be *most* sure

## The Framework

We propose evaluating models across four pillars instead of accuracy alone:

| Pillar | What it measures |
|---|---|
| **Discrimination** | Standard accuracy, precision, recall, F1 |
| **Calibration** | Expected Calibration Error (ECE) — does confidence match reality? |
| **High-Confidence Audit** | Of the model's most confident predictions, what fraction are silently wrong? |
| **Explainability** | Grad-CAM spatial auditing — does the model attend to anatomically plausible regions? |

These combine into the **Clinical Readiness Index (CRI)**:

```
CRI = 0.40·Accuracy + 0.25·(1−ECE) + 0.20·(1−HCE) + 0.15·Generalization
```

## Methodology

- **Dataset:** Public Brain Tumor MRI Dataset (Nickparvar et al.), 7,023 training + 1,600 test images, 4 classes (glioma, meningioma, notumor, pituitary), test set perfectly balanced at 400/class
- **Architecture:** MobileNetV2 and ResNet50, ImageNet-pretrained, final 50 layers unfrozen for fine-tuning
- **Class imbalance handling:** Focal Loss (γ=2.0, α=0.25)
- **Training:** Adam optimizer (lr=1e-4), early stopping (patience=10), ReduceLROnPlateau, max 30 epochs, seed 42
- **Statistical rigor:** Multi-seed validation (seeds 42, 789, 999), bootstrap 95% confidence intervals (10,000 resamples)
- **Explainability audit:** Grad-CAM heatmaps with quantitative edge-bias metrics

Full details in the paper (Section III).

## Repository Contents

```
train_mobilenetv2_paper_exact.py   — training script matching the paper's exact protocol
generate_gradcam.py                — Grad-CAM heatmap generation for misclassified cases
generalization_test.py             — cross-dataset (Figshare) zero-shot transfer evaluation
```

## Reproducing Results

```bash
# 1. Download the Brain Tumor MRI Dataset (Nickparvar et al.) from Kaggle
# 2. Run train_mobilenetv2_paper_exact.py in a GPU environment (Colab/Kaggle recommended)
# 3. Outputs: y_true.npy, y_pred.npy, y_pred_probs.npy, and the trained model
```

**Reproducibility note:** independent reruns of this protocol will not reproduce the paper's exact figures to the decimal — Table V in the paper documents seed-to-seed variance as an expected, normal property of the training process, not an error.

## The Audit Tool

The framework described here is implemented as a runnable auditing tool: **[MedTrust-Audit](https://github.com/malaikaarif/MedTrust-Audit)** — a FastAPI dashboard that computes the CRI and all four pillars from a model's saved predictions, with a working Grad-CAM explainability section and support for auditing externally-provided prediction arrays.

## Citation

If referencing this work:

> Safdar, I., Raza, Z., Arif, M. "Beyond Accuracy: A Multi-Pillar Clinical Trust Framework for Brain Tumor MRI Classification." Submitted, IEEE, 2026.

## License

MIT

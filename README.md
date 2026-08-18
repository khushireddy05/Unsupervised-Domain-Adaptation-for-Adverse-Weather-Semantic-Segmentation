# Unsupervised Domain Adaptation for Adverse Weather — Semantic Segmentation

This project studies **unsupervised domain adaptation (UDA)** for semantic segmentation under adverse weather and low-visibility conditions. It uses labeled **Cityscapes** images as the source domain and **ACDC** images as the unlabeled target domain, with experiments progressing from source-only baselines to DAFormer-style adaptation.

## Project overview

Semantic-segmentation models trained on clear daytime street scenes often degrade in rain, fog, snow, and nighttime conditions. This project evaluates that domain gap and explores adaptation methods that improve target-domain predictions without using target-domain labels during training.

The work is organized into three phases:

1. **Source-domain baseline** — train and evaluate segmentation models on Cityscapes.
2. **Domain-gap analysis** — evaluate the source-trained model on ACDC adverse-weather scenes.
3. **Unsupervised domain adaptation** — apply a DAFormer-style teacher–student adaptation pipeline using labeled Cityscapes and unlabeled ACDC images.

Performance is evaluated primarily with mean Intersection over Union (**mIoU**) and per-class IoU, alongside qualitative segmentation predictions.

## Repository structure

```text
.
├── phase-1.ipynb                         # Source-domain baseline experiments
├── phase-2.ipynb                         # Target-domain evaluation/domain gap
├── Phase 1/
│   ├── phase1-cityscape.ipynb
│   ├── phase1-report.pdf
│   └── Results/                              # Baseline plots and predictions
├── Phase 2/
│   ├── phase2-acdc.ipynb
│   └── Results/                              # Domain-gap plots and predictions
├── Phase 3/
│   └── phase3-daformer with results.ipynb   # UDA experiments and results
├── Proposal.pdf
└── literature-validation-report.docx
```

Datasets and model checkpoints are intentionally excluded from Git because of their size and licensing/distribution requirements.

## Data

Download the datasets from their official sources and arrange them in the locations expected by the notebooks:

- **Cityscapes**: `Phase 1/Data/`
- **ACDC**: `Phase 2/Data/`

You may need to update dataset-root variables in the notebooks if you use a different directory layout.

## Running the experiments

Use a Python environment with Jupyter and the deep-learning/data-science packages imported by the notebooks (including PyTorch, Transformers, NumPy, pandas, Matplotlib, Pillow, and scikit-learn).

Run the notebooks in phase order:

1. `Phase 1/phase1-cityscape.ipynb`
2. `Phase 2/phase2-acdc.ipynb`
3. `Phase 3/phase3-daformer with results.ipynb`

A CUDA-capable GPU is recommended for training and adaptation.

## Outputs

The tracked result folders include:

- training curves;
- per-class IoU comparisons;
- source-to-target domain-gap visualization; and
- qualitative semantic-segmentation predictions.

## Notes

- Cityscapes and ACDC remain subject to their respective licenses.
- Checkpoint files (`.pth`, `.pt`, and `.ckpt`) are ignored and should be stored separately or published through a model/artifact hosting service.


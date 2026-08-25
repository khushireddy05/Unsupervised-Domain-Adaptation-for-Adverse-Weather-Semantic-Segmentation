# Unsupervised Domain Adaptation for Adverse Weather — Semantic Segmentation

This project studies **unsupervised domain adaptation (UDA)** for semantic segmentation under adverse weather and low-visibility conditions. It uses labeled **Cityscapes** images as the source domain and **ACDC** images as the unlabeled target domain, with experiments progressing from source-only baselines to DAFormer-style adaptation.

## Project overview

Semantic-segmentation models trained on clear daytime street scenes often degrade in rain, fog, snow, and nighttime conditions. This project evaluates that domain gap and explores adaptation methods that improve target-domain predictions without using target-domain labels during training.

The work is organized into three phases (Phase 3 covers everything from the initial self-training
pipeline through the ablation study, the tuned final model, and the proposal's weather-aware
masking extension — see **Current status** below for exactly what's done vs. still running):

1. **Source-domain baseline** — train and evaluate segmentation models on Cityscapes.
2. **Domain-gap analysis** — evaluate the source-trained model on ACDC adverse-weather scenes.
3. **Unsupervised domain adaptation** — a DAFormer-style teacher–student pipeline (self-training,
   BatchNorm freeze, feature-distance regularization, MIC), an ablation study isolating each
   component's contribution, a tuned final configuration, and a weather-aware masking extension
   (the proposal's own research question, not just DAFormer/MIC reproduction).

Performance is evaluated with mean IoU (**mIoU**), per-class IoU, pixel accuracy, and per-condition
(fog/rain/snow/night) breakdowns, alongside qualitative segmentation predictions.

## Current status

| Stage | Result | Status |
|---|---|---|
| Phase 1 — Cityscapes baseline | 73.18% mIoU (TTA) | Done |
| Phase 2 — ACDC zero-shot domain gap | 49.02% avg mIoU (−24.16 pts vs. source) | Done |
| Phase 3 — first combined self-training run (BN freeze + feature-distance + MIC, 20 epochs) | 53.27% avg mIoU; night regressed slightly below zero-shot (34.64% vs 35.35%) | Done |
| Phase 3 — ablation study (isolating BN freeze / feature-distance / MIC) | 48.09% → 49.71% → 51.50% (6 epochs each) | Done |
| Phase 3 — final tuned run (+ rare-class sampling, night-tuned augmentation, consistency loss, tuned threshold) | **54.19% avg mIoU** (fog 67.4 / rain 56.6 / snow 57.8 / night 34.9) — best result so far, night still marginally below its 35.35% zero-shot baseline | Done |
| Phase 3 — weather-aware masking extension (proposal's own research question, not generic MIC) | Weather-aware MIC beats generic MIC: +0.38 avg mIoU, **+2.18 on night specifically** (34.06% vs 31.88%, 5-epoch controlled test) | Done |
| Phase 3 — confidence-threshold sweep (0.90 / 0.95 / 0.968 / 0.985) | No strong overall trend; small, consistent effect favoring night at higher thresholds | Done |
| Phase 3 — fold weather-aware MIC into the *final* tuned config | 🔴 **Failed twice via CLI push** with `AcceleratorError: no kernel image is available for execution on the device` — needs a browser-triggered run with GPU T4 ×2 explicitly selected (see below) | 🔴 **Next action / where things are stuck** |
| Consolidated final report + presentation | Not started (deliberately deferred) | Pending |

**Where to pick this up:** `Phase 3/phase3-final-weather-aware.ipynb` (kernel
`khushireddymacha/phase-3-final-weather-aware`) has failed **twice** when triggered via the Kaggle
CLI (`kaggle kernels push`), both times with the same CUDA accelerator-compatibility error, within
minutes of starting. This is the same issue the `phase-3-extensions` run hit before it — CLI pushes
can't set the GPU accelerator type, and whatever Kaggle assigns by default right now isn't
compatible with the pre-installed PyTorch build. **The fix that worked last time:** open
https://www.kaggle.com/code/khushireddymacha/phase-3-final-weather-aware/edit in a browser,
Settings → Accelerator → **GPU T4 ×2** (explicitly, not the CLI default), then **Save Version →
Save & Run All (Commit)**. Do **not** re-push via CLI again for this kernel until it's succeeded at
least once with T4 ×2 — the code itself is not the problem, so debugging it further will waste
time.

Once run successfully, this is the last open experimental question — does weather-aware MIC push
night above its 35.35% zero-shot baseline once combined with rare-class sampling, night-tuned
augmentation, and the consistency loss? Everything else in the proposal's task list is done.

- **If the run is going or finished successfully:** check status with
  `kaggle kernels status khushireddymacha/phase-3-final-weather-aware`, then pull results with
  `kaggle kernels output khushireddymacha/phase-3-final-weather-aware -p "Phase 3/kaggle_output_final_weather_aware"`.
  Read `history.csv` under the downloaded `final_weather_aware_artifacts/final_weather_aware_tuned/`
  folder for per-epoch numbers, or check the log's stdout for the printed comparison against the
  known 54.19%/34.9% generic-MIC result.
- **If it shows `ERROR`:** pull the log the same way and check for
  `AcceleratorError: ... no kernel image is available for execution on the device`. If that's the
  error, it's not a code bug — see the *Running on Kaggle* gotcha below (open the notebook in the
  browser, explicitly select **GPU T4 ×2**, and re-run via "Save Version"). If it's a different
  error, that's a real bug to fix in `Phase 3/make_phase3_final_weather_aware_notebook.py` (edit
  that file, not the generated `.ipynb` directly, then regenerate).
- **Once you have a result:** update this table's row above with the actual numbers, and if
  weather-aware MIC wins, that becomes the project's final headline result — worth folding into
  `phase3-report.pdf` / `phase3-summary.pdf` and mentioning to the supervisor.

## Repository structure

```text
.
├── phase-1.ipynb                                          # Early Phase 1 draft
├── phase-2.ipynb                                          # Early Phase 2 draft
├── Phase 1/
│   ├── phase1-cityscape.ipynb
│   ├── phase1-report.pdf
│   └── Results/                                           # Baseline plots and predictions
├── Phase 2/
│   ├── phase2-acdc.ipynb
│   ├── phase2-report.pdf
│   └── Results/                                           # Domain-gap plots and predictions
├── Phase 3/
│   ├── phase3-daformer with results.ipynb                 # First combined run (53.27% avg mIoU)
│   ├── notebookf470b78dee.ipynb                           # Superseded: pre-bugfix ablation run, kept for history only
│   ├── make_phase4_notebook.py                            # Generates the ablation + final-tuned notebook
│   ├── phase4-daformer research and performance proofs.ipynb   # Ablations + final tuned run (source)
│   ├── phase 3 - FINAL.ipynb                              # Executed results: 54.19% avg mIoU (best)
│   ├── make_phase3_extensions_notebook.py                 # Generates the masking + threshold notebook
│   ├── phase3-extensions.ipynb                            # Source: weather-aware MIC vs generic + threshold sweep
│   ├── phase3-extensions with results.ipynb               # Executed results for the above
│   ├── make_phase3_final_weather_aware_notebook.py        # Generates the follow-up integration notebook
│   ├── phase3-final-weather-aware.ipynb                   # NOT YET RUN — next action, see Current status
│   ├── kernel-metadata.json / kaggle_push_final_weather_aware/  # Kaggle CLI push configs
│   ├── phase3-report.pdf                                  # Full narrative report (all of Phase 3)
│   ├── phase3-summary.pdf                                 # One-page attempt-by-attempt summary table
│   └── Results/                                           # Charts extracted/generated from the runs above
├── Proposal.pdf
└── literature-validation-report.docx
```

Datasets and model checkpoints are intentionally excluded from Git because of their size and licensing/distribution requirements (see `.gitignore` — `*.pth`/`*.pt`/`*.ckpt` and any `Data/` folder are never tracked, even under `Phase 3/kaggle_output/`, which also isn't tracked wholesale — only the small `history.csv` result files inside it are).

## Data

Download the datasets from their official sources and arrange them in the locations expected by the notebooks:

- **Cityscapes**: `Phase 1/Data/`
- **ACDC**: `Phase 2/Data/`

You may need to update dataset-root variables in the notebooks if you use a different directory layout.

## Running the experiments

Use a Python environment with Jupyter and the deep-learning/data-science packages imported by the notebooks (including PyTorch, Transformers, NumPy, pandas, Matplotlib, Pillow, and scikit-learn).

Run the notebooks in order:

1. `Phase 1/phase1-cityscape.ipynb`
2. `Phase 2/phase2-acdc.ipynb`
3. `Phase 3/phase3-daformer with results.ipynb` (first combined run)
4. `Phase 3/phase4-daformer research and performance proofs.ipynb` (ablations + final tuned run)
5. `Phase 3/phase3-extensions.ipynb` (weather-aware masking + threshold sweep)
6. `Phase 3/phase3-final-weather-aware.ipynb` (**not yet run** — see Current status)

A CUDA-capable GPU is recommended for training and adaptation. All Phase 3 notebooks after step 3
were run on Kaggle, not locally.

### Running on Kaggle

Every Phase 3 notebook auto-discovers its inputs by folder name, so attach these 4 Kaggle datasets
to any Phase 3 notebook you run (all already used by every successful run so far):

- `khushireddymacha/leftimg` — Cityscapes images (`leftImg8bit`)
- `soumikrakshit/cityscapes-coarse-fine` — Cityscapes labels (`gtFine`)
- `khushireddymacha/acdc-dataset` — ACDC images + ground truth (`rgb_anon` + `gt`)
- `khushireddymacha/phase1-model` — Phase 1 checkpoint (`best_model.pth`)

The Kaggle CLI (`pip install kaggle`, credentials in `~/.kaggle/kaggle.json` — use the **Legacy API
Key** option under kaggle.com/settings, not the newer `KGAT_`-prefixed token, which the current
`kaggle` PyPI package doesn't support yet) can push/monitor/pull runs:

```
kaggle kernels push -p "Phase 3/kaggle_push_final_weather_aware"        # upload + trigger a run
kaggle kernels status khushireddymacha/phase-3-final-weather-aware      # check progress
kaggle kernels output khushireddymacha/phase-3-final-weather-aware -p "Phase 3/kaggle_output"  # pull results
```

**Known gotcha:** the CLI can't select the GPU accelerator type (`enable_gpu` is just on/off) — a
CLI-triggered push once defaulted to an accelerator whose CUDA build was incompatible with the
notebook environment (`AcceleratorError: no kernel image is available for execution on the
device`), failing within ~3 minutes every time regardless of GPU type tried except one. **GPU T4
×2**, selected explicitly in the browser's notebook settings before "Save Version," is the
configuration every successful run has used — if a run fails immediately with that error, open it
in the browser and re-select T4 ×2 rather than debugging the notebook code.

## Outputs

The tracked result folders and reports include:

- training curves and validation-mIoU-over-epochs plots;
- per-class and per-condition IoU comparisons, plus pixel accuracy;
- source-to-target domain-gap visualization;
- qualitative semantic-segmentation predictions (source-only vs. adapted, per condition); and
- `Phase 3/phase3-report.pdf` (full narrative writeup) and `Phase 3/phase3-summary.pdf` (one-page
  attempt-by-attempt table) — read these first for a fast catch-up on everything done so far.

## Notes

- Cityscapes and ACDC remain subject to their respective licenses.
- Checkpoint files (`.pth`, `.pt`, and `.ckpt`) are ignored and should be stored separately or published through a model/artifact hosting service — every Kaggle run's checkpoints live in that run's own Kaggle kernel output, not in this repo.
- Notebook filenames without "with results" / "FINAL" are generator *templates* (no execution
  outputs); their `make_*_notebook.py` counterpart regenerates them if you need to tweak an
  experiment. Filenames with "with results" or "FINAL" are the executed, downloaded-from-Kaggle
  versions with real training curves and numbers in them.


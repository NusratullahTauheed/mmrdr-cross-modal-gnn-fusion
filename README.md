# Cross-Modal Attention and Modality-Graph Gated Fusion for Referable Diabetic Retinopathy Detection

Code, saved outputs and results for the article *Cross-modal attention and modality-graph gated fusion of
transfer-learned encoders for referable diabetic retinopathy detection* (Nusratullah Tauheed and Iftikhar Naseer,
Faculty of Computer Science and Information Technology, Superior University, Lahore, Pakistan).

## Description

The model detects referable diabetic retinopathy (DR grade ≥ 2) from up to four inputs per sample: colour fundus
photography (CFP), ultra-widefield imaging (UWF), optical coherence tomography (OCT) and a 30-dimensional simulated
descriptor vector. Frozen EfficientNet-B0 encoders turn each input into a 512-d node; a fusion head combines the
nodes with descriptor-conditioned cross-modal attention, a learnable modality graph with GRU message passing, and an
adaptive softmax gate whose per-sample weights show each modality's contribution. Missing modalities are handled with
zero placeholders and a validity mask; robustness is measured by withholding one modality at a time.

The notebooks are the ones that produced the published results. Their code and saved outputs are unchanged; only
explanatory markdown cells and an environment-recording cell were added.

## Dataset information

| Item | Value |
|---|---|
| Name | MMRDR — Multi-Modal Retinal images for in-depth DR analysis |
| Creators | Zhenyu Tang, Lilong Wang, Qianqian Liang, Shuyue Xue, Chengcheng Feng, Lili Ran, Lingzhi Chen, Zhen Guo, Xiaosong Wang, Jun Li |
| Source | figshare, version 2 (2025): https://doi.org/10.6084/m9.figshare.29423747.v2 |
| Data descriptor | Tang et al. (2026), *Scientific Data* 13:639, https://doi.org/10.1038/s41597-026-07005-9 |
| License | CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/ |
| Size | about 18.6 GB (multi-part zip) |
| Content used | MMRDR-CFP (11,118 images, DR grade 0–4), MMRDR-UWF (10,404, DR grade 0–4), MMRDR-OCT (2,938, DME grade 0–2); all `tr*` and `ts*` files |

**The dataset is not redistributed here.** Download it from the official figshare record and unzip it so that
`DatasetRT/` contains `MMRDR-CFP/`, `MMRDR-OCT/` and `MMRDR-UWF/`, each with `img/` and its annotation CSV.

**How samples are formed.** One sample = one MMRDR file identifier (`tr000001`, …) and one eye (`lr` 0 → OS,
1 → OD). Images of different modalities with the same identifier and eye share a sample. The MMRDR subsets come from
different sources, so this is pairing **by identifier, not by patient**: where a sample has both a CFP and a UWF grade,
the grades agree in 1,386 of 5,071 samples (27.3 %). Label = CFP grade if present, else UWF grade, else OCT grade, so
the 773 OCT-only samples are labelled with a DME grade. 17,224 samples in total.

**Simulated descriptor stream (`data/metadata.csv`).** MMRDR has no clinical records. Every identifier carries
simulated values (age, diabetes duration, HbA1c, blood pressure, BMI, visual acuity, intraocular pressure and eight
categorical variables; `metadata_is_synthetic = 1`). The simulated values depend on the label: over the 17,224
samples, single-variable AUCs for separating referable from non-referable samples range from 0.571 (intraocular
pressure) to 0.766 (visual acuity). The script that generated the file is not part of this repository.

## Code information

| Notebook | Role | Main outputs |
|---|---|---|
| `notebooks/00_build_manifest.ipynb` | Builds `DatasetRT/manifest.csv` from the MMRDR annotation files and copies `data/metadata.csv` next to it (added for reproducibility; checks equality with an existing manifest) | `manifest.csv` |
| `notebooks/01_stage1_encoder_training.ipynb` | **Stage 1** (original `P2_model_v5(Final).ipynb`): trains the CFP, UWF and descriptor encoders on its own split of the 16,451 samples with a CFP or UWF image; re-evaluates the checkpoint | `stage6_full_effnet_b0_best.pt`, `stage6_resumed_final_metrics.csv` |
| `notebooks/02_stage2_fusion_training_and_evaluation.ipynb` | **Stage 2** (original `P2_model_v6(Final).ipynb`): caches node embeddings, trains the fusion head, selects checkpoint and threshold on validation, tests once, runs the modality ablation, makes figures and tables | everything in `results/` |
| `notebooks/03_split_overlap_check.ipynb` | Re-creates both splits and counts Stage-2 validation/test samples that were Stage-1 training samples; verifies the test order against the saved gate weights (CPU only) | overlap table |
| `results/` | Saved outputs of notebooks 01 and 02: metrics, robustness, gate weights, threshold search, training history, tables workbook, figures, and the fusion-head checkpoint `approved_fast_attention_gnn_best.pt` | — |
| `data/metadata.csv` | Simulated descriptors (input) | — |
| `CODEBOOK.md` | Meaning of every code in the shared files | — |

## Requirements

Python 3.12 on Google Colab with: torch, torchvision, timm, numpy, pandas, scikit-learn, Pillow, matplotlib,
seaborn, tqdm, openpyxl (`requirements.txt`). Library versions were not recorded during the original runs (Colab
default runtime, July 2026); the environment cell at the top of notebooks 01 and 02 prints them on any rerun.

## Computing infrastructure

Google Colab (Linux) with one NVIDIA Tesla T4 GPU (16 GB) for training and feature extraction; the Stage-1
re-evaluation (notebook 01, cells 7–11) ran on the Colab CPU runtime.

## Usage instructions

1. Download MMRDR from https://doi.org/10.6084/m9.figshare.29423747.v2 and unzip it into
   `MyDrive/MMRDR_Project/DatasetRT/`.
2. Copy this repository's `data/` folder to `MyDrive/MMRDR_Project/data/` and `results/` to `MyDrive/MMRDR_Project/results/`.
3. Run `00_build_manifest.ipynb` (writes `DatasetRT/manifest.csv` and `DatasetRT/metadata.csv`).
4. Run `01_stage1_encoder_training.ipynb` on a T4 GPU (writes `stage1_outputs/`).
5. Run `02_stage2_fusion_training_and_evaluation.ipynb` on a T4 GPU (writes `approved_fast_model_outputs/`). If that
   folder already holds a feature cache or checkpoint, the notebook reuses them; delete them to retrain.
6. Optionally run `03_split_overlap_check.ipynb` (CPU) to reproduce the overlap numbers.

## Methodology

* **Preprocessing.** RGB conversion, resize to 224 × 224, ImageNet mean/std normalisation; a missing or unreadable
  image becomes a zero tensor with validity flag 0. Descriptors: numeric → median imputation + z-score; categorical →
  most-frequent imputation + one-hot; fitted on the training split of each stage (30 features).
* **Stage 1 (notebook 01).** Split: samples with CFP or UWF (16,451), 70/15/15 stratified, seed 42 → 11,515 / 2,468 / 2,468.
  Two ImageNet-initialised `tf_efficientnet_b0` encoders (CFP, UWF) + Linear(1280→512), LayerNorm, GELU, Dropout 0.30;
  descriptor encoder 30→256→512; softmax gate over the three embeddings and classifier 512→256→2. AdamW (lr 1e-4,
  weight decay 1e-4), cosine schedule, 5 epochs, batch 8, mixed precision, clipping 1.0, weighted sampler and
  class-weighted cross-entropy (label smoothing 0.03), flip and ±10 % brightness/contrast augmentation. Best validation
  F1 (argmax) at epoch 4 (0.8718).
* **OCT branch.** EfficientNet-B0 with ImageNet weights; its projection layer is not in the Stage-1 checkpoint and
  keeps its random initialisation.
* **Stage 2 (notebook 02).** Split: all 17,224 samples, 70/15/15 stratified, seed 42 → 12,056 / 2,584 / 2,584, drawn
  independently of Stage 1. Frozen encoders; four 512-d nodes + mask cached. Fusion head: descriptor gate
  `m = σ(W e + b)`, nodes × (1 + m), 4-head self-attention with key-padding mask, residual + LayerNorm, feed-forward
  (dropout 0.20); 4 × 4 learnable adjacency with masked softmax, linear messages, GRU-cell update, residual + LayerNorm;
  masked softmax gate over nodes and gate-weighted sum; classifier 512→256→2. AdamW (lr 1e-3, weight decay 1e-4), cosine
  over 30 epochs, batch 256, weighted sampler, cross-entropy with label smoothing 0.03, clipping 1.0, modality dropout
  (images 0.10, descriptor 0.03). Best validation F1 at 0.50 → epoch 3 (0.9211). Threshold: 181 values in
  [0.05, 0.95], best validation F1 → 0.46.

## Evaluation method and assessment metrics

Hold-out test split scored once at the validation-selected threshold: accuracy, precision, recall, F1-score,
per-class report and confusion matrix; ROC-AUC and PR-AUC; Brier score and reliability curve. Missing-modality
robustness: each modality withheld from every test sample in turn. Modality contribution: distribution of the gate
weights over the test split. Internal reference: the Stage-1 model on its own test split (notebook 01).

## Main results

| Model / condition (test split) | n | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| Stage-2 fusion, all inputs (τ = 0.46) | 2,584 | 0.9443 | 0.9270 | 0.9191 | 0.9230 | 0.9784 | 0.9695 | 0.0468 |
| CFP withheld | 2,584 | 0.7601 | 0.6337 | 0.8051 | 0.7092 | 0.8366 | 0.7739 | 0.1745 |
| UWF withheld | 2,584 | 0.8781 | 0.8451 | 0.8136 | 0.8291 | 0.9410 | 0.9077 | 0.0916 |
| Descriptors withheld | 2,584 | 0.9156 | 0.8775 | 0.8924 | 0.8849 | 0.9689 | 0.9523 | 0.0648 |
| OCT withheld | 2,584 | 0.9373 | 0.9265 | 0.8988 | 0.9124 | 0.9766 | 0.9665 | 0.0511 |
| Stage-1 model, own test split (τ = 0.41) | 2,468 | 0.9246 | 0.8834 | 0.9057 | 0.8944 | 0.9725 | 0.9541 | 0.0630 |

Mean gate weights on the test split: CFP 0.6129, UWF 0.2244, descriptors 0.1368, OCT 0.0259.

**Partition overlap.** 1,740 of the 2,584 Stage-2 test samples (67.3 %) and 1,697 of the 2,584 validation samples
(65.7 %) belong to the Stage-1 training partition, so the CFP, UWF and descriptor encoders had been trained on them.
The Stage-2 results are therefore likely optimistic. The Stage-1 result is free of this overlap.

## Conclusions

The fusion head reaches high discrimination and good calibration on MMRDR, and the gate weights rank the modalities in
the same order as the ablation (CFP > UWF > descriptors > OCT). Because of the partition overlap, the identifier-based
pairing and the simulated, label-dependent descriptors, these figures describe the fusion mechanism on this benchmark
and are not an estimate of clinical performance.

## Limitations

* Stage-2 validation and test samples overlap with the Stage-1 training partition (see above).
* Cross-modality pairing is by identifier only; labels come from one modality per sample; OCT-only samples carry a DME grade.
* The descriptor stream is simulated and label-dependent; its generation script is not available.
* The OCT encoder is not DR-supervised and its projection is untrained.
* One run, one seed, no external validation; library versions were not recorded.

## Citations

* Code: Tauheed N, Naseer I. (2026). *mmrdr-cross-modal-gnn-fusion: v1.0 – Code for the PeerJ Computer Science article* (Version v1.0). Zenodo. https://doi.org/[Zenodo DOI, added after the v1.0 release]
  — repository: https://github.com/NusratullahTauheed/mmrdr-cross-modal-gnn-fusion
* Dataset: Tang Z, Wang L, Liang Q, Xue S, Feng C, Ran L, Chen L, Guo Z, Wang X, Li J. (2025). MMRDR (Multi-Modal
  Retinal images for in-depth DR analysis), version 2. figshare. https://doi.org/10.6084/m9.figshare.29423747.v2
* Data descriptor: Tang Z, et al. (2026). A multimodal retinal image dataset for diabetic retinopathy detection using
  foundation models. *Scientific Data* 13:639. https://doi.org/10.1038/s41597-026-07005-9

## License

Code: MIT (`LICENSE`). MMRDR images and annotations remain under CC BY 4.0 by their creators; files derived from the
MMRDR annotations are shared under CC BY 4.0 with attribution to MMRDR.

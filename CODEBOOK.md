# Codebook

Meaning of every coded value in the files shared in `data/` and `results/` and in the manifest written by notebook 00.

## `DatasetRT/manifest.csv` (written by notebook 00)

| Column | Values |
|---|---|
| `sample_id`, `patient_id` | MMRDR file identifier (`tr######` = MMRDR training file, `ts######` = MMRDR test file). `patient_id` equals `sample_id`; it is **not** a real patient identifier. |
| `eye` | `OD` = right eye (MMRDR `lr` = 1); `OS` = left eye (MMRDR `lr` = 0) |
| `fundus_path`, `oct_path`, `uwf_path` | absolute image path; empty = modality not available for this sample |
| `fundus_grade`, `uwf_grade` | DR severity: 0 = no DR, 1 = mild NPDR, 2 = moderate NPDR, 3 = severe NPDR, 4 = PDR |
| `oct_grade` | diabetic macular edema: 0 = no DME, 1 = non-centre-involved DME, 2 = centre-involved DME |
| `dr_grade` | sample label: `fundus_grade` if present, else `uwf_grade`, else `oct_grade` |
| `fundus_lesion`, `uwf_lesion` | list of 7 binary flags (0 = absent, 1 = present) in the order: microaneurysm, hard exudate, intraretinal hemorrhage, venous beading / IRMA, neovascularization, vitreous hemorrhage, retinal detachment |
| `oct_lesion`, `dme_label` | always empty (kept for compatibility) |

Derived in the notebooks: `referable_dr` = 1 if `dr_grade` ≥ 2, else 0 (0 = non-referable, 1 = referable).

## `data/metadata.csv` (simulated descriptor stream)

| Column | Type / values |
|---|---|
| `sample_id`, `patient_id` | MMRDR file identifier (one row per identifier, shared by both eyes) |
| `age` | years |
| `diabetes_duration` | years |
| `hba1c` | % |
| `systolic_bp`, `diastolic_bp` | mmHg |
| `bmi` | kg/m² |
| `visual_acuity` | simulated value in the range 0–0.942 (unit not defined by the simulation) |
| `intraocular_pressure` | mmHg |
| `metadata_is_synthetic` | 1 = simulated (all rows) |
| `gender` | `Male`, `Female` |
| `ethnicity` | `Asian`, `White`, `Hispanic`, `Black`, `Other` |
| `diabetes_type` | `Type 1`, `Type 2` |
| `hypertension`, `hyperlipidemia`, `kidney_disease` | `Yes`, `No` |
| `smoking_status` | `Never`, `Former`, `Current` |
| `alcohol_status` | `Never`, `Occasional`, `Regular` |

No value is missing in this file; the imputation steps in the notebooks therefore have no effect on it.

## `results/` (saved outputs of notebooks 01 and 02)

| File | Codes |
|---|---|
| `approved_fast_attention_gnn_test_metrics.csv` | one row: threshold, `test_n`, metrics for the referable class (proportions); `brier_score` = mean squared error of the predicted probability |
| `approved_fast_attention_gnn_modality_robustness.csv` | `condition`: `full_modalities` = all inputs; `fundus_missing`, `oct_missing`, `uwf_missing`, `metadata_missing` = that modality withheld from every test sample |
| `approved_fast_attention_gnn_gate_weights.csv` | one row per test sample (in test-split order); `fundus_gate`, `oct_gate`, `uwf_gate`, `metadata_gate` = gate weights, summing to 1 over the modalities present; exactly 0 = modality absent |
| `approved_fast_attention_gnn_gate_summary.csv` | `describe()` statistics of the gate weights (count, mean, std, min, quartiles, max) |
| `approved_fast_attention_gnn_threshold_search.csv` | validation metrics for each of the 181 thresholds |
| `approved_fast_attention_gnn_history.csv` | per-epoch training loss and validation metrics at threshold 0.50 |
| `stage6_resumed_final_metrics.csv` | Stage-1 model on its own test split (from notebook 01, cell 11) |
| `paper_results_tables.xlsx` | sheets `final_metrics`, `classification_report`, `modality_robustness`, `gate_weights_summary` |
| `approved_fast_attention_gnn_best.pt` | fusion-head weights (PyTorch `state_dict` under `model_state`) |
| feature-cache node order | 0 = CFP, 1 = OCT, 2 = UWF, 3 = descriptor; mask 1 = present, 0 = absent |
| class indices | 0 = non-referable DR, 1 = referable DR |

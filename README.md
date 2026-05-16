# Stabilizing Dynamic Early Warning Systems via Latent Feature Representation (OULAD Pipeline)

Official repository for the research paper: **"Stabilizing Dynamic Early Warning Systems via Latent Feature Representation in AE-LSTM and AE-Transformer"**. 

This folder contains the complete, standalone implementation pipeline optimized for the **OULAD (Open University Learning Analytics Dataset)**. The core framework introduces deep autoencoders as universal convergence stabilizers to mitigate training volatility and eliminate hazardous gradient/loss spikes induced by highly sparse student clickstream sequences.

---

## 🛠️ Pipeline Architecture & Cell Breakdown

The notebook is structurally modularized into 11 logical processing zones, ensuring strict segregation of concerns and reproducibility:

### 1. Environment & Determinism (`CELL 1`)
Enforces global deterministic seed parameters (`seed=42`) across PyTorch, NumPy, and random generation backends. This guarantees 100% exact replication of convergence paths, training behaviors, and metric scores on any physical device.

### 2. Ingestion & Storage Integration (`CELL 2`)
Establishes data streams between Google Colab and the cloud file storage layer. Ingests student demographic profiles and automatically merges fragmented historical engagement logs into a unified structure.

### 3. Exploratory Data Analysis (`CELL 3`)
Profiles missing value densities and data types. Binarizes target categories by converting raw administrative outcomes into a standardized academic risk status matrix (`0` for Safe/Pass, `1` for At-Risk/Fail) and graphs class balance thresholds.

### 4. Leakage-Free Feature Engineering (`CELL 4`)
Partitions student indices into Train, Validation, and Test sets (60:20:20 ratio) based strictly on unique IDs. This mathematically isolates the pure testing environment from historical leakage. Long log lists are vectorized into structured 3D sequence tensors: 
$$\mathbf{X} \in \mathbb{R}^{\text{Students} \times \text{Weeks} \times \text{VLE Attributes}}$$

### 5. Neural Network Architecture Specifications (`CELL 5`)
Declares structural object blueprints for baseline models (`Vanilla-LSTM`, `Vanilla-Transformer`) and the proposed hybrid frameworks (`AE-LSTM`, `AE-Transformer`). Implements a constrained 32-unit information bottleneck space utilizing dense sigmoidal projection bounds.

### 6. Multi-Horizon Joint Optimization Pipeline (`CELL 6`)
Simulates dynamic real-world campus deployment by tracking performance iteratively at target milestones (Weeks 5, 10, 20, 30, and 38). Normalization boundaries are derived exclusively per week from training matrices. Models are optimized utilizing a strict Multi-Objective Loss Function:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{classification}} + 0.05 \cdot \mathcal{L}_{\text{reconstruction}}$$

### 7. Quantitative Recapitulation & Visual Diagnostic Logs (`CELL 7 & 8`)
Processes multidimensional evaluation logs into structured tables. Renders longitudinal performance trajectory charts, group validation charts, and cumulative ROC curves. Generates dual-axis convergence curves to diagnose training optimization paths and prints empirical 2x2 confusion matrices.

### 8. Post-Hoc Model Explainability via SHAP (`CELL 9`)
Deploys explainable AI (XAI) using `GradientExplainer` algorithms to dismantle neural network "black-box" constraints. Quantifies specific behavioral feature attrition weights, providing actionable insights into which VLE interaction types actively mitigate or trigger student academic failure risk.

### 9. Operational Hardware & Hardware Profiling (`CELL 10`)
Tracks physical device footprints within native CUDA environments. Measures isolated processing time per execution loop (`Step Time`), peak memory reservation footprints (`Peak VRAM`), and systemic load thresholds (`System RAM`).

### 10. Inferential Stability Verification (`CELL 11`)
Executes a validation suite evaluating structural capacity variants ($z \in \{16, 32, 64\}$) to empirically confirm the information bottleneck equilibrium. Conducts independent two-sample Student's t-tests over validation histories to mathematically validate that the stabilization advantage is statistically significant ($p < 0.05$).

---

## 📊 Dataset Availability & Resources

The raw OULAD benchmark files required to execute this experimental pipeline can be acquired from either of the following open-access public data repositories:
- **UCI Machine Learning Repository**: [Open University Learning Analytics Dataset (OULAD Official Release)](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset)
- **Kaggle Datasets**: [Student Demographics & Online Education Data (OULAD Alternative)](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)

---

## 🚀 Execution & Requirements

### Technical Dependencies
- Python >= 3.8
- PyTorch >= 2.0
- Scikit-Learn >= 1.0
- SHAP >= 0.40
- SciPy, Psutil, Pandas, NumPy, Matplotlib, Seaborn

### Running the Notebook
1. Download the raw dataset files (`studentInfo.csv`, `vle.csv`, and `studentVle.csv`) from the resource links provided above.
2. Place your raw OULAD files inside your designated Google Drive folder directory path as configured in `CELL 2`.
3. Open the file `stabilizing_ews_latent_oulad.ipynb` inside Google Colab.
4. Select a GPU Runtime Environment (`Runtime` -> `Change runtime type` -> `T4 GPU`).
5. Execute the cells sequentially from `CELL 1` to `CELL 11`. All quantitative results and model checkpoints (`.pth`) will save automatically to your Drive results folder.

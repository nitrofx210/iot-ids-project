# ML-Based Intrusion Detection System for IoT Networks

**Student:** Aryan Thapa | AM.SC.P2CSN25003  
**Program:** M.Tech Cybersecurity and Systems of Networks  
**Institution:** Amrita School of Computing, Amritapuri  
**Duration:** 12 Weeks (July – September 2026)

---

## Project Overview

A hybrid Machine Learning-based Intrusion Detection System (IDS) trained on the **Bot-IoT dataset** that:

- Classifies known IoT attack categories (DDoS, DoS, Reconnaissance, Theft) using **Random Forest**
- Detects novel/zero-day anomalies using **Isolation Forest**
- Combines both in a confidence-based hybrid decision pipeline
- Will expose detections via a **Flask REST API** (Week 7)
- Will log all events to **SQLite** and visualise via **Streamlit** (Week 8–9)
- Will send real-time **Telegram alerts** (Week 10)

---

## Week 2 Results (Baseline Models)

| Model | Type | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|------|----------|-----------|--------|----------|---------|
| **Random Forest (Binary)** | Supervised | 99.86% | 0.9999 | 0.9987 | 0.9993 | 1.0000 |
| **Isolation Forest** | Unsupervised | 75.68% | 1.0000 | 0.7544 | 0.8600 | 0.9867 |
| **Hybrid RF + IF** | Combined | 99.87% | — | — | 0.9993 | — |

**Random Forest Multi-Class (attack category):** 98.70% accuracy across DDoS, DoS, Reconnaissance, Normal, Theft.

**5-Fold Cross-Validation:** Mean F1 = 0.9994 ± 0.0002 (low variance confirms no overfitting)

**Top predictive feature:** `N_IN_Conn_P_DstIP` (inbound connections per destination IP) — 43.6% importance, consistent with DDoS/DoS flood mechanics.

> Full methodology, class-imbalance handling, and confusion matrices are documented in `reports/Week2_Report.docx`.

---

## Architecture

```
IoT Network Traffic
        │
        ▼
[ Feature Extraction — 10 Best Features ]
        │
        ▼
┌───────────────────────┐
│   Stage 1: Random     │  ──► Known Attack Label + Confidence
│   Forest Classifier   │       (F1 = 0.9993)
└───────────────────────┘
        │ (Low confidence / Normal)
        ▼
┌───────────────────────┐
│  Stage 2: Isolation   │  ──► Anomaly Score
│  Forest Detector      │       (F1 = 0.8600 standalone)
└───────────────────────┘
        │
        ▼
[ Decision Logic Layer ]  ──► Hybrid F1 = 0.9993
        │
   ┌────┴────┐
   ▼         ▼
Flask API   SQLite Log     (Week 7-8)
   │             │
   ▼             ▼
Streamlit    Telegram      (Week 9-10)
Dashboard     Alerts
```

---

## Dataset

**Bot-IoT Dataset — Best 10 Features** (Koroniotis et al., 2019) — UNSW Canberra  
Download: https://research.unsw.edu.au/projects/bot-iot-dataset

Used files (pre-split — note the Training/Testing files use comma separators while the combined reference file uses semicolons):
- `UNSW_2018_IoT_Botnet_Final_10_best_Training.csv` (2,934,817 records)
- `UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv` (733,705 records)

Place downloaded CSVs in `data/raw/` (gitignored — not committed to this repo)

**Class imbalance:** ~7,930:1 (attack:normal) in raw data. Resolved via 100:1 controlled undersampling of the attack class (NOT SMOTE — see Week 2 report Section 4.3 for rationale).

---

## Project Structure

```
iot-ids-project/
├── data/
│   ├── raw/                    # Raw Bot-IoT CSV files (not committed)
│   └── processed/              # Cleaned, balanced, scaled .npy arrays (not committed)
├── notebooks/
│   ├── 01_EDA.ipynb            # Week 2: Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb  # Week 2: Cleaning, balancing, scaling
│   ├── 03_RandomForest.ipynb   # Week 2: Binary + multi-class RF
│   └── 04_IsolationForest.ipynb# Week 2: Anomaly detection + hybrid pipeline
├── models/
│   ├── random_forest_binary.pkl
│   ├── random_forest_multiclass.pkl
│   ├── isolation_forest.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── hybrid_thresholds.json
├── api/
│   └── app.py                  # Flask REST API (skeleton ready, Week 7)
├── dashboard/                   # Streamlit dashboard (Week 9)
├── alerts/                      # Telegram bot (Week 10)
├── utils/
│   ├── preprocessing.py
│   └── predict.py               # HybridIDS inference class
├── reports/
│   ├── Week1_Report.docx
│   ├── Week2_Report.docx
│   └── *.png                    # All generated charts/confusion matrices
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Weekly Progress

| Week | Status | Deliverable |
|------|--------|-------------|
| Week 1 | ✅ Complete | Problem statement, literature review, architecture, report |
| Week 2 | ✅ Complete | EDA, preprocessing, RF (F1=0.999) + IF (F1=0.86) + Hybrid (F1=0.999) |
| Week 3 | 🔄 Next | Hyperparameter tuning, stratified Theft handling, XGBoost benchmark |
| Week 4–6 | ⏳ Planned | Model comparison, SHAP explainability |
| Week 7–8 | ⏳ Planned | Flask API + SQLite logging |
| Week 9–10 | ⏳ Planned | Streamlit dashboard + Telegram alerts |
| Week 11–12 | ⏳ Planned | Integration, final report, defence |

---

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/yourusername/iot-ids-project.git
cd iot-ids-project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly flask flask-cors streamlit sqlalchemy requests joblib imbalanced-learn jupyter ipykernel python-dotenv lime

# 4. Download Bot-IoT Training + Testing CSVs → place in data/raw/

# 5. Run notebooks in order
jupyter notebook notebooks/01_EDA.ipynb
# then 02_Preprocessing.ipynb → 03_RandomForest.ipynb → 04_IsolationForest.ipynb
```

> **Note:** Use unpinned package versions on Windows to avoid compiler build errors with pre-built wheels.

---

## Known Issues & Lessons Learned (Week 2)

- **Always load train/test splits separately** — loading the combined reference file alongside the pre-split files causes data leakage and inflated (100%) accuracy.
- **Check CSV separators per file** — the combined Bot-IoT file uses `;`, but the dedicated Training/Testing files use `,`.
- **Avoid SMOTE on extremely small minority classes** (<500 samples) — it generates unrealistic synthetic duplicates. Controlled undersampling of the majority class is more honest for highly imbalanced security datasets.

Full root-cause analysis in `reports/Week2_Report.docx`, Section 10.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Models | scikit-learn (RandomForestClassifier, IsolationForest) |
| Explainability | SHAP, LIME (Week 6) |
| API | Flask (Week 7) |
| Dashboard | Streamlit (Week 9) |
| Database | SQLite + SQLAlchemy (Week 8) |
| Alerts | Telegram Bot API (Week 10) |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |

---

## References

- Koroniotis et al. (2019). Bot-IoT Dataset. *Future Generation Computer Systems*, 100, 779–796.
- Mirsky et al. (2018). Kitsune. *NDSS 2018*.
- Meidan et al. (2018). N-BaIoT. *IEEE Pervasive Computing*, 17(3), 12–22.
- Breiman (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Liu et al. (2008). Isolation Forest. *IEEE ICDM 2008*.

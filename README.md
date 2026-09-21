# 🧠 Parkinson's Disease Detection Using Explainable XGBoost

### An Explainable, Threshold-Optimized Machine Learning Framework for Parkinson's Disease Detection from Sustained-Vowel Vocal Biomarkers

**Team:** Guardians of Tomorrow
**Institution:** K.L.N. College of Engineering
**Program:** B.E. Computer Science and Engineering (IoT)
**Theme:** Health Care
**Problem Statement ID:** TRN-05
**Team ID:** Tn001

### 👥 Team Members

* **Jai Krishna K** — Team Leader
* **Hazi Khan U**

### 👩‍🏫 Mentor

**Mrs. Binu Shoba D S**
Assistant Professor, Department of IoT
K.L.N. College of Engineering

---

## 📌 Overview

Parkinson's Disease (PD) is a progressive neurological disorder in which changes in speech and voice can occur alongside other symptoms. This project investigates whether **sustained-vowel vocal biomarkers** can be used as a low-cost, non-invasive screening aid for Parkinson's Disease.

We developed an **end-to-end explainable machine learning pipeline** that performs preprocessing, feature selection, class balancing, model optimization, explainability analysis, and threshold calibration before making a prediction.

The final system uses **XGBoost** with **eight SHAP-selected vocal biomarkers** and a threshold optimized for screening-oriented sensitivity.

> **Important:** This project is intended as a machine-learning research and screening-support system, not as a standalone medical diagnostic tool.

---

## 🎯 Problem Statement

Conventional Parkinson's Disease diagnosis generally relies on clinical evaluation and specialist assessment. Access to specialists can be limited, particularly in resource-constrained regions.

Voice analysis provides a potential low-cost and non-invasive source of biomarkers. However, developing a reliable ML-based screening system presents several challenges:

* High-dimensional and redundant acoustic features can cause overfitting.
* Small datasets can make model generalization difficult.
* Class imbalance can bias predictions toward the majority class.
* Black-box predictions can be difficult to interpret.
* The default **0.50 classification threshold** may not be appropriate for screening-oriented applications where sensitivity is important.

### Objective

To develop an **explainable and threshold-optimized ML pipeline** that can screen for Parkinson's Disease using vocal biomarkers extracted from a sustained-vowel recording.

---

## 🔬 Methodology

The proposed pipeline consists of the following stages:

```text
Sustained-Vowel Voice Recording
            ↓
Acoustic Feature Extraction
            ↓
Data Preprocessing
            ↓
Correlation Filtering
            ↓
L1 Regularization (LASSO)
            ↓
Recursive Feature Elimination (RFE)
            ↓
SMOTE Class Balancing
            ↓
Standard Scaling
            ↓
XGBoost Classification
            ↓
SHAP Explainability
            ↓
Threshold Optimization
            ↓
Parkinson's / Non-Parkinson's Prediction
```

---

## 🧪 Vocal Biomarkers

The dataset contains acoustic measurements extracted from sustained-vowel recordings.

Examples of the vocal features include:

* Fundamental frequency — `MDVP:Fo(Hz)`
* Maximum fundamental frequency — `MDVP:Fhi(Hz)`
* Minimum fundamental frequency — `MDVP:Flo(Hz)`
* Jitter measurements
* Shimmer measurements
* Harmonicity measures
* Noise-related measures
* Additional frequency and amplitude-based biomarkers

The feature-selection pipeline progressively reduces redundant features and identifies the most informative biomarkers for classification.

---

## ⚙️ Machine Learning Pipeline

### 1. Correlation Filtering

Highly correlated features are identified and redundant variables are removed.

**Purpose:**

* Reduce multicollinearity
* Reduce feature redundancy
* Improve model efficiency

### 2. L1 Regularization / LASSO

LASSO applies an L1 penalty that can drive the coefficients of less useful features toward zero.

**Purpose:**

* Feature selection
* Reduce model complexity
* Remove less informative variables

### 3. Recursive Feature Elimination (RFE)

RFE recursively removes less important features until a compact feature subset is obtained.

### 4. SMOTE

**Synthetic Minority Over-sampling Technique (SMOTE)** is used to address class imbalance by generating synthetic samples for the minority class.

### 5. Standard Scaling

Features are normalized using `StandardScaler` to place numerical variables on a comparable scale.

### 6. XGBoost

The final classifier uses **Extreme Gradient Boosting (XGBoost)** because of its ability to model nonlinear relationships and interactions between acoustic biomarkers.

### 7. SHAP Explainability

**SHAP (SHapley Additive exPlanations)** is used to understand the contribution of individual features to model predictions.

This provides an interpretable layer that can help explain **why a particular prediction was produced**.

### 8. Threshold Optimization

Instead of relying only on the default probability threshold of `0.50`, different thresholds are evaluated to identify a screening-oriented operating point.

A lower threshold can increase sensitivity, although this may also increase false-positive predictions.

---

## 📊 Model Performance

On the held-out evaluation data, the final model achieved:

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **92.31%** |
| Precision | **93.33%** |
| Recall    | **96.55%** |
| F1-Score  | **94.92%** |
| ROC-AUC   | **97.24%** |

The model correctly identified **28 of 29 Parkinson's Disease cases** in the held-out evaluation set.

For comparison, the untuned baseline achieved:

| Metric   |   Baseline |
| -------- | ---------: |
| Accuracy | **82.05%** |
| ROC-AUC  | **88.62%** |

> These results are based on the project's evaluation setup and dataset. They should not be interpreted as clinical validation or evidence of diagnostic performance in a general patient population.

---

## 🎚️ Threshold Optimization

The standard binary classification threshold is:

```text
Threshold = 0.50
```

For screening applications, missing a positive case can be important. Therefore, the project investigates a lower decision threshold to increase sensitivity.

The optimized screening threshold used in the project is:

```text
Threshold = 0.10
```

This threshold is intended for **screening-oriented sensitivity**, not for establishing a medical diagnosis.

---

## 🧠 Explainable AI

A major component of this project is interpretability.

SHAP is used to determine:

* Which vocal biomarkers influence predictions
* Whether a feature pushes a prediction toward or away from the PD class
* The relative importance of selected biomarkers
* How individual predictions can be explained

This helps move the system from a simple **"prediction-only" model** toward an **interpretable ML decision-support framework**.

---

## 💻 Technology Stack

### Programming

* Python

### Machine Learning

* XGBoost
* Scikit-learn
* SHAP
* Imbalanced-learn
* LASSO
* RFE

### Data Processing

* Pandas
* NumPy
* StandardScaler
* SMOTE

### Visualization

* Matplotlib
* Seaborn

### Application / Deployment

* Streamlit
* Joblib
* Git & GitHub

---

## 📁 Project Structure

```text
Parkinsons-Disease-Detection/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── best_xgb_model_shap_tuned.joblib
│   └── scaler.joblib
│
├── data/
│   └── parkinsons.csv
│
├── notebooks/
│   └── Parkinsons_ML_Pipeline.ipynb
│
├── assets/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── shap_summary.png
│   └── feature_importance.png
│
└── README.md
```

*Update the folder names above to match the actual files in your repository.*

---

## 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 🖥️ Application

The Streamlit interface provides a user-friendly prediction workflow where vocal biomarker values can be entered and processed by the trained ML model.

The system can provide:

* Prediction result
* Prediction probability
* Screening threshold
* Selected biomarker information
* Explainability information
* Model-based interpretation

---

## 📱 Future Scope

Future development may include:

* Direct voice recording through a web/mobile interface
* Automated acoustic feature extraction from uploaded audio
* Mobile application integration
* Longitudinal voice monitoring
* Digital-twin-inspired patient monitoring
* IoT-enabled audio sensing
* Larger and more diverse clinical datasets
* External validation across different populations
* Integration with clinical decision-support workflows

The **digital twin concept** is considered a future extension for longitudinal monitoring; it has not been established as clinically validated for Parkinson's Disease within this project.

---

## 🌍 SDG Mapping

### SDG 3 — Good Health and Well-being

**Targets:**

* **3.4** — Reduce premature mortality from non-communicable diseases and promote well-being.
* **3.8** — Access to quality essential health-care services.

**Contribution:**
The project explores early, non-invasive and low-cost voice-based screening support while providing explainable model outputs.

### SDG 9 — Industry, Innovation and Infrastructure

**Target:**

* **9.5** — Enhance scientific research and technological capabilities.

**Contribution:**
The project applies explainable AI, machine learning and IoT-ready sensing concepts to digital healthcare.

---

## ⚠️ Disclaimer

This project is an **academic research prototype**. It is designed for machine-learning research and screening-support experimentation.

It is **not a medical diagnostic device** and should not be used to diagnose, rule out, or make treatment decisions for Parkinson's Disease. Clinical assessment by qualified healthcare professionals remains necessary.

---

## 👨‍💻 Team

### Guardians of Tomorrow

**K.L.N. College of Engineering**
**B.E. Computer Science and Engineering (IoT)**

**Team Members**

* Jai Krishna K — Team Leader
* Hazi Khan U

**Mentor**

* Mrs. Binu Shoba D S — Assistant Professor, Department of IoT

---

## ⭐ Project Highlights

```text
✔ Explainable Machine Learning
✔ XGBoost Classification
✔ SHAP-Based Feature Selection
✔ LASSO + RFE Feature Reduction
✔ SMOTE Class Balancing
✔ Standard Scaling
✔ Threshold Optimization
✔ Sustained-Vowel Vocal Biomarkers
✔ Streamlit Prediction Dashboard
✔ Low-Cost and Non-Invasive Screening Concept
```

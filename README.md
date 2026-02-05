# 🧬 GA-LSTM for SMILES-Based Chemical Classification

This repository contains an end-to-end **machine learning pipeline** for **multi-class chemical compound classification** based on **SMILES representations**, combining:

* Feature extraction from SMILES strings
* Data preprocessing & scaling
* Dimensionality reduction using **PCA**
* **Genetic Algorithm (GA)** for feature selection
* **LSTM neural network** for final classification

The project is adapted from an academic thesis and restructured into a **clean, modular, and industry-ready pipeline** suitable for data science portfolios.

---

## 🚀 Project Overview

Chemical compounds represented as SMILES strings contain rich structural information. However, these representations require careful feature engineering and optimization to achieve strong predictive performance.

This project addresses that challenge by:

1. Extracting handcrafted molecular features from SMILES
2. Reducing redundancy via PCA
3. Selecting optimal feature subsets using a Genetic Algorithm
4. Modeling sequential patterns using an LSTM classifier

The final model supports **multi-class classification** (3 classes) and demonstrates stable generalization performance.

---

## 🧠 Methodology

### 1️⃣ Feature Extraction

* Input: SMILES strings
* Output: Numerical molecular descriptors
* Implemented in: `src/feature_extraction.py`

### 2️⃣ Preprocessing

* Handling missing values
* Feature scaling using `StandardScaler`
* Implemented in: `src/preprocessing.py`

### 3️⃣ PCA Dimensionality Reduction

* Reduces feature dimensionality
* Removes multicollinearity
* Improves training stability
* Implemented in: `src/pca_reduction.py`

### 4️⃣ Genetic Algorithm (GA)

* Binary chromosome representation
* Fitness evaluation using cross-validation
* Selects optimal subset of PCA features
* Implemented in: `src/ga_optimizer.py`

### 5️⃣ LSTM Classification

* Multi-class LSTM model
* Early stopping for regularization
* Implemented in: `src/lstm_model.py`

### 6️⃣ Inference

* Converts encoded predictions back to original class labels
* Implemented in: `src/inference.py`

---

## 📁 Project Structure

```
.
├── data/
│   └── raw/
│       └── smiles_dataset.csv
│
├── src/
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   ├── pca_reduction.py
│   ├── ga_optimizer.py
│   ├── lstm_model.py
│   └── inference.py
│
├── run_pipeline.py
├── test.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1. Create & Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Full Pipeline

```bash
python run_pipeline.py
```

---

## 📊 Results

* **Task**: Multi-class classification (3 classes)
* **Test Accuracy**: ~72–73%
* **Training Strategy**: Early stopping
* **Generalization**: Train and validation accuracy closely aligned (no overfitting)

Example inference output:

```
Predicted classes (encoded): [2 2 2 1 0]
Predicted classes (original): [3 3 3 2 1]
```

---

## 🧪 Why GA + LSTM?

* **GA** efficiently searches large feature spaces and avoids greedy selection
* **PCA** stabilizes GA search by reducing noise
* **LSTM** captures sequential dependencies in transformed feature space

This combination showed improved stability compared to baseline models during experimentation.

---

## 📌 Notes

* This repository focuses on **pipeline design and modeling strategy**, not domain-specific chemistry interpretation
* The implementation emphasizes **clean structure, reproducibility, and best practices**

---

## 👤 Author

**Bayu Chandra Putra**

Data Analyst | [LinkedIn](linkedin.com/bayuchandraputra)

---

## 📜 License

This project is released for educational and portfolio purposes.


← Back to Portfolio Index:  
https://github.com/bayucp30/portfolio-data-analyst

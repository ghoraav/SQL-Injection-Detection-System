# 🔐 SQL Injection Detection System 

---

## 📌 Project Overview

This project presents a **deep learning–based system for detecting and classifying SQL Injection (SQLi) attacks** using **Long Short-Term Memory (LSTM)** networks.

The system follows a **two-stage pipeline**:

1. **Binary classification** – determines whether a given SQL query is malicious or benign.  
2. **Multi-class classification** – identifies the specific SQL injection attack type.

The primary goal is to improve detection accuracy while reducing overfitting and improving generalization on unseen queries.  
The complete project specification, methodology, experiments, and results are taken from the submitted project report. :contentReference[oaicite:0]{index=0}

---

## 🎯 Objectives

- Build an LSTM-based model to detect SQL Injection attacks.
- Classify detected SQL injections into multiple attack categories.
- Reduce overfitting using regularization techniques.
- Improve generalization to unseen SQL queries.
- Evaluate the model using standard performance metrics.

---

## 🧠 Key Contributions

- A **pure LSTM-based architecture** for sequential SQL query analysis.
- A **two-stage classification framework** (binary + multi-class).
- **Character-level modeling** to capture fine-grained SQL syntax patterns.
- **Synthetic data generation** for balancing rare SQLi attack classes.
- Integration of **dropout and early stopping** to mitigate overfitting.

---

## 🛠️ Methodology

### 1. Data Preprocessing

- A verified SQL query dataset is loaded.
- Each query is labeled as:
  - `0` – benign
  - `1` – SQL injection
- SQL injection queries are also labeled with an **attack type**.
- Class imbalance is handled by:
  - Downsampling overrepresented classes.
  - Generating synthetic SQL queries for underrepresented attack types.
- Legitimate queries are also balanced using the same synthesis approach.
- **Character-level tokenization** is applied.
- All sequences are padded to a fixed length of **100 characters**.

---

### 2. Model Architecture

Two models share a similar architecture.

#### Embedding Layer
Converts character indices into dense vector representations.

#### LSTM Layers
- First LSTM layer: 64 units  
- Second LSTM layer: 32 units  
These layers learn sequential and contextual patterns in SQL queries.

#### Dense + Dropout Layer
- ReLU activation
- Dropout rate: **0.3**

#### Output Layer
- Binary classifier: sigmoid activation
- Multi-class classifier: softmax activation

---

### 3. Training Strategy

#### Stage 1 – Binary Classification
Detects whether a query is an SQL injection.

#### Stage 2 – Multi-Class Classification
Classifies only detected SQLi queries into specific attack types.

#### Training Enhancements

- **Early stopping**
  - Training stops if validation loss does not improve for 3 epochs.
- **Dropout regularization**
  - Reduces overfitting and improves generalization.

---

### 4. Inference Pipeline

For a given user query:

1. The query is preprocessed and tokenized.
2. The binary model predicts whether it is SQLi.
3. If malicious, the multi-class model predicts the attack type.
4. Probabilities for both stages are returned.

---

## 🧩 Algorithm (High-Level)

Input: SQL Query
↓
Preprocess query (lowercase, tokenize, map, pad)
↓
Embedding
↓
LSTM layers
↓
Dense + Dropout
↓
Binary classification
↓
If SQLi → Multi-class classification
↓
Return prediction and probabilities


---

## 📊 Dataset Preparation Summary

- Original dataset size: 30,919 queries
- After balancing:
  - Total samples: 9,000
  - Benign: 4,500
  - SQLi: 4,500
- Each SQL injection attack type is balanced to 500 samples.

---

## 📈 Experimental Results

### Binary Classification

- Test accuracy: **99.83%**

### Multi-Class Classification

- Test accuracy: **97.00%**

The binary model shows near-perfect performance in identifying malicious queries.  
The multi-class model consistently learns to distinguish different SQL injection attack categories.

---

## 🧪 Example Predictions

| Query | SQLi | Probability | Attack Type |
------|-----|-------------|------------
SELECT * FROM users | No | 0.0022 | Not SQLi
1'; DROP TABLE users; -- | Yes | 1.0000 | Error-based
SELECT * FROM products WHERE category='Electronics' | No | ~0.0000 | Not SQLi

---

## 🧱 Design Highlights

- Sequential modeling using LSTM to capture:
  - token ordering
  - long-term dependencies
  - syntactic variations in SQL statements
- Character-level representation enables detection of subtle injection patterns that token-level models may miss.

---

## 🧪 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

Both models are evaluated on held-out test data after training.

---

## ⚠️ Overfitting Control

The following strategies are used:

- Dropout layers
- Early stopping
- Balanced dataset generation

These techniques significantly improve generalization to unseen SQL queries.

---


## 🧩 Algorithm (High-Level)


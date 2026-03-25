# AI Quantum Solver: Machine Learning for Disordered XXZ Spin Chains

## 🚀 Overview

This project presents a **physics-informed machine learning pipeline** for predicting properties of a disordered quantum many-body system: the spin-1/2 XXZ chain.

The workflow combines:

* Exact diagonalization (C)
* Data generation from first principles
* Machine learning regression (Python)
* API deployment (FastAPI)
* Interactive UI (Streamlit)
* Containerized deployment (Docker & Docker Compose)

👉 The goal is to **replace expensive exact diagonalization with fast ML inference**.

---

## ⚛️ Physical Problem

We consider the disordered XXZ Hamiltonian:

$$
H = \sum_i \left( J (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y) + J_z S_i^z S_{i+1}^z \right) + \sum_i h_i S_i^z
$$

* Spin-1/2 chain
* Random onsite disorder $( h_i \in [-W, W] )$
* Fixed magnetization sector $K_{\uparrow}$

Exact diagonalization is performed in **symmetry-reduced Hilbert spaces**, enabling efficient dataset generation.

---
## 🧠 Key Idea

Instead of solving the Hamiltonian every time:

> 💥 Train a machine learning model to predict physical observables directly from system parameters.

---
## 🧠 Machine Learning Objective

Predict physical observables from disorder realizations:

### Targets

* Ground-state energy $E_0$
* Spectral gap $\Delta = E_1 - E_0$

### Key Result

* $R^2 \approx 0.96$ for $E_0$
* Gap prediction is significantly harder → reflects underlying many-body complexity

---

## 💡 Key Insight

The model learns **global disorder statistics** rather than site-specific values:

* $|h|_{\text{mean}}$
* variance
* standard deviation

👉 This reveals that:

> Ground-state energy is governed by global disorder properties, while the spectral gap depends on fine-grained many-body structure.

---

## 🏗️ Project Architecture

```
C (Exact Physics)
   ↓
Dataset (CSV)
   ↓
Python (ML Model)
   ↓
FastAPI (Backend)
   ↓
Streamlit (Frontend)
   ↓
Docker (Deployment)
```

---
## ⚙️ Components
## ⚙️ Components

### 1. 🔬 Exact Solver (C)

* Block-diagonal Hamiltonian construction (fixed magnetization)
* Bit-based state representation (no tensor products)
* Exact diagonalization (LAPACK)
* Efficient dataset generation

---

### 2. 📊 Machine Learning (Python)

* Features:

  * Disorder fields $h_i$
  * Statistical descriptors (mean, std, variance)
* Models:

  * Random Forest (baseline)
* Targets:

  * $E_0$
  * Gap

---

### 3. 🚀 FastAPI Backend

* Endpoint: `/predict`
* Input:

```json
{
  "Jxy": 1.0,
  "Jz": 1.2,
  "W": 3.0,
  "h": [...]
}
```

* Output:

```json
{
  "predicted_E0": -7.01
}
```

---

### 4. 🖥️ Streamlit Frontend

* Interactive input of parameters
* Real-time prediction
* User-friendly interface

---
---

## ⚙️ Tech Stack

* C (exact diagonalization)
* Python (NumPy, Pandas, Scikit-learn)
* FastAPI (REST API)
* Streamlit (UI)
* Docker & Docker Compose

---

## 🧪 Run the Project

### 🔥 One command

```bash
docker compose up --build
```

---

### 🌐 Access

* API docs: http://127.0.0.1:8000/docs
* Streamlit UI: http://127.0.0.1:8501

---

## 📡 API Usage

### Endpoint

```
POST /predict
```

### Example Input

```json
{
  "Jxy": 1.0,
  "Jz": 1.2,
  "W": 3.0,
  "h": [0.5, -1.2, 0.8, -0.3, 1.1, -0.7, 0.4, -0.9, 0.2, 0.6, -0.5, 1.0]
}
```

### Example Output

```json
{
  "predicted_E0": -7.02,
  "N": 12
}
```

---

## 🖥️ Streamlit Interface

* Input physical parameters
* Enter disorder configuration
* Instant prediction of ( E_0 )

---

## 📊 Results

| Quantity | Model Performance                 |
| -------- | --------------------------------- |
| $E_0$  | Excellent $R² ≈ 0.96$            |
| Gap      | Difficult (non-trivial structure) |

---

## 🧠 Scientific Value

This project demonstrates:

* Physics-informed feature engineering
* Learning emergent properties of many-body systems
* Limits of ML for chaotic quantum observables

---

## 🚀 Future Work

* Neural networks / XGBoost
* Prediction of localization (MBL classification)
* Physics-Informed Neural Networks (PINNs)
* Landau-Zener dataset extension
* Cloud deployment (AWS / Render)

---

## 👨‍🔬 Author

**Dr. Maseim Bassis Kenmoe**
Theoretical Physicist | AI & Data Science
Founder – Deep Quantum Science

---

## ⭐ Motivation

This project demonstrates how **AI can accelerate scientific computing**, bridging:

* quantum physics
* machine learning
* real-world software systems

---

## 📬 Contact

Feel free to connect for collaborations or opportunities.

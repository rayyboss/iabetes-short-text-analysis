# Application of Deep Learning Techniques for the Classification of Diabetes Using Short-Text Analysis

This repository contains the official implementation and research code for my undergraduate thesis in partial fulfillment of the requirements for the award of the degree of **Bachelor of Science in Mathematics** from the **University of Mines and Technology (UMaT)**.

## 📌 Project Overview
Patient-generated health data on social media offers vital textual insights for early detection and chronic disease monitoring. This project implements, evaluates, and benchmarks an integrated machine learning and deep learning framework to classify unstructured medical short-text data (1,877 curated Reddit samples) into three clinical categories: **Symptom**, **Treatment**, and **Complication**.

## 📊 Model Architectures & Performance Comparison
The research comparatively benchmarks a baseline statistical classifier against sequential and transformer-based deep learning networks:

1. **BERT (Bidirectional Encoder Representations from Transformers):** Fine-tuned using the `bert-base-uncased` architecture to capture bidirectional contextual semantic features. 
2. **Support Vector Machine (SVM):** Implemented with a Linear Kernel over sparse TF-IDF (Term Frequency-Inverse Document Frequency) text representations.
3. **CNN + Bi-LSTM (CLSTM):** A hybrid 1D-Convolutional Neural Network combined with a Bidirectional Long Short-Term Memory network to extract local n-gram spatial features and model temporal textual sequences concurrently.

### Experimental Performance Metrics Summary

| Performance Metric | Baseline SVM | Hybrid CNN + Bi-LSTM | Fine-Tuned BERT |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 86% | 74% | **88%** |
| **Weighted F1-Score** | 0.86 | 0.75 | **0.88** |
| **Complication AUC** | 0.96 | 0.89 | **0.96** |
| **Symptom AUC** | 0.96 | 0.85 | **0.96** |
| **Treatment AUC** | 0.97 | 0.90 | **0.96** |

*Key Insight:* Transformer-based architectures (BERT) excel heavily at extracting deep semantics from noisy, informal social media healthcare text, outperforming deep sequential combinations. On sparse, minor text datasets, classic high-dimensional statistical algorithms like SVM remain highly competitive benchmarks over complex neural layers prone to data constraints.

## 🛠️ Data Preprocessing Pipeline
* **Text Normalization:** Lowercasing, punctuation filtering, emoji removal, and slang normalization (e.g., mapping informal text variations to standard clinical expressions).
* **Tokenization & Lemmatization:** Structural morphological reductions executed via NLTK/SpaCy processing.
* **Vectorization Matrices:** 
  * *SVM:* Max feature TF-IDF vector blocks.
  * *CNN+Bi-LSTM:* Token integer-sequences padded uniformly to a length of 100.
  * *BERT:* Token IDs and multi-head attention masks padded/truncated to a max sequence length of 128 tokens.

## 💻 Tech Stack & Environment
* **Language:** Python 3.10
* **Frameworks:** PyTorch, TensorFlow 2.15, Hugging Face Transformers, Scikit-Learn
* **Data Engineering:** Pandas, NumPy, NLTK, SpaCy
* **Hardware:** Run entirely on Google Colab with GPU acceleration enabled.

## 🚀 How to Run the Environment
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   ```
2. **Install local dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute the classification model scripts:**
   ```bash
   python train_bert.py
   python train_svm.py
   ```

## 👨‍💻 Author & Attribution
* **Researcher:** Patrick Ray Agbo (BSc Mathematics Graduate, UMaT)
* **Project Advisor:** Assoc. Professor Joseph Acquah (Mathematical Science Programme, UMaT)
* **Affiliation:** School of Railways and Infrastructure Development, Essikado Campus, University of Mines and Technology, Tarkwa, Ghana.
* **Research Completed:** August, 2025
* linkedIn: https://www.linkedin.com/in/patrick-agbo-797b24321?utm_source=share_via&utm_content=profile&utm_medium=member_ios 

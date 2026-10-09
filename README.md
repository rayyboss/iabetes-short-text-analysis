# Diabetes Short-Text Analysis Using Hybrid Deep Learning

A hybrid deep learning framework combining BERT, CNN, and Bi-LSTM models for advanced classification and context analysis of diabetes-related short-text data. Built with Python and PyTorch.

## 📌 Project Overview
Short-text data (like clinical notes or patient queries) often suffers from data sparsity and lack of context. This project addresses those challenges by combining contextual embeddings with spatial and sequential neural networks to achieve high classification accuracy.

## 🛠️ Core Architecture
The pipeline leverages an ensemble architecture to capture both local semantic features and long-range text dependencies:
*   **BERT:** Extracts deep, bidirectional contextual word embeddings from medical text.
*   **CNN:** Extracts local n-gram features and structural patterns from the embeddings.
*   **Bi-LSTM:** Captures forward and backward sequential dependencies across text sequences.

## 💻 Tech Stack
*   **Language:** Python
*   **Libraries:** PyTorch, Hugging Face Transformers, Scikit-Learn, Pandas, NumPy
*   **Environment:** Jupyter Notebook / Google Colab 

## 👨‍💻 Author
**Patrick Ray Agbo**
*   BSc Mathematics | University of Mines and Technology (UMaT)
*   Email: patrickraymond398@gmail.com

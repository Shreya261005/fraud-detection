# 💳 Credit Card Fraud Detection – Machine Learning with SMOTE

This is a machine learning project for detecting fraudulent credit card transactions using **Python** and **Random Forest Classifier**. The dataset is highly imbalanced, so we apply **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the classes. The model is trained and evaluated in a Jupyter Notebook or Google Colab environment.

## 🔧 Features

- Uses real-world credit card transaction data
- Handles **class imbalance** using SMOTE
- Trains a **Random Forest** classifier
- Evaluates model performance using:
  - Confusion matrix
  - Precision, recall, F1-score
  - ROC-AUC score
- Visualizes:
  - Class distribution before and after SMOTE
  - Top 10 most important features

## 💡 What You’ll Learn

- How to handle imbalanced datasets in classification problems
- How to use SMOTE for oversampling minority classes
- Applying ensemble methods like Random Forest
- Validating model performance with appropriate metrics
- Visualizing feature importances for interpretability
- Using Python libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, and `imbalanced-learn`

## 🚀 How to Run

1. Make sure Python 3 is installed
2. Install required packages:

```bash
pip install imbalanced-learn seaborn scikit-learn matplotlib pandas numpy

# MI-SVM: Multiclass Imbalanced Support Vector Machine

## Description
MI-SVM is a multiclass classification approach designed for imbalanced datasets. It integrates Support Vector Machines with a kernel transformation technique to enhance minority class representation. The method employs pairwise class decomposition, hierarchical classification, and support vector–based boundary detection to improve classification performance.

---

## Features
- Handles multiclass imbalanced datasets
- Kernel transformation for improved class separation
- Pairwise class decomposition strategy
- Hierarchical classification mechanism
- Support vector–based boundary detection

---

## Project Structure
MI-SVM/
│
├── main.py              # Script to run experiments
├── misvm_model.py       # Core MI-SVM implementation
├── requirements.txt
└── README.md

---

## Requirements
- Python 3.x
- numpy
- pandas
- scikit-learn

Install dependencies:
pip install -r requirements.txt

---

## How to Run
1. Prepare your dataset (see preprocessing requirements below)
2. Update dataset loading in `main.py`
3. Run the script:

python main.py

---

## Data Preprocessing Requirements
This repository does not include preprocessing scripts, as preprocessing may vary depending on the dataset.

Users are expected to ensure:
- All input features are numeric
- No missing values are present
- Features are scaled/normalized if required
- Class labels are properly encoded (e.g., integers)
- Data is split into training and testing sets

---

## Input Format (Example)
Feature1, Feature2, Feature3, Label  
0.25, 1.45, 3.10, 1  
0.60, 2.30, 0.85, 2  

---

## Output
The model provides:
- Predicted class labels
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Training and testing time

---

## Reproducibility Note
Due to randomness and system-level differences, results may slightly vary from those reported in the study. However, the overall performance trends remain consistent.

---

## Citation
If you use this work, please cite the corresponding research paper.

---

## License
This project is intended for academic and research purposes.

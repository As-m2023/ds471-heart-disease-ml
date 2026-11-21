# Heart Disease Prediction — DS471 Machine Learning Project

Predicting angiographically confirmed coronary heart disease for 303 patients of the
[UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) dataset
with three traditional models and a neural network.

**Group members:** _[Student 1 – ID]_, _[Student 2 – ID]_, _[Student 3 – ID]_, _[Student 4 – ID]_

**GitHub repository:** https://github.com/As-m2023/ds471-heart-disease-ml

## Pipeline at a glance

| Stage | Approach |
|---|---|
| 1 · Preparation & EDA | Binary target (num ≥ 1); mode imputation of 6 missing values; rare `restecg` code merged; outliers capped at train-fitted IQR fences (linear model / NN only); engineered `ischaemia_signs` (count of 5 clinical warning signs, r = 0.71); one-hot encoding for `cp`, `slope`, `thal`; standardisation; chi-square, Mann–Whitney and mutual-information relevance |
| 2 · Splitting | Stratified 60 / 20 / 20 → 181 / 61 / 61 patients |
| 3 · Models | Logistic Regression, Decision Tree, Gradient Boosting, Keras MLP |
| 4 · Tuning | Grid / randomised search with repeated stratified 5-fold CV (ROC-AUC); NN random search; NN training/validation curves, overfitting experiment, learning curves, complexity curves |
| 5 · Evaluation | Accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrices, ROC/PR curves, bootstrap 95% CIs |

## Test-set results (61 patients, threshold 0.5)

| Model | CV AUC | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.891** | **0.885** | **0.889** | **0.857** | 0.909 | **0.873** | 0.929 |
| Decision Tree | 0.856 | 0.820 | 0.774 | 0.857 | 0.788 | 0.814 | 0.898 |
| Gradient Boosting | 0.883 | 0.852 | 0.880 | 0.786 | 0.909 | 0.830 | **0.949** |
| Neural Network | 0.892 | 0.869 | 0.885 | 0.821 | 0.909 | 0.852 | 0.919 |

Logistic Regression is the recommended model: consistently at the top across cross-validation, validation and test, and fully interpretable.

## Files

```
DS471_Heart_Disease_Prediction.ipynb   executed notebook (complete workflow)
data/processed.cleveland.data          raw UCI Cleveland file
figures/                               15 figures used in the report
results/                               CSV / JSON outputs
DS471_Project_Report.pdf               project report
requirements.txt
```

## Run it

Google Colab: upload the notebook **and** `data/processed.cleveland.data` (keep the `data/` folder), then *Run all*.
Locally:

```bash
pip install -r requirements.txt
jupyter notebook DS471_Heart_Disease_Prediction.ipynb
```

Seed 42 and deterministic TensorFlow ops make the results reproducible. A full run takes about 12 minutes on a 2-core CPU.

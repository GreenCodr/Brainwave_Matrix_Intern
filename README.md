# Brainwave_Matrix_Intern

# Fake News Detection (TF‑IDF + Logistic Regression)
A fast, strong baseline for fake news detection using TF‑IDF features and Logistic Regression, trained on the ISOT Fake & Real News dataset (True.csv, Fake.csv). The repo includes a Kaggle-ready notebook, clean training script, clear visualizations, and reproducible setup.

# Highlights
	•	End‑to‑end pipeline: load data, clean, vectorize (TF‑IDF), train Logistic Regression, evaluate, visualize, and save artifacts.
	•	Strong baseline: TF‑IDF + Logistic Regression is a proven, competitive approach for text classification and fake‑news baselines.
	•	Visuals included: confusion matrix, ROC, PR curve, top features for model interpretability.
	•	Kaggle + local ready: works in a Kaggle Notebook (Add Data → ISOT) and as a standalone Python script.
 
# Dataset
This project uses the ISOT Fake & Real News dataset from Kaggle, which provides two CSVs: Fake.csv and True.csv, each with title, text, subject, date columns; labels are constructed as fake=1, real=0. The dataset aggregates real articles (e.g., Reuters) and fake articles (sites flagged by fact-checkers) primarily from 2016–2017.
	•	Do not commit the dataset to the repo; instruct users to download it from Kaggle and place CSVs locally or attach it to a Kaggle Notebook session.
	•	Example paths in Kaggle after “Add data”:
	•	/kaggle/input/fake-and-real-news-dataset/Fake.csv
	•	/kaggle/input/fake-and-real-news-dataset/True.csv
 
# Project Structure
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── fake_news_kaggle.ipynb
├── src/
│   └── fake_news_detection.py
├── assets/
│   ├── cm.png
│   ├── roc.png
│   ├── pr.png
│   └── top_features.png
├── .gitignore
└── LICENSE

# Results
Example metrics from a typical run on ISOT (will vary by split and settings):
	•	Accuracy ≈0.99, F1 ≈0.99, ROC‑AUC ≈1.00 on the hold‑out set.
Included visualizations:
	•	Confusion Matrix, ROC Curve, Precision–Recall Curve, and Top Features bar charts.
 
# Visualizations
	•	Confusion Matrix (Real vs Fake).
	•	ROC Curve with AUC.
	•	Precision–Recall Curve with Average Precision.
	•	Top features (tokens) pushing toward Real (class 0) vs Fake (class 1) from the Logistic Regression coefficients.
These plots improve interpretability and presentation; they are produced via Matplotlib/Seaborn and sklearn’s metrics utilities.


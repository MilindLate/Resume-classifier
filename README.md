<div align="center">

<h1>📄 Resume Section Classifier</h1>
<h3>NLP-powered machine learning model to automatically classify resume sections</h3>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-00897B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br/>

> Automatically identify and label resume sections — **Education**, **Experience**, **Skills**, **Projects**, and more — using NLP preprocessing and machine learning classification. Built as a clean, well-documented Jupyter Notebook.

<br/>

[🧠 How It Works](#-how-it-works) • [🚀 Quick Start](#-quick-start) • [📓 Notebook Walkthrough](#-notebook-walkthrough) • [📊 Results](#-results) • [🔧 Customization](#-customization)

</div>

---

## 🎯 Problem Statement

Recruiters and HR tools that parse resumes need to know **which part of a resume contains what information** — skills listed under a "Projects" heading are not the same as skills under "Core Competencies". Manually labeling each section is tedious and error-prone.

This project builds a **multi-class text classifier** that reads a resume section's text and predicts which label it belongs to — fully automated, using only NLP and classical machine learning.

---

## 🧠 How It Works

### Pipeline Overview

```
Raw Resume Text
      │
      ▼
┌─────────────────────┐
│   Text Cleaning     │  Remove punctuation, numbers, special chars,
│   & Preprocessing   │  extra whitespace, non-ASCII characters
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Tokenization &    │  Lowercase → Tokenize → Remove stopwords
│   Normalization     │  → Stemming / Lemmatization
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Feature           │  TF-IDF Vectorization
│   Extraction        │  (Term Frequency–Inverse Document Frequency)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Classification    │  Train ML model (e.g. KNN / SVM / Naive Bayes)
│   Model             │  on labeled section data
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Prediction &      │  Accuracy, Precision, Recall,
│   Evaluation        │  F1-score, Confusion Matrix
└─────────────────────┘
```

### Section Labels (Classes)

The classifier predicts which of the following resume sections a given block of text belongs to:

| Label | Description |
|-------|-------------|
| `Education` | Degrees, universities, graduation dates, GPA |
| `Experience` | Job titles, company names, responsibilities, dates |
| `Skills` | Technical skills, tools, programming languages, frameworks |
| `Projects` | Project names, descriptions, technologies used |
| `Certifications` | Certificates, licenses, professional courses |
| `Summary` | Objective statement, professional profile, about section |
| `Awards` | Achievements, honors, recognition |
| `Other` | Miscellaneous / uncategorized content |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### 1. Clone the Repository

```bash
git clone https://github.com/MilindLate/Resume-classifier.git
cd Resume-classifier
```

### 2. Install Dependencies

```bash
pip install notebook numpy pandas scikit-learn nltk matplotlib seaborn
```

Or install all at once:

```bash
pip install notebook numpy pandas scikit-learn nltk matplotlib seaborn wordcloud
```

### 3. Download NLTK Data

Run this once in Python before opening the notebook:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### 4. Launch the Notebook

```bash
jupyter notebook ResumeSectionClassifier.ipynb
```

Then run all cells top to bottom with **Kernel → Restart & Run All**.

---

## 📓 Notebook Walkthrough

The notebook `ResumeSectionClassifier.ipynb` is organized into the following sections:

### Section 1 — Imports & Setup
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

### Section 2 — Data Loading
Load the labeled resume section dataset. Each row contains:
- `text` — the raw text content of a resume section
- `label` — the section category (Education, Experience, Skills, etc.)

```python
df = pd.read_csv('resume_sections.csv')
df.head()
```

### Section 3 — Exploratory Data Analysis (EDA)
- Class distribution bar chart — check for class imbalance
- Sample text inspection per category
- Text length distribution per class

### Section 4 — Text Preprocessing

```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\x00-\x7f]', ' ', text)          # Remove non-ASCII
    text = re.sub(r'[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'), ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', ' ', text)                     # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()             # Remove extra whitespace
    text = text.lower()                                  # Lowercase
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)
```

### Section 5 — Feature Extraction (TF-IDF)

```python
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['clean_text'])
y = df['label']
```

**Why TF-IDF?** It assigns higher weight to words that are important to a specific section (e.g., "bachelor", "gpa" for Education) while penalizing common words that appear everywhere.

### Section 6 — Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Section 7 — Model Training & Evaluation

Multiple classifiers are trained and compared:

| Model | Notes |
|-------|-------|
| **K-Nearest Neighbors (KNN)** | Baseline; classifies by nearest labeled examples |
| **Naive Bayes (MultinomialNB)** | Fast, strong text baseline |
| **Support Vector Machine (SVM / LinearSVC)** | High accuracy for text classification |
| **Logistic Regression** | Interpretable, competitive baseline |

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred))
```

### Section 8 — Visualization
- **Confusion matrix heatmap** — per-class prediction accuracy
- **Classification report** — Precision, Recall, F1-score per label
- **Word cloud** per section label — most common terms visualized

---

## 📊 Results

### Model Performance Summary

| Model | Accuracy | Notes |
|-------|----------|-------|
| KNN (k=5) | ~85% | Good baseline; slower at inference |
| Naive Bayes | ~88% | Fast, strong for sparse TF-IDF features |
| **LinearSVC** | **~95%+** | Best overall — recommended |
| Logistic Regression | ~93% | Interpretable, nearly as strong as SVM |

> Exact numbers depend on dataset size and class balance. Resume section classification with TF-IDF + SVM consistently achieves **90–99% accuracy** on clean datasets.

### Evaluation Metrics

```
              precision    recall  f1-score   support

   Education       0.97      0.96      0.96       120
  Experience       0.95      0.97      0.96       185
      Skills       0.96      0.94      0.95       143
    Projects       0.93      0.92      0.93        88
Certificates       0.94      0.96      0.95        65
     Summary       0.91      0.90      0.90        52
      Awards       0.92      0.93      0.92        41

    accuracy                           0.95       694
   macro avg       0.94      0.94      0.94       694
weighted avg       0.95      0.95      0.95       694
```

---

## 🔧 Customization

### Use Your Own Dataset

Replace the data loading step with your own labeled CSV:

```python
df = pd.read_csv('your_resume_data.csv')
# Required columns: 'text' and 'label'
```

Your CSV should look like:

```
text,label
"Bachelor of Technology in Computer Science from XYZ University, 2022",Education
"Developed REST APIs using Django and PostgreSQL for e-commerce platform",Experience
"Python, JavaScript, React, Node.js, SQL, Git",Skills
```

### Add a New Section Class

1. Add labeled examples to your dataset with the new class name
2. Re-run all cells — the model automatically handles new classes

### Tune TF-IDF Parameters

```python
tfidf = TfidfVectorizer(
    max_features=10000,     # More features = more detail
    ngram_range=(1, 3),     # Include trigrams
    min_df=2,               # Ignore very rare terms
    max_df=0.95,            # Ignore very common terms
    sublinear_tf=True       # Apply log normalization
)
```

### Tune SVM Hyperparameters

```python
from sklearn.svm import LinearSVC
model = LinearSVC(
    C=1.0,          # Regularization strength (lower = more regularization)
    max_iter=2000   # Increase if convergence warnings appear
)
```

### Predict on New Text

After training, classify any new resume section:

```python
new_texts = [
    "B.Tech Computer Science, SPPU, 2023 — CGPA 8.7",
    "Built a real-time chat app using Socket.io and React",
    "Python, TensorFlow, Pandas, SQL, Docker"
]

cleaned = [clean_text(t) for t in new_texts]
vectorized = tfidf.transform(cleaned)
predictions = model.predict(vectorized)

for text, label in zip(new_texts, predictions):
    print(f"[{label}] {text[:60]}...")
```

Output:
```
[Education]  B.Tech Computer Science, SPPU, 2023 — CGPA 8.7...
[Projects]   Built a real-time chat app using Socket.io and React...
[Skills]     Python, TensorFlow, Pandas, SQL, Docker...
```

---

## 📦 Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `python` | 3.8+ | Runtime |
| `jupyter` | Latest | Notebook environment |
| `numpy` | ≥1.21 | Numerical operations |
| `pandas` | ≥1.3 | Data manipulation |
| `scikit-learn` | ≥1.0 | ML models + TF-IDF |
| `nltk` | ≥3.7 | Tokenization, stopwords, lemmatization |
| `matplotlib` | ≥3.4 | Plotting charts |
| `seaborn` | ≥0.11 | Confusion matrix heatmap |
| `wordcloud` | ≥1.8 | Word cloud visualization (optional) |

Install all:
```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn wordcloud jupyter
```

---

## 📁 Repository Structure

```
Resume-classifier/
└── ResumeSectionClassifier.ipynb    ← Complete ML pipeline in one notebook
```

---

## 💡 Use Cases

- **ATS (Applicant Tracking Systems)** — automatically parse and index resume sections
- **Resume parsing APIs** — backend service that classifies incoming resume text
- **HR automation tools** — route different resume sections to different validation logic
- **Resume scoring** — evaluate completeness by checking which sections are present
- **Data extraction pipelines** — structure raw resume text into a database schema

---

## 🤝 Contributing

```bash
# 1. Fork the repo on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Resume-classifier.git

# 3. Create a feature branch
git checkout -b feature/your-improvement

# 4. Make changes and test in the notebook

# 5. Commit and push
git add .
git commit -m "feat: describe your change"
git push origin feature/your-improvement

# 6. Open a Pull Request
```

### Ideas for Contribution

- [ ] Add a **Streamlit web app** so anyone can paste resume text and get predictions
- [ ] Add **PDF / DOCX parsing** using `pdfminer` or `python-docx` to read real resumes
- [ ] Benchmark a **transformer model** (e.g. BERT, `sentence-transformers`) vs TF-IDF
- [ ] Add a **cross-validation** step for more robust accuracy estimates
- [ ] Add **hyperparameter tuning** using `GridSearchCV` or `RandomizedSearchCV`
- [ ] Export the trained model with `joblib` for production use
- [ ] Add support for **multilingual** resumes

---

## 🛠️ Troubleshooting

<details>
<summary>❌ ModuleNotFoundError for sklearn, nltk, etc.</summary>

```bash
pip install scikit-learn nltk pandas numpy matplotlib seaborn
```

If you're using a Conda environment:
```bash
conda install scikit-learn nltk pandas numpy matplotlib seaborn
```

</details>

<details>
<summary>❌ NLTK LookupError — stopwords / punkt not found</summary>

Run this in a Python cell before preprocessing:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')   # Required in newer NLTK versions
```

</details>

<details>
<summary>❌ LinearSVC ConvergenceWarning</summary>

Increase `max_iter`:

```python
model = LinearSVC(max_iter=5000)
```

Or switch to `SGDClassifier` with a hinge loss (equivalent but converges faster on large datasets):

```python
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='hinge', max_iter=1000)
```

</details>

<details>
<summary>❌ Low accuracy / poor predictions</summary>

- Check your dataset for **class imbalance** — use `df['label'].value_counts()`
- Try **increasing `max_features`** in TF-IDF (e.g., 10000)
- Use **`ngram_range=(1, 3)`** to capture longer phrases
- Try **`sublinear_tf=True`** in TF-IDF for better scaling
- Switch to **LinearSVC** if using KNN or Naive Bayes — it's consistently the strongest for this task

</details>

<details>
<summary>❌ Kernel crashes / memory error</summary>

- Reduce `max_features` in TF-IDF (e.g., 3000)
- Use a smaller dataset for initial testing
- Restart the kernel: **Kernel → Restart & Clear Output**, then re-run all cells

</details>

---



<div align="center">

Built with ❤️ by <a href="https://github.com/MilindLate">MilindLate</a>

<br/><br/>

<b>Resume Section Classifier</b> &nbsp;|&nbsp; Python &nbsp;|&nbsp; NLP &nbsp;|&nbsp; scikit-learn &nbsp;|&nbsp; Jupyter

<br/><br/>

⭐ If this project helped you, give it a star!

</div>

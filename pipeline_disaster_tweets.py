"""
NLP pipeline for "Disaster Tweets" and Sentiment140
Project: Análise de Texto na Plataforma X sobre Desastres Ambientais
Authors: Team (Karla, Livya, Paulo, Rafael)
This script implements the full pipeline described in the project methodology:
- data loading (expects CSVs)
- preprocessing (cleaning, tokenization, lemmatization)
- vectorization (TF-IDF)
- modelling (LogisticRegression, MultinomialNB, LinearSVC)
- evaluation (accuracy, precision, recall, f1, confusion matrix)
- saving artifacts (models, vectorizer)

USAGE
1) Install dependencies:
   pip install -r requirements.txt
   # requirements.txt include: pandas numpy scikit-learn matplotlib nltk spacy
2) Run:
   python pipeline_disaster_tweets.py --disaster_path data/train.csv --out_dir outputs
"""

import argparse
import os
import re
import string
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception:
    nlp = None

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk_packages = ["stopwords", "wordnet", "omw-1.4", "punkt"]
for pkg in nltk_packages:
    try:
        nltk.data.find(pkg)
    except Exception:
        try:
            nltk.download(pkg)
        except Exception:
            pass

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"&\w+;", "", text)
    text = re.sub(r"[^0-9A-Za-z ]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> str:
    if not text:
        return ""
    if nlp is not None:
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if (not token.is_stop and token.is_alpha)]
        return " ".join(tokens)
    else:
        tokens = nltk.word_tokenize(text)
        lemmas = []
        for t in tokens:
            t = t.lower()
            if t in STOPWORDS:
                continue
            if t.isalpha():
                lemmas.append(lemmatizer.lemmatize(t))
        return " ".join(lemmas)


def preprocess_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).map(clean_text)
    tokenized = cleaned.map(tokenize_and_lemmatize)
    return tokenized


def load_disaster(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"id", "text", "target"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Disaster CSV must contain columns: {expected}. Found: {df.columns}")
    return df


def train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer, out_dir: str):
    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "nb": MultinomialNB(),
        "svc": LinearSVC(max_iter=2000)
    }
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        results[name] = {"model": model, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm}
        # save model
        joblib.dump(model, os.path.join(out_dir, f"model_{name}.joblib"))
        print(f"{name} -> acc: {acc:.4f}, f1: {f1:.4f}")
    joblib.dump(vectorizer, os.path.join(out_dir, "tfidf_vectorizer.joblib"))
    return results


def plot_class_distribution(df: pd.DataFrame, label_col: str, out_dir: str):
    counts = df[label_col].value_counts().sort_index()
    plt.figure()
    counts.plot(kind="bar")
    plt.title("Class distribution")
    plt.xlabel("class")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"))
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, classes, out_path: str):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("Loading disaster dataset...")
    df = load_disaster(args.disaster_path)
    print(f"Loaded {len(df)} rows")

    plot_class_distribution(df, "target", out_dir)

    print("Preprocessing text (this may take some time)...")
    df["text_clean"] = preprocess_series(df["text"])  # cleaned + lemmatized tokens

    print("Vectorizing with TF-IDF...")
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vect.fit_transform(df["text_clean"]).astype(np.float32)
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = train_and_evaluate(X_train, X_test, y_train, y_test, vect, out_dir)

    summary_rows = []
    for name, res in results.items():
        cm_path = os.path.join(out_dir, f"cm_{name}.png")
        plot_confusion_matrix(res["cm"], classes=[0,1], out_path=cm_path)
        summary_rows.append({"model": name, "acc": res["acc"], "prec": res["prec"], "rec": res["rec"], "f1": res["f1"]})
    summary_df = pd.DataFrame(summary_rows).sort_values(by="f1", ascending=False)
    summary_df.to_csv(os.path.join(out_dir, "model_summary.csv"), index=False)
    print("Done. Artifacts saved in", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaster_path", type=str, required=True, help="Path to disaster train.csv")
    parser.add_argument("--out_dir", type=str, required=False, default="outputs", help="Output folder")
    args = parser.parse_args()
    main(args)

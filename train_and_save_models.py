"""
Train toy models or use provided CSV files to train models and save them into `model/`.

Usage:
 - If you have training CSVs with columns `ID`, `text`, `target`, place them and update paths below.
 - Otherwise the script will create a tiny synthetic dataset so you can test the Flask app.

Saved files (in `model/`):
 - logistic_model.pkl
 - lgbm_model.pkl
 - catboost_model.pkl
 - tfidf_word.pkl
 - tfidf_char.pkl

"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from scipy.sparse import hstack


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# If you have real CSV paths, set TRAIN_CSV and (optionally) TEST_CSV
TRAIN_CSV = None


def simple_clean(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    return ' '.join(text.split())


def make_or_load_data():
    if TRAIN_CSV and os.path.exists(TRAIN_CSV):
        df = pd.read_csv(TRAIN_CSV)
        if 'text' not in df.columns or 'target' not in df.columns:
            raise ValueError('CSV must have text and target columns')
        X = df['text'].astype(str).apply(simple_clean).values
        y = df['target'].values
        return X, y

    # Otherwise create a tiny synthetic dataset
    X = np.array([
        'I love this product',
        'This is the worst experience ever',
        'Amazing service and friendly staff',
        'I hate it, very bad',
        'Best purchase I made',
        'Terrible, not recommended',
        'Absolutely fantastic',
        'Do not buy this',
        'Happy with the quality',
        'Very disappointed'
    ])
    y = np.array([1,0,1,0,1,0,1,0,1,0])
    return X, y


def train_and_save():
    X, y = make_or_load_data()

    # TF-IDF
    tfidf_word = TfidfVectorizer(max_features=15000, ngram_range=(1,3), sublinear_tf=True, min_df=1)
    tfidf_char = TfidfVectorizer(max_features=5000, analyzer='char_wb', ngram_range=(3,5), sublinear_tf=True)

    X_word = tfidf_word.fit_transform(X)
    X_char = tfidf_char.fit_transform(X)
    X_all = hstack([X_word, X_char])

    X_train, X_hold, y_train, y_hold = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic
    lr = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    lr.fit(X_train, y_train)

    # LightGBM
    lgb = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    lgb.fit(X_train, y_train)

    # CatBoost
    cb = CatBoostClassifier(iterations=100, learning_rate=0.05, verbose=0)
    cb.fit(Pool(X_train, y_train))

    # Save
    joblib.dump(lr, os.path.join(MODEL_DIR, 'logistic_model.pkl'))
    joblib.dump(lgb, os.path.join(MODEL_DIR, 'lgbm_model.pkl'))
    joblib.dump(cb, os.path.join(MODEL_DIR, 'catboost_model.pkl'))
    joblib.dump(tfidf_word, os.path.join(MODEL_DIR, 'tfidf_word.pkl'))
    joblib.dump(tfidf_char, os.path.join(MODEL_DIR, 'tfidf_char.pkl'))

    print('Saved models to', MODEL_DIR)


if __name__ == '__main__':
    train_and_save()

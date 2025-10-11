import os
import re
import string
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# Basic English stop words list fallback
_STOP_WORDS = set([
    'a','an','the','and','or','but','if','while','of','at','by','for','with','about','against',
    'between','into','through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here','there','when',
    'where','why','how','all','any','both','each','few','more','most','other','some','such','no',
    'nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don',
    'should','now'
])


def load_models_and_vectorizers(model_dir: str):
    """Load models and tfidf vectorizers from a directory.

    Expects files: logistic_model.pkl, lgbm_model.pkl, catboost_model.pkl (or similar),
    and vectorizers like tfidf_word.pkl and tfidf_char.pkl. Uses joblib.load.
    Returns (models_dict, vectorizers_dict)
    """
    models = {}
    vectorizers = {}
    p = Path(model_dir)
    if not p.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Common filenames to look for
    candidates = list(p.glob('*.pkl')) + list(p.glob('*.joblib'))
    for f in candidates:
        name = f.stem.lower()
        try:
            obj = joblib.load(f)
        except Exception:
            # skip files that aren't joblib models
            continue

        if 'tfidf' in name and ('word' in name or 'char' in name):
            if 'word' in name:
                vectorizers['word'] = obj
            else:
                vectorizers['char'] = obj
        elif 'vectorizer' in name:
            # try to infer type
            if 'char' in name:
                vectorizers['char'] = obj
            else:
                vectorizers['word'] = obj
        elif 'logistic' in name or 'lr' in name or 'logreg' in name or 'logisticregression' in name:
            models['logistic'] = obj
        elif 'lgb' in name or 'lightgbm' in name or 'lgbm' in name:
            models['lgbm'] = obj
        elif 'cat' in name or 'catboost' in name:
            models['catboost'] = obj
        else:
            # try to guess by class
            cls_name = obj.__class__.__name__.lower()
            if 'log' in cls_name or 'logistic' in cls_name:
                models['logistic'] = obj
            elif 'lgb' in cls_name or 'lightgbm' in cls_name:
                models['lgbm'] = obj
            elif 'catboost' in cls_name or 'cat' in cls_name:
                models['catboost'] = obj
            elif isinstance(obj, TfidfVectorizer):
                # guess word vs char by analyzer or ngram
                if getattr(obj, 'analyzer', '') == 'char' or getattr(obj, 'ngram_range', (1,1))[1] >= 4:
                    vectorizers['char'] = obj
                else:
                    vectorizers['word'] = obj

    # Provide defaults if missing (raise error if no models)
    if not models:
        raise RuntimeError(f'No models found in {model_dir}. Expected logistic, lgbm, and catboost models saved as .pkl or .joblib')

    # Ensure both vectorizers exist; if only one exists, duplicate it for both
    if 'word' not in vectorizers and 'char' in vectorizers:
        vectorizers['word'] = vectorizers['char']
    if 'char' not in vectorizers and 'word' in vectorizers:
        vectorizers['char'] = vectorizers['word']

    return models, vectorizers


def clean_text(text: str) -> str:
    """Clean input text: remove URLs, mentions, hashtags, punctuation, stop words, and extra spaces."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    # remove urls
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # remove mentions and hashtags symbols but keep the words
    text = re.sub(r'[@#]+', ' ', text)
    # remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # remove numbers
    text = re.sub(r'\d+', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # remove stop words
    tokens = [t for t in text.split() if t not in _STOP_WORDS]
    return ' '.join(tokens)


def predict_ensemble(text: str, models: dict, vectorizers: dict) -> float:
    """Vectorize text using both word and char TF-IDF, get prediction probabilities from each model,
    average them, and return the positive-class probability.
    """
    if not text:
        return 0.0

    # Combine features: word and char TF-IDF vectors horizontally
    word_vec = vectorizers['word'].transform([text]) if 'word' in vectorizers else None
    char_vec = vectorizers['char'].transform([text]) if 'char' in vectorizers else None

    # Some models may expect only one of the vectors. We'll try both and let each model accept what it can.
    probs = []

    for key, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                # Try concatenated features if both exist
                if word_vec is not None and char_vec is not None:
                    # concatenate sparse matrices if needed
                    try:
                        from scipy.sparse import hstack
                        X = hstack([word_vec, char_vec])
                    except Exception:
                        X = np.hstack([word_vec.toarray(), char_vec.toarray()])
                elif word_vec is not None:
                    X = word_vec
                else:
                    X = char_vec

                p = model.predict_proba(X)[0]
                # assume binary classification; positive class is class 1
                if len(p) == 1:
                    prob = float(p[0])
                else:
                    prob = float(p[1])
                probs.append(prob)
            else:
                # fallback to predict and map to 0/1
                pred = model.predict([text])[0]
                probs.append(float(pred))
        except Exception:
            # if a model fails, skip it
            continue

    if not probs:
        return 0.0

    return float(np.mean(probs))

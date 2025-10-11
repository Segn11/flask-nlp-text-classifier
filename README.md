# Flask NLP Ensemble Classifier

This repository contains a small Flask app that loads pre-trained models and TF-IDF vectorizers from a `model/` directory and serves a web UI for text classification.

Expected model files (place into `model/`):
- logistic regression model (e.g. `logistic_model.pkl` or similar)
- LightGBM model (e.g. `lgbm_model.pkl`)
- CatBoost model (e.g. `catboost_model.pkl`)
- TF-IDF vectorizers: word-level (e.g. `tfidf_word.pkl`) and char-level (e.g. `tfidf_char.pkl`). If only one is present, it will be used for both.

Files added:
- `app.py` — Flask application entrypoint
- `utils.py` — helpers: load models, clean text, vectorize and ensemble predictions
- `templates/index.html` — frontend template
- `static/style.css` — basic styling
- `requirements.txt` — Python dependencies

How to run:
1. Create a virtual environment and install requirements:

   python -m venv myenv; .\myenv\Scripts\activate; pip install -r requirements.txt

2. Place your trained models and vectorizers into the `model/` folder (see expected names above).

3. Run the app:

   python app.py

4. Open http://127.0.0.1:5000 in your browser.

Notes:
- The app cleans input text, vectorizes with TF-IDF (word and char), obtains probabilities from the three models, averages them, and shows Positive (>= 0.5) or Negative (< 0.5).
- If any model fails to load or predict, the app will skip it. Ensure models implement `predict_proba` for proper probabilities.

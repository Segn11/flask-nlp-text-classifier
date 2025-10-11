from flask import Flask, render_template, request
import os
from utils import load_models_and_vectorizers, clean_text, predict_ensemble

app = Flask(__name__)

# load models from model/ directory (expects model.pkl files and vectorizers)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
models, vectorizers = load_models_and_vectorizers(MODEL_DIR)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    prob = None
    text = ''

    if request.method == 'POST':
        text = request.form.get('text', '')
        cleaned = clean_text(text)
        prob = predict_ensemble(cleaned, models, vectorizers)
        label = '✅ Positive' if prob >= 0.5 else '❌ Negative'
        result = {'label': label, 'probability': round(float(prob), 4)}

    return render_template('index.html', result=result, text=text)


if __name__ == '__main__':
    app.run(debug=True)

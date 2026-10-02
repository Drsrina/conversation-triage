"""Predição com modelos treinados."""

import os
import joblib
from typing import List, Dict

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


class TriageClassifier:
    """Carrega modelos salvos e expõe método de predição."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
        self.sentiment_model = joblib.load(os.path.join(model_dir, 'sentiment_model.pkl'))
        self.tags_model = joblib.load(os.path.join(model_dir, 'tags_model.pkl'))
        self.mlb = joblib.load(os.path.join(model_dir, 'mlb.pkl'))

    def predict(self, text: str) -> Dict:
        """Retorna sentimento e tags para uma conversa."""
        v = self.vectorizer.transform([text])
        sentiment = self.sentiment_model.predict(v)[0]
        tags_raw = self.tags_model.predict(v)[0]
        tags = [self.mlb.classes_[i] for i, val in enumerate(tags_raw) if val]
        return {'sentiment': sentiment, 'tags': tags}


def predict(text: str) -> Dict:
    """Função de conveniência."""
    classifier = TriageClassifier()
    return classifier.predict(text)
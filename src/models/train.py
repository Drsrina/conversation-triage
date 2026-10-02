"""Treino do classificador: sentimento (single-label) + tags (multi-label)."""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    classification_report,
)

from src.data.preprocess import preprocess_df, split_tags

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_baseline(df: pd.DataFrame, text_col: str = 'text',
                   sentiment_col: str = 'sentiment',
                   tags_col: str = 'tags'):
    """Treina baseline TF-IDF + Logistic Regression para sentimento e tags."""
    df = preprocess_df(df, col_text=text_col)

    # Split
    X = df['text_clean']
    y_s = df[sentiment_col]
    y_t = split_tags(df[tags_col])

    X_train, X_test, y_s_train, y_s_test, y_t_train, y_t_test = train_test_split(
        X, y_s, y_t, test_size=0.2, random_state=42, stratify=y_s
    )

    # Vectorizer
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    # --- Sentimento ---
    clf_s = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_s.fit(X_train_vec, y_s_train)
    y_s_pred = clf_s.predict(X_test_vec)

    sent_metrics = {
        'accuracy': accuracy_score(y_s_test, y_s_pred),
        'f1_macro': f1_score(y_s_test, y_s_pred, average='macro'),
        'report': classification_report(y_s_test, y_s_pred, output_dict=True),
    }

    # --- Tags (multi-label) ---
    mlb = MultiLabelBinarizer()
    y_t_train_enc = mlb.fit_transform(y_t_train)
    y_t_test_enc = mlb.transform(y_t_test)

    clf_t = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    clf_t.fit(X_train_vec, y_t_train_enc)
    y_t_pred = clf_t.predict(X_test_vec)

    tags_metrics = {
        'hamming_loss': hamming_loss(y_t_test_enc, y_t_pred),
        'f1_samples': f1_score(y_t_test_enc, y_t_pred, average='samples'),
        'f1_macro': f1_score(y_t_test_enc, y_t_pred, average='macro'),
    }

    # Salvar artefatos
    joblib.dump(vec, os.path.join(MODEL_DIR, 'vectorizer.pkl'))
    joblib.dump(clf_s, os.path.join(MODEL_DIR, 'sentiment_model.pkl'))
    joblib.dump(clf_t, os.path.join(MODEL_DIR, 'tags_model.pkl'))
    joblib.dump(mlb, os.path.join(MODEL_DIR, 'mlb.pkl'))

    return {
        'sentiment_model': clf_s,
        'tags_model': clf_t,
        'vectorizer': vec,
        'mlb': mlb,
        'sent_metrics': sent_metrics,
        'tags_metrics': tags_metrics,
        'X_test': X_test,
        'y_s_test': y_s_test,
        'y_t_test': y_t_test,
    }


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'conversas_ptbr.csv')
    if os.path.exists(data_path):
        print(f"Carregando dados reais de: {data_path}")
        data = pd.read_csv(data_path)
    else:
        print("Dados reais não encontrados, utilizando dados dummy...")
        data = pd.DataFrame({
            'text': [
                'Não consigo fazer login, o sistema dá erro!',
                'Obrigado pelo atendimento, tudo perfeito!',
                'Preciso de reembolso da última compra',
                'O app crashes ao abrir a tela de perfil',
            ],
            'sentiment': ['negativo', 'positivo', 'neutro', 'negativo'],
            'tags': [
                'login|problema_técnico',
                'atendimento',
                'reembolso',
                'problema_técnico|app',
            ],
        })

    result = train_baseline(data)
    print('\n=== SENTIMENTO ===')
    print(f"Accuracy: {result['sent_metrics']['accuracy']:.4f}")
    print(f"F1 macro: {result['sent_metrics']['f1_macro']:.4f}")
    print('\n=== TAGS ===')
    print(f"Hamming loss: {result['tags_metrics']['hamming_loss']:.4f}")
    print(f"F1 samples: {result['tags_metrics']['f1_samples']:.4f}")
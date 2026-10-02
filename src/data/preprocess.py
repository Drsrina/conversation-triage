"""Pré-processamento de texto para conversas de atendimento."""

import re
import string
from typing import List

import pandas as pd
import nltk
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

# Garante que os recursos do NLTK estão disponíveis
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('stemgers/rslp')
except LookupError:
    nltk.download('rslp', quiet=True)

_stemmer = RSLPStemmer()


def limpar_texto(texto: str) -> str:
    """Remove URLs, menções, emojis e normaliza para minúsculo."""
    if not isinstance(texto, str):
        return ""

    # Remove URLs
    texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto)

    # Remove menções e hashtags
    texto = re.sub(r'@\w+|#\w+', ' ', texto)

    # Remove emojis (intervalo Unicode amplo)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926\U0001f937\U0001f938\U0001f939"
        "\U0001f93a\U0001f93b\U0001f93c\U0001f93d"
        "\U0001f93e\U0001f93f\U0001f930\U0001f931"
        "\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]",
        flags=re.UNICODE,
    )
    texto = emoji_pattern.sub(' ', texto)

    # Minúsculas
    texto = texto.lower()

    # Remove pontuação
    texto = texto.translate(str.maketrans('', '', string.punctuation))

    # Remove espaços extras
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto


def stemming(texto: str) -> str:
    """Aplica stemming português (RSLP)."""
    tokens = word_tokenize(texto, language='portuguese')
    return ' '.join(_stemmer.stem(t) for t in tokens)


def preprocess_df(df: pd.DataFrame, col_text: str = 'text') -> pd.DataFrame:
    """Aplica todo o pipeline de limpeza a um DataFrame."""
    df = df.copy()
    df['text_clean'] = df[col_text].apply(limpar_texto)
    df['text_stem'] = df['text_clean'].apply(stemming)
    return df


def split_tags(series: pd.Series, sep: str = '|') -> pd.Series:
    """Converte tags em lista de labels."""
    return series.apply(
        lambda x: [t.strip() for t in str(x).split(sep) if t.strip()]
        if isinstance(x, str) else []
    )
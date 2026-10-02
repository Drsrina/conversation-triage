# 🧠 Conversation Triage

Repositório de aprofundamento e estudo, sobre Processos de Machine Learning e NLP.

Classificador de conversas de atendimento: **sentimento** (single-label) + **tags** (multi-label), servido como API FastAPI no Cloud Run.

## O que faz

Recebe uma conversa de atendimento e retorna:
- **Sentimento**: `positivo` / `neutro` / `negativo`
- **Tags** (multi-label): `reembolso`, `login`, `problema_técnico`, `cobrança`, etc.

## Stack

- **Modelo**: TF-IDF + Logistic Regression (baseline) → DistilBERT (M2)
- **API**: FastAPI + Pydantic
- **Container**: Docker
- **Deploy**: Google Cloud Run
- **Dados**: BigQuery
- **Dashboard**: Looker Studio / Streamlit

## Estrutura

```
conversation-triage/
├── notebooks/
│   └── 01_baseline.ipynb
├── src/
│   ├── data/preprocess.py
│   ├── models/train.py
│   ├── models/predict.py
│   └── api/main.py
├── static/index.html
├── data/raw/          (gitignored)
├── data/processed/    (gitignored)
├── models/            (gitignored)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Como rodar localmente

```bash
# 1. Treinar modelo
python -m src.models.train

# 2. Rodar API
uvicorn src.api.main:app --reload

# 3. Testar
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"conversation": "Não consigo fazer login"}'
```

## Métricas (M1: Baseline TF-IDF + Regressão Logística)

| Métrica | Valor |
|---|---|
| Accuracy (sentimento) | 0.7684 (76.84%) |
| F1 macro (sentimento) | 0.7644 (76.44%) |
| Hamming loss (tags) | 0.1019 |
| F1 samples (tags) | 0.1789 |


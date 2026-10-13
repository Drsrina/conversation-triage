Este repositório é o **[Conversation Triage](file:///c:/Users/Drysrina/Desktop/Code/conversation-triage/README.md)**: um projeto de estudo e aplicação prática de **Machine Learning** e **Processamento de Linguagem Natural (NLP)** focado na **triagem e classificação automática de conversas de atendimento/suporte ao cliente**.

---

### 🎯 O que o projeto faz

Dado o texto de uma conversa ou mensagem de atendimento (ex.: *"Não consigo acessar minha conta"*), o sistema faz duas predições simultâneas:

1. **Sentimento (Single-label):** classifica a mensagem como `positivo`, `neutro` ou `negativo`.
2. **Tags de Tópicos (Multi-label):** identifica categorias aplicáveis, como `login`, `reembolso`, `problema_técnico`, `cobrança`, etc.

---

### 🛠️ Arquitetura e Stack

- **Machine Learning & NLP:**
  - *Baseline:* Vetorização com **TF-IDF** (unigramas/bigramas) + **Regressão Logística** via Scikit-Learn.
  - *Evolução (M2):* Fine-tuning de Transformers (**DistilBERT**).
- **Backend / API:** [FastAPI](file:///c:/Users/Drysrina/Desktop/Code/conversation-triage/src/api/main.py) com validação via Pydantic para expor o endpoint de inferência (`/predict`).
- **Containerização & Deploy:** [Dockerfile](file:///c:/Users/Drysrina/Desktop/Code/conversation-triage/Dockerfile) preparado para deploy no **Google Cloud Run**.
- **Fundamentos & Teoria:** Documentação detalhada em [1processo/1-Detalhamento dos Métodos.md](file:///c:/Users/Drysrina/Desktop/Code/conversation-triage/1processo/1-Detalhamento%20dos%20M%C3%A9todos.md) explicando por que o pipeline funciona (Teorema de Cover, Teoria da Informação de Shannon para IDF, Estimativa de Máxima Verossimilhança e Bayes).

---

### 📁 Estrutura do Projeto

- **[src/](file:///c:/Users/Drysrina/Desktop/Code/conversation-triage/src/)**: Código-fonte modular:
  - `src/data/`: Pré-processamento e limpeza de texto.
  - `src/models/`: Scripts de treino (`train.py`) e inferência (`predict.py`).
  - `src/api/`: Aplicação FastAPI (`main.py`).
- **[notebooks/](file:///c:/Users/Drysrina/Desktop/Code/conversation-triage/notebooks/)**: Notebooks de experimentação e análise exploratória (ex.: `01_baseline.ipynb`).
- **[1processo/](file:///c:/Users/Drysrina/Desktop/Code/conversation-triage/1processo/)**: Documentação do processo de concepção, fundamentação matemática e escolha de datasets.
- **[static/](file:///c:/Users/Drysrina/Desktop/Code/conversation-triage/static/)**: Interface web estática simples para testes visuais.
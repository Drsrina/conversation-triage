# 📦 Datasets Candidatos — Conversation Triage

Este documento cataloga e avalia os conjuntos de dados em português (pt-BR) analisados para o treinamento, validação e benchmark dos modelos do **Conversation Triage**.

---

## 🎯 Critérios de Avaliação

Para atender aos requisitos da arquitetura do projeto, os datasets foram avaliados sob quatro dimensões principais:

1. **Idioma & Registro Linguístico:** Conversas e atendimentos reais ou realistas em português brasileiro (pt-BR).
2. **Anotação de Sentimento:** Presença de classes de polaridade (`positivo`, `neutro`, `negativo`) ou dados que permitam mapeamento direto.
3. **Classificação Temática (*Multi-label*):** Categorização por intenções ou áreas (ex.: `login`, `reembolso`, `dúvidas`, `suporte técnico`).
4. **Viabilidade de Ingestão:** Disponibilidade pública, formato estruturado e facilidade de carga via Python.

---

## 🗂️ Análise Detalhada dos Datasets

### 1. `RichardSakaguchiMS/brazilian-customer-service-conversations`
> **Status:** 🏆 **Dataset Principal (Em Uso no Projeto)**  
> **Origem:** [Hugging Face Datasets](https://huggingface.co/datasets/RichardSakaguchiMS/brazilian-customer-service-conversations)

* **Descrição:** Diálogos e mensagens contextualizadas de atendimento ao cliente em português do Brasil cobrindo múltiplos setores da economia (e-commerce, serviços financeiros, telecomunicações e serviços essenciais).
* **Por que foi escolhido:**
  * Apresenta anotações nativas de **sentimento** (`positivo`, `neutro`, `negativo`).
  * Inclui tags de **intenções e tópicos** (ex.: `reclamação`, `dúvida`, `elogio`, `cancelamento`, `cobrança`), alinhando-se exatamente à tarefa multirrótulo do pipeline.
  * Estrutura de fácil conversão e pré-processamento para pipelines Scikit-Learn e PyTorch/Hugging Face.

#### 💻 Como carregar e salvar no projeto:
```python
from datasets import load_dataset
import pandas as pd

# Download direto via Hugging Face Hub
dataset = load_dataset("RichardSakaguchiMS/brazilian-customer-service-conversations")

# Conversão para DataFrame Pandas
df = dataset["train"].to_pandas()

# Exportação para a pasta de dados brutos
df.to_csv("data/raw/conversas_ptbr.csv", index=False)
print(f"Dataset salvo com sucesso! Total de linhas: {len(df)}")
```

---

### 2. B2W-Reviews01
> **Status:** 📚 **Opção para Pré-treino e Benchmark de Sentimento**  
> **Origem:** [Kaggle — B2W-Reviews01](https://www.kaggle.com/datasets/fredericods/ptbr-b2w-reviews01) / GitHub

* **Descrição:** Mais de 130.000 avaliações de clientes coletadas em plataformas de e-commerce brasileiras (Americanas, Submarino, Shoptime).
* **Campos Principais:**
  * `review_text`: Corpo do texto da avaliação.
  * `overall_rating`: Nota de 1 a 5 estrelas.
  * `reviewer_gender`, `category`: Metadados do produto e categoria.
* **Mapeamento de Polaridade:**
  $$\text{Sentimento} = \begin{cases} \text{negativo}, & \text{se } \text{rating} \in \{1, 2\} \\ \text{neutro}, & \text{se } \text{rating} = 3 \\ \text{positivo}, & \text{se } \text{rating} \in \{4, 5\} \end{cases}$$
* **Vantagens:** Reconhecido como o principal benchmark aberto para NLP em língua portuguesa no contexto de varejo/consumo.
* **Limitações:** O estilo textual reflete avaliações de produtos (*reviews*), que possuem nuances sintáticas distintas de uma interação conversacional de suporte em tempo real.

---

### 3. Reclamações Reclame Aqui & Consumidor.gov
> **Status:** ⚠️ **Base Complementar para Casos Críticos e Detecção de Fraude/Cobrança**  
> **Origem:** Kaggle / Portais de Dados Abertos

* **Descrição:** Histórico de reclamações e solicitações de mediação de consumidores em diversos segmentos de mercado.
* **Campos Principais:**
  * `titulo`: Assunto do problema relatado.
  * `descricao`: Detalhamento circunstanciado da ocorrência.
  * `categoria` / `assunto`: Motivos pré-categorizados (ex.: *Estorno de cobrança*, *Falha de autenticação*, *Atraso de entrega*).
  * `status_resolucao`: Indicador se a disputa foi sanada pelo prestador.
* **Pontos de Atenção:**
  * **Forte viés de sentimento:** Mais de $95\%$ dos textos são de polaridade estritamente negativa.
  * Requer filtragem cuidadosa para evitar desbalanceamento severo no classificador de sentimento, sendo mais recomendado para enriquecer o classificador de tópicos e triagem de urgência.

---

### 4. Multilingual Customer Support Ticket Classification
> **Status:** 🌐 **Candidato para Expansão Multilíngue**  
> **Origem:** Kaggle

* **Descrição:** Coleção internacional de tickets de suporte técnico abrangendo vários idiomas, contendo uma expressiva partição em português.
* **Campos Principais:**
  * `ticket_text`: Descrição do chamado de suporte.
  * `queue` / `department`: Fila de atendimento de destino.
  * `priority`: Nível de urgência (`baixa`, `média`, `alta`, `crítica`).
  * `ticket_type`: Classificação funcional do chamado.
* **Vantagens:** Ideal para cenários em que o sistema de triagem precisa rotear tickets não apenas por assunto, mas também estimar prioridade operacional e SLA.

---

## 📊 Matriz Comparativa dos Datasets

| Dataset | Fonte | Volume | Sentimento Nativo | Multi-label / Tópicos | Estilo do Texto | Recomendação no Projeto |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Brazilian Customer Service** | Hugging Face | Médio | Sim | Sim | Conversa / Diálogo | **Principal (Baseline & Treino)** |
| **B2W-Reviews01** | Kaggle / GitHub | $> 130\text{k}$ | Via estrelas | Categoria de produto | Review de produto | Benchmark de Sentimento |
| **Reclame Aqui / Consumidor.gov** | Kaggle | Alto | Fortemente enviesado | Sim (categorias reais) | Reclamação formal | Enriquecimento de tópicos complexos |
| **Multilingual Tickets** | Kaggle | Variado | Parcial | Sim (departamentos) | Ticket de suporte | Expansão multilíngue / Priorização |

---

## 🚀 Próximos Passos de Engenharia de Dados

1. **Ingestão Automatizada:** Script reprodutível em `scripts/download_data.py` utilizando `datasets` e validação de schema.
2. **Estratégia de Data Augmentation:** Avaliar técnicas de retrotradução (*back-translation*) ou expansão sintética para balanceamento de classes minoritárias de sentimento e categorias raras.
3. **Data Splits:** Separação estratificada de $70\%$ treino, $15\%$ validação e $15\%$ teste com garantia de distribuição consistente entre rótulos multirrótulo via `IterativeStratifiedKFold`.
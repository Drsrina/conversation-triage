## Introdução ao Projeto Conversation Triage

### Sobre este projeto (conversation-triage)
Você está no repositório Conversation Triage 
É uma projeto de aplicação de Machine Learning e API para triagem e classificação de conversas de suporte/atendimento:

O que faz:
Análise de Sentimento (positivo, neutro, negativo).
Classificação de Tags / Categorias (multi-label, como login, reembolso, problema_técnico, cobrança).
Tecnologias utilizadas:
ML: Scikit-Learn (TF-IDF + Regressão Logística) e suporte a modelos Transformer (DistilBERT).
Backend/API: 

FastAPI
 servindo endpoints de predição (/predict).
Deploy/Container: Docker e Google Cloud Run.

---
### Teoremas Aplicados

1. Teorema de Cover (Cover's Theorem on Separability of Patterns) — O principal motivo de funcionar
O que diz: Um problema de classificação que não é linearmente separável em um espaço de baixa dimensão torna-se, com alta probabilidade, linearmente separável quando projetado em um espaço de dimensionalidade muito maior.
Como usamos aqui: O texto puro é complexo e não-linear. Ao aplicar o TF-IDF com unigramas e bigramas, projetamos cada frase em um espaço vetorial esparso de 5.000 dimensões. Por conta do Teorema de Cover, um classificador estritamente linear (a Regressão Logística) consegue separar classes e tags com alta eficácia nesse espaço de alta dimensão, mesmo sendo um modelo simples.

2. Teorema de Bayes e Probabilidade Condicional
O que diz: Relaciona a probabilidade a priori, a verossimilhança e a probabilidade a posteriori: $$P(Y = c \mid X) = \frac{P(X \mid Y = c) \cdot P(Y = c)}{P(X)}$$
Como usamos aqui: A Regressão Logística modela diretamente a probabilidade a posteriori $P(\text{sentimento} \mid \text{texto})$ via função sigmoide/softmax. Em classificação probabilística e NLP, esse é o pilar para calibrar o limiar (threshold) de decisão das predições.
3. Teorema da Máxima Verossimilhança (Maximum Likelihood Estimation - MLE) e Otimização Convexa
Como usamos aqui: O treinamento da Regressão Logística resolve um problema de otimização convexa:
A função de perda (Log-Loss / Entropia Cruzada com regularização $L_2$) é estritamente convexa.
Pelo teorema de otimização convexa, qualquer mínimo local encontrado pelo otimizador (lbfgs) é garantidamente o mínimo global ótimo.
Pelas propriedades assintóticas de estimadores de máxima verossimilhança, os pesos convergem para estimativas consistentes e eficientes.
4. Teoria da Informação de Shannon (Fundamento do TF-IDF)
Como usamos aqui: O cálculo do IDF (Inverse Document Frequency): $$\text{IDF}(t) = \log\left(\frac{N}{\text{DF}(t)}\right)$$ baseia-se no conceito de auto-informação (surprisal) de Claude Shannon: palavras muito raras carregam alta quantidade de informação/especificidade semântica, enquanto palavras ubíquas (como "de", "a", "o") carregam informação próxima de zero.

### Legendas
NLP com TF-IDF + Regressão Logística
TF-IDF (Term Frequency – Inverse Document Frequency)
Regressão Logística (Logistic Regression)
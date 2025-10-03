# environmental-disaster-analysis-x-data
Análise de dados do X (antigo Twitter) para classificar publicações sobre desastres ambientais usando  Machine Learning.

## 👩‍💻 Autores

* **Karla Maria Ramos da Silva** – RA: 10730503
* **Livya Kaiser de Albuquerque** – RA: 10433409
* **Paulo Henrique Lasmar Teles** – RA: 10728776
* **Rafael Hessel Sichetti** – RA: 10375395

---

# PROJETO APLICADO II:

## PROJETO DE ANÁLISE DE TEXTO NA PLATAFORMA X SOBRE DESASTRES AMBIENTAIS

---

## 🎯 Objetivo

Investigar como os usuários da rede social **X (antigo Twitter)** se expressam em contextos de **desastres ambientais**, por meio da análise automática de tweets.

> **Hipóteses:**
>
> * Tweets relacionados a desastres ambientais apresentam padrões linguísticos específicos que podem ser detectados automaticamente.
> * Técnicas de ciência de dados e aprendizado de máquina podem apoiar o monitoramento de informações críticas em tempo real.

---

## 📊 Dados Utilizados

* **Sentiment140** (Stanford University) – 1,6 milhão de tweets rotulados em **positivo, negativo e neutro**.
* **Disaster Tweets (Kaggle Competition)** – 10 mil tweets classificados como **relacionados** ou **não relacionados** a desastres ambientais.

**Metadados principais:**

* Texto do tweet
* Polaridade ou rótulo (positivo/negativo/neutro; desastre/não desastre)
* ID do tweet
* Localização (quando disponível)
* Palavras-chave e hashtags associadas

---

## 📁 Estrutura do Repositório

```
.
├── dados/            # Conjunto de dados brutos e tratados
├── notebooks/        # Análises em Jupyter Notebook
├── scripts/          # Scripts auxiliares (pré-processamento e modelos)
├── requirements.txt  # Bibliotecas necessárias
└── README.md         # Documentação do projeto
```

---

## 📌 Metodologia

1. **Coleta**: Uso das bases abertas *Sentiment140* e *Disaster Tweets*.
2. **Tratamento**: Limpeza dos textos, remoção de stopwords, links, emojis e lematização.
3. **Exploração**: Estatísticas descritivas, gráficos de distribuição e nuvens de palavras.
4. **Modelagem**: Aplicação de algoritmos de aprendizado de máquina (Naive Bayes, Regressão Logística, SVM e Redes Neurais).
5. **Avaliação**: Cálculo de métricas como acurácia, precisão, recall, F1-score e matriz de confusão.

---

## 🔧 Ambiente de Desenvolvimento

Projeto desenvolvido em **Jupyter Notebook**, utilizando **Python 3.13**.

**Bibliotecas principais:**

* `pandas`, `numpy` – Manipulação de dados
* `matplotlib`, `seaborn` – Visualização gráfica
* `nltk`, `spacy` – Processamento de linguagem natural
* `scikit-learn` – Modelos de aprendizado de máquina
* `tensorflow`/`keras` – Redes neurais (opcional)

---

## ⚙️ Como Executar

1. Clone o repositório:

```bash
git clone https://github.com/usuario/projeto-x-desastres.git
cd projeto-x-desastres
```

2. (Opcional) Crie um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Execute os notebooks:

```bash
jupyter notebook
```

Abra a pasta `notebooks/` e rode os arquivos `.ipynb`.

---

## 📚 Bibliotecas

* pandas
* numpy
* matplotlib
* seaborn
* nltk
* spacy
* scikit-learn
* tensorflow / keras (opcional)

---

## 🧠 Conclusão Esperada

Espera-se que os modelos consigam:

* Classificar corretamente tweets relacionados a **desastres ambientais**, ajudando na detecção de informações críticas.
* Identificar padrões de linguagem e sentimentos expressos pelos usuários em situações de crise.

Assim, o projeto reforça a importância da ciência de dados como ferramenta para o **monitoramento social em tempo real** e para apoiar a **gestão de emergências ambientais**.

---

👉 Quer que eu já prepare esse texto em **README.md** formatado para você subir diretamente no GitHub?

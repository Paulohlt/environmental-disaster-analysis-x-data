# PROJETO APLICADO II: PROJETO DE ANÁLISE DE TEXTO NA PLATAFORMA X SOBRE DESASTRES AMBIENTAIS

Environmental Disaster Tweet Classification – PA II

Classificação de tweets da plataforma X (antigo Twitter) relacionados a desastres ambientais, utilizando técnicas de Processamento de Linguagem Natural (PLN) e Aprendizado de Máquina.

Projeto desenvolvido como parte da disciplina Projeto Aplicado II – Ciência de Dados, Mackenzie.

## Autores
* **Karla Maria Ramos da Silva** – RA: 10730503  
* **Livya Kaiser de Albuquerque** – RA: 10433409  
* **Paulo Henrique Lasmar Teles** – RA: 10728776  
* **Rafael Hessel Sichetti** – RA: 10375395  

## Objetivo do Projeto

Desenvolver um pipeline de machine learning capaz de classificar automaticamente tweets como:

* **1** — relacionados a desastres ambientais, ou  
* **0** — não relacionados  

utilizando o conjunto de dados Disaster Tweets (Kaggle).

O projeto inclui:

* Pré-processamento de texto  
* Vetorização com TF-IDF  
* Treinamento de múltiplos modelos  
* Avaliação e comparação de métricas  
* Salvamento de artefatos (modelos, vetorizador, métricas)

## Estrutura do Repositório

├── pipeline_disaster_tweets.py        
├── notebook_disaster_pipeline.ipynb   
├── outputs/                           
├── requirements.txt                   
└── README.md                          


## Ambiente Virtual

Este projeto utiliza ambiente virtual para garantir isolamento e evitar conflitos de dependências.

**Criar ambiente virtual**

python -m venv .venv

**Ativar (Windows)**

.venv\Scripts\activate


**Instalar dependências**

pip install -r requirements.txt


##  Como Executar o Projeto

Existem duas formas:

### **1) Executar o Pipeline via Script (.py)**
python pipeline_disaster_tweets.py --disaster_path train.csv --out_dir outputs


Isso irá:

* Carregar e pré-processar o dataset  
* Vetorizar textos com TF-IDF  
* Treinar Logistic Regression, Naive Bayes e SVC  
* Gerar métricas e tabelas  
* Salvar tudo em `outputs/`

### **2) Executar o Notebook**

jupyter notebook

Abra: notebook_disaster_pipeline.ipynb


No notebook você encontra:

* Análise exploratória (EDA)  
* Distribuição das classes  
* Treinamento dos modelos  
* Visualizações (matriz de confusão, distribuição)  
* Discussões e interpretações  

## Resumo dos Resultados

Os três modelos apresentaram desempenho próximo, com acurácia entre **80% e 81%**.

**Melhor modelo (F1-score):**
* **Support Vector Classifier — F1 = 0.7595**

**Outros modelos avaliados:**
* Regressão Logística — F1 = 0.7498  
* Naive Bayes — F1 = 0.7472  

Os resultados completos estão no arquivo: outputs/model_summary.csv


E as matrizes de confusão em:

outputs/cm_logreg.png, 
outputs/cm_nb.png, 
outputs/cm_svc.png


## Requisitos

* Python 3.10+  
* nltk  
* pandas  
* scikit-learn  
* matplotlib  
* numpy  

(Instalados automaticamente via **requirements.txt**)

## Conclusão Resumida

O modelo **Support Vector Classifier (SVC)** apresentou o melhor equilíbrio entre precisão e revocação, tornando-se o mais adequado para a tarefa de identificar tweets realmente relacionados a desastres ambientais.

O pipeline desenvolvido permite replicar facilmente o processo de treinamento, análise e geração de artefatos, facilitando futuras melhorias, como:

* Técnicas de balanceamento  
* Modelos baseados em transformadores  
* Deploy em aplicações reais  







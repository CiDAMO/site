---
title: Análise de sentimentos em reviews de restaurantes
date: 2020-07-27
author: [Dennis Gonçalves Lemes, Victor Gabriel Souza Barbosa, Vinícyus Araújo Brasil]
image: images/blog/review-restaurantes/capa.jpg
bg_image: images/blog/review-restaurantes/capa.jpg
categories: ["Projetos"]
tags: ["Classificação"]
description:
draft: false
type: "post"
---

O presente projeto busca usar técnicas de machine learning para classificar os  reviews de restaurantes em reviews positivos ou negativos, sendo um problema estudado no campo de análise de sentimentos.

A tarefa é clássica no campo do processamento de linguagem natural (PLN) como mostra Kharde (2016). Os modelos de Machine Learning tem tido resultados melhores que os modelos baseados no léxico, o que mostra um novo padrão em PLN. Um exemplo é o Google, que desenvolve e usa modelos em seus produtos (como o BERT no Google Search).

Ilustraremos o post com partes dos códigos, sendo que os scripts completos podem ser encontrados no nosso **Github**.

## Base de dados

Usaremos os reviews do site **Trip Advisor**, visto que é um dos maiores sites do ramo e possui vasto número de avaliações. A coleta dos dados foi feita via web scraping, dado que o site não fornece uma API para tal finalidade.

Nosso Web Scraping tem como base um código disponibilizado por uma usuária do Github e adaptamos para extrair os dados para nosso modelo. O código pode ser obtido [aqui](https://github.com/susanli2016/NLP-with-Python/blob/master/Web%20scraping%20Hilton%20Hawaiian%20Village%20TripAdvisor%20Reviews.py). Ao adaptarmos, nosso objetivo era obter todos os reviews de cada restaurante selecionado previamente e transformá-los em arquivos .csv, a princípio tínhamos como objetivo restaurantes de todo o Brasil, porém devido a quantidade massiva de dados, nosso dataset contém restaurantes apenas da cidade de Curitiba. Os restaurantes escolhidos foram os restaurantes da cidade que continham mais reviews, ou seja, não houve discriminação no que tange a tipos de restaurantes ou tipo de comida, por exemplo. Os links dos restaurantes foram coletados manualmente e colocados em um arquivo txt e geramos o arquivo csv com os reviews com o script gera_csv. O csv tinha como formato 3 colunas, sendo elas:

	ID, Nota de Review (1-5), Texto de Review (Comentário feito pelo usuário)

Os reviews foram separados da seguinte forma: reviews positivos tem nota 4 ou 5 e reviews negativos tem nota 1, 2 ou 3. Então, no csv, temos uma variável binária: o review positivo tem valor igual à 1 e o review negativo tem valor igual à 0. Tal forma de classificar os dados foi feita por dois motivos principais. O primeiro é a subjetividade de cada pessoa no que tange às notas, ou seja, pessoas diferentes tem ideias diferentes do que é um restaurante nota 3, por exemplo. Dividir o banco de dados em reviews positivos ou negativos elimina esta subjetividade.

Outro motivo é que o Trip Advisor deleta restaurantes com muitas reviews negativas do seu site, o que resulta em muito mais reviews positivas do que negativas presentes. Devido a isso, foram coletadas por volta de 144 mil reviews no total, dentro delas 16 mil negativas.Para que o dataset fique equilibrado, selecionamos aleatoriamente 16 mil reviews positivos, criando uma base de dados com 32 mil reviews.

Com isso, possuíamos os dados para serem trabalhados. O código para esta parte é o arquivo gera_csv_final no Github.

## Modelos

Os modelos que usaremos se baseiam em Kharde(2016). Usaremos modelos que são clássicos de tarefas de aprendizado supervisionado, sendo eles:
- Regressão Logística;
- SVM (na sua "versão" polinomial);
- Random Forest;

O outro modelo que usaremos é o BERT, apresentado em Devlin et al (2018), que funciona de uma forma um pouco diferente dos outros modelos acima apresentados. O modelo é uma rede neural pré-treinada com *Self-Supervised Learning* em textos (como o Wikipedia, por exemplo) que utiliza técnicas como o *Attention* e *Fine-Tuning*, entre outras. Os autores mostram que tal tipo de arquitetura atingiu resultados sublimes em várias tarefas clássicas de PLN. Visto isso, adicionamos o BERT ao projeto visando compará-lo com os modelos clássicos. A implementação foi baseada no código de [Kahnhtran](https://github.com/chriskhanhtran/bert-for-sentiment-analysis).

## Vetorização


Para que os modelos que usaremos possam entender a linguagem humana, devemos transformar as palavras em vetores numéricos. Há três principais formas de se fazer isso: a vetorização TFIDF, Word2Vec e o Dynamic Word Embedding.

A primeira forma usa o método TF-IDF (*Term Frequency - Inverse Document Frequency*) para transformar as palavras em vetores. Usaremos este método de vetorização nos modelos de Regressão Logística, no Random Forest e no SVM. Usaremos, para tal, a função tfidfVectorizer() do **scikit-learn**.

	t_vector = TfidfVectorizer()
	t_vector.fit(data_clean['review_body'])
	train_X = t_vector.transform(X_train['review_body'])
	test_X = t_vector.transform(X_test['review_body'])

Já o Word2Vec é uma forma de vetorização de palavras de forma que, palavras que tem significados próximos sejam representadas por vetores que tem o ângulo perto de zero. Tal ângulo é medido pela similaridade do cosseno como representado na figura abaixo.

{{< figure src="/images/blog/review-restaurantes/panda.png" title="Figura 1: ilustração do método da similaridade do cosseno, usado pelo Word2Vec (Karani, 2018)" >}}

O outro método de vetorização, o Dynamic Word Embedding, é similar ao Word2vec mas lida com o problema de palavras com múltiplos significados, criando um vetor novo (com *features* adequadas) para cada instância da palavra. Há um rede neural treinada em milhões de textos, onde a refinaremos para o nosso problema, treinando ela novamente a fim de ajustar o modelo pré-treinado. Este método é a forma que o modelo BERT que usaremos trata as palavras.

## Aplicação dos modelos

Como nosso objetivo a prior se trata de mostrarmos como o **BERT** pode ser extremamente eficiente no nosso projeto, usamos alguns outros modelos para que sua eficiência seja medida.
Todo os modelos usados estão presentes na biblioteca do **scikit-learn**, os modelos utilizados são:
- Regressão Logística;
- Random Forest;
- Support Vector Machine (SVM);
- BERT;

Um trecho de código aplicando a Regressão Logística como exemplo:

    clf = LogisticRegression()
 	clf.fit(x_train, y_train)
 	log_pred = clf.predict(x_test)

 A implementação do BERT foi baseada no código do [Kahnhtran](https://github.com/chriskhanhtran/bert-for-sentiment-analysis). O BERT, por sua vez tem uma diferença em sua aplicação, como podemos mostrar em um exemplo:

	exemplos_inputs, exemplos_masks  = preprocessing_for_bert(review_exemplo)
	exemplo_data = TensorDataset(exemplos_inputs, exemplo_masks)
	exemplo_dataloader = DataLoader(exemplo_data, batach_size=batch_size)

*Essa parte pode ser definida como uma "preparação dos dados", que antecipa a predição do **BERT***.

	probs = bert_predict(bert_classifier, exemplo_dataloader)

*Esse trecho efetivamente realiza a predição dos dados.*

 Para calcularmos, por exemplo, a acurácia de nosso modelo usamos algo como:

	accuracy_score(log_pred, y_test)

## Resultados

Enfim, utilizamos a métrica da acurácia e algumas outras para testarmos o potencial do nosso modelo, entre elas estão **acurácia, precisão, recall,  escore f1 e a AUC (área sob a curva)**, feito isso criamos uma tabela para comparação intuitiva de "desempenho".

|          | Reg. Logística | Random Forest | SVM (Polinomial)  |   BERT  |
|----------|:--------------:|:-------------:|:-----------------:|:-------:|
| Acurácia |     0.87095    |    0.81844    |      0.88571      | 0.89469 |
| Precisão |     0.87435    |    0.82322    |      0.88573      | 0.89910 |
|  Recall  |     0.86800    |    0.81833    |      0.88573      | 0.89148 |
| F1 Score |     0.87116    |    0.81833    |      0.88571      | 0.89532 |
|    AUC   |     0.87096    |    0.81847    |      0.88572      | 0.95802 |


## Conclusão

Os dois melhores modelos, o BERT e o SVM, conseguem acertar, na média, 89% do total de reviews. É notório que o BERT teve um resultado um pouco melhor no indicador AUC, como mostra a tabela acima, confirmando o que a literatura havia apontado.


## Referências

- Allamar, J. jalammar.github.io/illustrated-bert/. Acesso em 20/04/2020.
- Carvalho, MH de. *Estudo Comparativo dos Métodos de Word Embedding na Análise de Sentimentos.* Páginas 20-24. 2018
- Devlin J., Chang M., Lee K., Toutanova K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* 2018.
- Howard, J. *Fine-tuned Language Models for Text Classification*. 2018
- Karani, S. *Introduction to Word Embedding and Word2Vec*. Disponível em https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa . Acesso em 13/05/2020. 2018.
- Kharde, V. *Sentiment Analysis of Twitter Data: A Survey of Techniques*. 2016
- Nayak, P. *www.blog.google/products/search/search-language-understanding-bert/*. 2019. Acesso em 05/05/2020.
- Rajaraman, A.; Ullman, J.D. Mining of Massive Datasets. 2011.

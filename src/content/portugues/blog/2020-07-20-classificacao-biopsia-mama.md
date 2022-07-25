---
title: Classificação da qualidade de biópsia de mama usando redes neurais
date: 2020-07-20
author: [Augusto Pereira, João Victor da Silva, Leonardo Dionizio, Milena Andreuzo Cardoso]
image: images/blog/classificacao-biopsia-mama/capa.jpg
bg_image: images/blog/classificacao-biopsia-mama/capa.jpg
categories: ["Projetos"]
tags: ["Classificação"]
description:
draft: false
type: "post"
---

Automatizar trabalhos utilizando a tecnologia é do interesse de múltiplas empresas, laboratórios e afins do Brasil e do mundo, inclusive do Laboratório de Patologia Experimental da PUC-PR. Esse interesse se tornou uma parceria com uma equipe do processo seletivo do CiDAMO - nossa equipe - e deu tão certo que se tornou um projeto dentro do grupo!

**Dentre as técnicas atuais em Ciência de Dados, o [aprendizado de máquina](https://pt.wikipedia.org/wiki/Aprendizado_de_m%C3%A1quina) (*Machine Learning*, em inglês, sigla ML) é uma das principais, mais utilizadas e requisitadas no mercado de profissionais de tecnologia**. Armados apenas com o básico dela, o grupo que continha:

* um geógrafo;
* um veterano do CiDAMO;
* um programador iniciante; e
* uma estudante de bioinformática

se encarregou de criar um projeto relacionado a reconhecer imagens médicas e tinha tudo para entrar em pânico quando, enquanto muitos outros grupos faziam a própria pesquisa independente, o geógrafo sugeriu um projeto em parceria, no qual os resultados tinham que ser efetivos. **Felizmente, ao invés de desistir e caçar um problema menos intimidador e com um [dataset](https://pt.wikipedia.org/wiki/Conjunto_de_dados) mais robusto, a equipe aceitou o desafio.**

Conhecimento básico de _Python_ e da biblioteca [scikit-learn](https://scikit-learn.org/stable/) e nenhuma especialização na área. O que foi esse projeto? Uma brincadeira? Uma frustração? E se te dissessem que o resultado foi melhor do que esperávamos, você acreditaria? Talvez isso já seja óbvio ao leitor mais atento, que reteu o título do post, mas a ideia aqui não é mostrar só onde chegamos, e sim **mostrar a jornada até chegarmos no resultado** e também como ela irá continuar.

Ficou interessado? Confira nosso problema e as técnicas que utilizamos para resolvê-lo!

## O Problema

O projeto em si consiste na análise de imagens médicas. **Em parceria com o Laboratório de Patologia da PUCPR, adquirimos imagens de micrografias do tecido mamário**, obtidas a partir de lâminas contendo biópsias de pacientes, visualizadas em microscópio óptico.

As micrografias fazem parte do exame clínico para verificação do diagnóstico, do tratamento e prognóstico de câncer de mama, ou então - com viés acadêmico - para estudos da eficácia de tratamentos. Nem todas as imagens obtidas com as lâminas são adequadas para análise e a figura logo abaixo mostra alguns exemplos disso. Imagens adquadas são como a 1. na figura abaixo, com todos os seus quadrantes preenchidos pelo tecido mamário. Imagens inadequadas podem conter:

* quantidade de tecido insuficiente (muitos espaços em branco) (2.A na figura)
* dobras no tecido (2.B na figura);
* artefatos (objetos que não são o tecido, como fios de cabelo ou instrumentos) (2.C na figura); ou
* presença de vasos sanguíneos com hemácias (2.D na figura).


{{< figure src="https://i.imgur.com/iogFocr.png" title="Alguns exemplos de imagens utilizadas" >}}

**Portanto, faz parte da rotina do laboratório que um profissional utilize de seu tempo para avaliar e separar as imagens em adequadas ou inadequadas, processo que leva algumas horas e requer um profissional bem treinado.** Somente depois dessa classificação, o tecido contido nas imagens será avaliado quanto a presença ou não de tecido canceroso, além de suas dimensões e padrão.

Uma grande quantidade de tempo seria poupada com a tradução do processo de classificação e separação das imagens em adequadas e inadequadas em uma ferramenta automática. Eis o nosso problema: criar um sistema que aprenda a separar as imagens do tecido mamário em duas categorias distintas, utilizando é claro nosso querido *Machine Learning*.

Agora que você já está por dentro da situação inicial, pode conhecer nossa primeira aproximação.

## O método inicial

Três métodos podem ser utilizados em *Machine Learning*. No aprendizado supervisionado, o algoritmo é alimentado com dados que incluem já as saídas desejadas. Também é possível o aprendizado não supervisionado, quando o algoritmo é alimentado com dados sem as saídas desejadas, onde a máquina mesmo deve prever as saídas. Ainda, o algoritmo pode ser treinado por meio de recompensas ou punições, no aprendizado por reforço.

Em quaisquer desses métodos, o objetivo é treinar o algoritmo a reconhecer padrões e conseguir fazer previsões ao analisar um novo conjunto de dados.

Uma vez que temos 786 imagens previamente classificadas entre adequadas e inadequadas, **nossa tarefa se define como um problema de aprendizado supervisionado.** Antes de seguirmos, é pertinente deixar o leitor avisado que utilizamos _Python_ durante não só a abordagem pioneira, mas durante todo o projeto - algo que é bem comum aqui no CiDAMO, salvo os projetos envolvendo [Julia](https://pt.wikipedia.org/wiki/Julia_(linguagem_de_programa%C3%A7%C3%A3o)).

### OpenCV: lidando com imagens

```
import cv2 as cv
```

Quando se juntam os temas _Python_ e visão computacional, vem à mente de um conhecedor do assunto duas bibliotecas distintas: [_PIL_](https://pt.wikipedia.org/wiki/Python_Imaging_Library) e [_OpenCV_](https://opencv.org/). **A escolhida para o projeto foi a *OpenCV*.** Por que? Não foi pela compatibilidade, comodidade ou conhecimento prévio, mas pela facilidade de gerar histogramas de cor com ela. Quem sabe a _PIL_ seja para nós um tesouro inexplorado, cheia de funcionalidades que nos ajudassem: de fato é uma possibilidade, mas quando se é leigo na questão e se tem um prazo de quatro semanas não é possível gastar muito tempo na escolha de ferramentas.

Outra pergunta importante: por que gerar histogramas foi um fator tão importante na decisão? Por algo simples que foi percebido logo na primeira reunião do grupo: **lembra das imagens com muitos espaços em branco? Grande parte das imagens são inadequadas devido a tal característica.** Esses gráficos nos ajudam, portanto, a identificar um dado tão importante que foi adicionado ao nosso dataset: quantidade de branco nas imagens.

Futuramente, usaríamos a *OpenCV* em _scripts_ que redimensionavam as amostras, mas vamos chegar lá.

### Gradient Boosting Classifier

```
from sklearn.ensemble import GradientBoostingClassifier
```

Um dos melhores na categoria de modelos baseados em árvores de decisão, **o _Gradient Boosting_ aprende de forma gradual, aditiva e sequencial**, ou seja, adiciona modelos simples ao principal visando minimizar a função perda através do gradiente, de onde deriva o nome do método. Esse processo de minimização é feito em cada iteração onde esses modelos simples - geralmente árvores de decisão bem pequenas - olham para um subconjunto do conjunto original.


{{< figure src="https://i.imgur.com/pLbTA44.gif" title="Exemplo de dados sendo classificados por uma Árvore de Decisão. Fonte: R2D3 [1]" >}}

Utilizando essa técnica simples, por muitos já conhecida, e apenas dois dados por amostra: a quantidade de branco e os bytes do arquivo (ambos normalizados), **a acurácia inicial já beirava incríveis 90%**.


## Deep Learning

Um braço importante do ML que foi incluso neste projeto é o [*Deep Learning*](https://pt.wikipedia.org/wiki/Aprendizagem_profunda) ou aprendizagem profunda.

**Para quem acompanha programação e tecnologia, *Deep Learning* está cada vez mais presente**, citado às vezes como a mina de ouro que garantirá emprego para muitos profissionais de TI. Se ouve muito, mas nem todos sabem o que essa palavra significa: o *Deep Learning* vira um conceito fantasma, algo que todos querem trabalhar e saber, mas poucos compreendem. O que é, afinal, *Deep Learning*?

**O *Deep Learning* é, resumidamente, um processo que utiliza redes neurais artificiais baseadas nas sinapses do cérebro humano** e inclui múltiplas camadas de processamento para obtenção de melhores resultados no treinamento, ajustando essas camadas a cada iteração.


{{< figure src="https://miro.medium.com/max/1500/1*bhFifratH9DjKqMBTeQG5A.gif" title="Exemplo de imagens sendo classificadas por uma Rede Neural. Fonte: Towards Data Science [2]" >}}

### Por que Deep Learning?

Antes mesmo de escolhermos qual seria o projeto relacionado a imagens médicas que escolheriamos, já havia em alguns membros do grupo a vontade de trabalhar com *Deep Learning*. Por que? Como dito anteriormente, todos querem conhecer e trabalhar com ele. Além disso, quando se junta *Machine Learning* e imagens, muitos associam imediatamente à *Deep Learning* (provavelmente pelas convoluções, mas vamos chegar lá).

Foi sorte que o até então abstrato *Deep Learning* se encaixou perfeitamente como método para resolver nosso problema. Se é possível pensar uma rede neural como uma simulação do cérebro humano, como uma criança, a rede vê algo várias vezes até compreendê-lo. Isso significa que ela vai **aprender sozinha a identificar dobras, vasos e artefatos** facilitando - e muito - nosso trabalho.


## Ambientes de trabalho

Um local como um escritório, em que os membros estão no mesmo ambiente, muitas vezes já é suficientemente caótico para que um integrante de uma equipe não tenha ideia do que seus companheiros de time estejam fazendo. **Esse problema se intensifica exponencialmente num mundo em que preferencialmente 100% do trabalho deve ser remoto.** Felizmente o trabalho fragmentado é extremamente corriqueiro na vida do profissional da área de tecnologia - algumas empresas ousadas do ramo nem possuem escritório - e **existem múltiplas ferramentas que facilitam o compartilhamento de código.**

Compartilhar código tem a ver com acesso aos mesmos arquivos e compartilhar edições e alterações nos _scripts_ ou no dataset. É necessário um ambiente de desenvolvimento que tenha suporte em caso de conflito de edições, atualize os arquivos nas máquinas de cada um e controle o versionamento das bibliotecas utilizadas para coincidir em todos os computadores.

Sondamos várias ferramentas de compartilhamento de código, as principais você confere a seguir.

### GitHub e Conda-Enviroments: as ferramentas que **NÃO** utilizamos

```
git pull
```

Não precisa ser um _expert_ em programação para saber que o [GitHub](https://github.com/) é a ferramenta mais comum para gestão de projetos relacionados a programação. Com uma dinâmica incluindo tanto repositórios locais quantos remotos, **a ferramenta é toda pensada para suportar o seu fluxo de trabalho.** Suportado em diversos sistemas operacionais e tendo tanto interface gráfica quanto suporte para terminal, com o **GitHub é possível gerir projetos, administrar conflitos, trabalhar em ramos separados, entre muitas outras funcionalidades.**

Mas, sozinho, ele não era o suficiente. Então foi nos apresentados os **Conda-Enviroments: uma forma de controlar o versionamento das bibliotecas a partir de arquivos _yml_.** Semelhante ao muito utilizado [_virtualenv_](https://virtualenv.pypa.io/en/latest/), os conda-enviroments fazem parte do [_Anaconda Distribution_](https://www.anaconda.com/), um ambiente de trabalho destinado a ciência de dados que já havia sido recomendado para nós no inicio do processo seletivo do CiDAMO.

O interessante do software é que a forma que ele controla o versionamento se assemelha a uma máquina virtual, não alterando o comportamento do _Python_ configurado mediante suas preferências. Agora que já temos um ambiente quase perfeito para programar, podemos começar, certo? **Não, o GitHub e o Anaconda foram engavetados.**

Primeiro, nosso arquivo _yml_ deu problema para alguns membros. Segundo, conseguir **processar efetivamente _Deep Learning_ em uma máquina requer um acelerador gráfico (GPU) com placas CuDA** (foi testado o treino sem o uso de GPU, e ele levaria 11 vezes mais tempo). Apenas a NVIDIA cria placas de vídeo pensadas também para _Deep Learning_ e nem todos têm acesso às placas da empresa. Além disso, o NVIDIA Toolkit não é um bom exemplo de didática e o processo para configurar o uso de CuDA é um tanto quanto exaustivo para leigos.

Não tínhamos a GPU necessária para o nível de processamento que desejávamos. Mas sabe quem tinha? A Google.

### Apresentando Google Colab, nosso melhor amigo

```
from google.colab import drive
drive.mount('/content/drive/')
```

Foi uma surpresa quando foi introduzido para nós o [_Google Colaboratory_](colab.research.google.com), uma ferramenta de escrita e compilação de código escrito em _Python_ que possui compatibilidade com o Google Drive e, mais importante, **a possibilidade de usar GPUs remotas gratuitamente.**

Qualquer nova imagem, pasta, alteração no dataset, modelo salvo, além de ficar gravado em uma pasta no drive, é atualizado instantaneamente - sem medo do famoso aviso de conflito quando se dá um `git pull`. O Google Colab nos dá uma quantidade considerável de armazenamento em RAM, permitindo uma ótima capacidade de processamento, e importou perfeitamente todas as bibliotecas que utilizamos.

Já que é um ambiente em navegador, utilizar o Google Colab não requer que se baixe - ou pague - nada, não tem problemas com nenhum sistema operacional e não ocupa a memória do seu computador: dá pra assistir sua série na Netflix tranquilamente enquanto espera o treinamento da sua rede acabar. E, é claro, o Colab tem um tema escuro muito agradável aos olhos.

#### Um guia rápido de como estourar a mémoria do Google

Já demos um pequeno spoiler sobre essa seção quando dissemos que utilizamos a _OpenCV_ para redimensionar nossas amostras. O motivo é simples, mas exige uma certa matemática para explicá-lo.

A API que utilizamos no projeto carrega as imagens em matrizes tridimensionais no formato Altura x Largura x RGB. Sendo que temos amostras de 1600x1200 pixels e 786 imagens, a entrada da rede neural se configura numa matriz quadridimensional 786 x 1600 x 1200 x 3. Cada número contido nessa matriz, sendo normalizado, ocupa o espaço de um _float32_, ou seja, 4 bytes de memória.

Se você fizer as contas agora, vai ver que **a quantidade de RAM necessária para armazenar esses dados é irreal.**

Por essa e outras razões, realizamos uma série de rotinas antes de treinar nossa rede.

## Pré-Processamento

Quando se está trabalhando com números, é cômodo que eles estejam dentro de uma escala. Nas redes neurais não é diferente,  **os valores ideais são aqueles entre 0 e 1**. Por isso, sabendo que o padrão RGB está dentro de um intervalo natural entre 0 e 255, **dividimos todos os valores por 255**. A necessidade dessa normalização foi testada e aprovada pelo grupo.

Além disso, **também precisávamos reduzir a dimensão das imagens** - memória não é infinita - de forma que não houvessem distorções que comprometessem o desempenho da rede. **Optamos por trabalhar com imagens 400x300 pixels, reduzindo bastante a memória necessária para armazenamento**. Esse processo de redução das dimensões, como dito anteriormente, foi feito utilizando a biblioteca _OpenCV_.

Agora que nosso leitor já está por dentro de todas as circunstâncias do projeto, podemos introduzir o coração do nosso projeto: a rede neural.

## Keras e as redes neurais convolucionais

Vimos que **as redes neurais são esse robusto encadeamento de funções em uma arquitetura que relaciona camadas de neurônios, que recebem valores de entrada (inputs) e processam resultados de saída (outputs)**. Cada camada de neurônios apresenta funções que aplicam pesos e vieses aos valores a elas passados (o input inicial da rede, ou o resultado da camada anterior), tomando esse resultado intermediário como valor a ser processado por uma função de ativação, ou seja, por uma operação que dá a intensidade do sinal de saída do neurônio - viu como a rede é realmente inspirada nas sinapses?

Quando uma camada emite seus sinais (os outputs de cada neurônio), de maneira que eles sirvam para entrada em todos os neurônios da próxima camada, é dito que essas são camadas densas. Vejamos adiante um exemplo de camadas densas:

{{< figure src="https://i.imgur.com/uSOdDss.jpg" title="Exemplo de uma Rede Neural. Fonte: Carnegie Mellon University [3]" >}}


Pois bem, os estudos sobre _computer vision_ avançaram sobre essas redes neurais, identificando que algumas camadas convolucionais ajudam a aumentar a eficiência de modelos de classificação. Mas o que são as camadas de convolução?

**Convolução, em termos bastante simplificados, são operações que seguem uma lógica de varredura sobre uma matriz**. Trata-se de uma estratégia que vem sendo aplicada em _Deep Learning_, mas que já faz parte de algoritmos de tratamentos de imagens, como aqueles que a gente usa no instagram para tirar a cara pálida de tantas noites trabalhando em códigos!

Uma camada convolucional toma uma porção do input (a imagem como lançada na rede, ou o output da camada anterior) aplica uma função matricial em que pesos são multiplicados aos pixels avizinhados e sua soma é aplicada ao pixel central. Essa operação é repetida por toda a extensão da imagem. Usualmente, isso é feito de maneira que a saída da operação seja uma imagem mais simplificada, com menor número de pixels e com realce de algumas características: padrões de textura, limites horizontais dos elementos, limites verticais, aumento de contraste etc.

No fluxo de dados em uma rede neural, o papel das redes convolucionais é exemplificado adiante:

{{< figure src="https://i.imgur.com/h9OlTs2.jpg" title="Exemplo de uma Rede Neural Convolucional. Fonte: MC.AI [4]" >}}

A imagem abaixo mostra justamente esse processo de varredura, que caracteriza algoritmos convolucionais:

{{< figure src="https://i.imgur.com/G9HSfdE.gif" title="Exemplo de um processo de varredura. Fonte: Medium [5]" >}}

Se a varredura é feita com janelas (_kernels_) de 3x3 com os pesos postos na imagem abaixo, teremos, respectivamente, a simples manutenção da imagem original, o realce das texturas e a evidenciação dos limites verticais à esquerda.

{{< figure src="https://i.imgur.com/deV20Da.png" title="Exemplo de kernels aplicados. Fonte: Setosa [6]" >}}

Este [artigo](https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8) e este [exemplo interativo](https://poloclub.github.io/cnn-explainer/#article-relu) ajudam a compreender o papel das convoluções nas redes neurais.

O processo todo parece bastante complexo, mas o amadurecimento do ecossistema de bibliotecas de Ciências de Dados em linguagem _Python_ fornece um conjunto de ferramentas bastante arrojadas para a aplicação dessas soluções: é aí que entra o [_Keras_](https://keras.io/).

**_Keras_ é uma API, uma Interface de Programação de Aplicação, que faz uma mediação entre o cientista de dados e o _TensorFlow_, biblioteca dedicada a operações computacionais complexas, como geração de redes neurais**. Trata-se de uma API de alto nível, ou seja, uma ferramenta em que são abstraídos diversos detalhes do processo de geração de redes neurais, de forma que a operação se torna muito mais fácil, ágil e intuitiva. Por essa razão, essa foi a ferramenta utilizada para a geração do nosso modelo de classificação de imagens.


### Stacking

Esse método faz parte dos métodos baseados em [_Ensemble_](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205) e seu objetivo é conseguir combinar os modelos já treinados para obter um resultado mais robusto e satisfatório.

De forma bem simples para utilizar o _stacking_ separamos o conjunto em dois conjuntos distintos, E e D. Considere o conjunto E como o treinamento e D como conjunto de validação. Dentro do conjunto E treine seus aprendizes fracos e faça as predições em cima do conjunto D, pegue as predições do conjunto D feitas pelos aprendizes fracos e treine um outro modelo, sendo as entradas desse modelo apenas as predições dos aprendizes fracos. Esses dois conjuntos iniciais E e D podem ser feitos com [**K-fold**](https://pt.wikipedia.org/wiki/Valida%C3%A7%C3%A3o_cruzada#M%C3%A9todo_k-fold), por exemplo o conjunto E pode ser os primeiros K-1 folds e o conjunto D pode ser o K-ésimo fold.

O que fizemos não foi exatemente um _stacking_ mas possui a mesma ideia: **combinamos os dois modelos citados anteriormente, _Gradient Boosting Classifier_ e Redes Neurais Convolucionais**, onde o _Gradient Boosting_ olha apenas para a quantidade de branco e os bytes de cada foto e a rede neural olha diretamente para as imagens redimensionadas, ou seja, possui um passo extra nesse caminho para o _stacking_.

Nós utilizamos **[Regressão Logística](https://pt.wikipedia.org/wiki/Regress%C3%A3o_log%C3%ADstica)** como nossa segunda camada de método, conseguindo **incríveis 92% de acurácia**.

## Próximos passos

**Obtivemos bons resultados, mas podemos ir além.** A quantidade de 786 imagens que utilizamos para o treinamento pode parecer grande, mas não é, quando comparamos com datasets de outros trabalhos em processamento de imagens, que chegam nas milhares de fotos. Além disso, as amostras eram apenas de cinco pacientes diferentes. Para melhorar o modelo e ampliar sua resposta, o plano é adquirir novos dados, complementando o que já temos.

**Também planejamos testar outras arquiteturas de redes convolucionais.** A que utilizamos por ora é das mais simples, um modelo sequencial, mas existem outras mais complexas e que apresentam outras possibilidades de arranjo de camadas.

Além disso, **temos em perspectiva a criação de um aplicativo para aplicação do algoritmo em laboratório, uma implementação mais _user-friendly_ do nosso trabalho, sem que os pesquisadores e/ou médicos entrem em contato com o código em si**, o que facilitaria ainda mais o trabalho deles.

Entretanto, o que facilitaria mesmo o trabalho do laboratório e é o sonho (alto, pois por que não sonhar alto?) com esse projeto, seria um resultado final automatizado. **Lembramos que após a seleção das imagens adequadas, avalia-se a presença e quantidade de tecido canceroso. Já imaginou esse processo feito por uma máquina? Liberando o tempo do profissional que analisa as imagens, tornando o processo mais rápido e padronizado... nós imaginamos, e planejamos chegar lá!**

## Referências

[1] ADAPTADO de R2D3: Visual Introduction to Machine Learning. <b>R2D3</b>. Disponível em: <http://www.r2d3.us/visual-intro-to-machine-learning-part-1/>. Acesso em 21 de julho de 2020.

[2] EVERYTHING you need to know about Neural Networks and Backpropagation — Machine Learning Easy and Fun. <b>Towards Data Science</b>. Disponível em: <https://towardsdatascience.com/everything-you-need-to-know-about-neural-networks-and-backpropagation-machine-learning-made-easy-e5285bc2be3a>. Acesso em 21 de julho de 2020.

[3] CONVOLUTIONAL Neural Networks. <b>Carnegie Mellon University</b>. Disponível em: <http://course.ece.cmu.edu/~ece739/lectures/18739-2020-spring-lecture-07-cnn.pdf>. Acesso em 21 de julho de 2020.

[4] CONVOLUTION Neural Network?. <b>MC.AI</b>. Disponível em: <https://mc.ai/convolution-neural-network/>. Acesso em 21 de julho de 2020.

[5] TRANSPOSED Convolutions explained with… MS Excel!. <b>Medium</b>. Disponível em: <https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8>. Acesso em 21 de julho de 2020.

[6] GERADO em Image Kernels. <b>Setosa</b>. Disponível em: <https://setosa.io/ev/image-kernels/>. Acesso em 21 de julho de 2020.

[7] GULLI, A.; PAL, S. **Deep Learning with Keras**. 1a edição. Inglaterra: Packt Publishing, 2017.

[8] CHOLLET, F. **Deep Learning With Python**. 1a edição. Nova Iorque: Manning Publications, 2017.
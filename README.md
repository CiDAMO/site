# Site do CiDAMO (versão nova em progresso)

Este é o repositório do site do grupo CiDAMO (Ciência de Dados, Aprendizagem de Máquina e Otimização) da UFPR.
Este repositório é aberto para manutenção pelos membros do CiDAMO.
Caso você não seja do CiDAMO e queira contribuir alguma coisa, por favor **abra um issue primeiro**.

Disclaimer: Este repositório contém uma versão modificada do tema <https://github.com/themefisher/airspace-hugo> na pasta `themes/airspace-hugo/`.

A licença para o código do CiDAMO é MPL-2.0, listada em [LICENSE](LICENSE), mas as imagens aqui contidas não são de domínio público e seu uso é restrito.

O resto deste documento é direcionado para desenvolvedores do site.

---

Este site usa o [Hugo](https://gohugo.io) como framework.
Infelizmente o código foi feito a partir de um tema pré-existente, então é muito mais complicado do que gostaríamos, e ninguém sabe 100% do que está acontecendo.
Dito isso, segue uma descrição breve de cada diretório, em ordem de importância:

- `content/portugues`: Diretório principal do conteúdo do site. Cada arquivo na raiz deste diretório é uma página, que pode ou não estar sendo utilizada. O formato varia um pouco por causa do template. Cada pasta aqui que tenha um arquivo `_index.md`, gera uma página nova.
  - `content/portugues/author/`: Pasta com os autores do blog.
  - `content/portugues/blog/`: Pasta com os blogs.
  - `content/portugues/equipe/`: Pasta com a equipe. Provavelmente um merge com a pasta `author` seria útil.
  - `content/portugues/eventos/`: Pasta com os eventos.
- `assets`: Contém os arquivos Javascript e [SCSS](https://sass-lang.com). Os arquivos SCSS são compilados automaticamente.
- `layouts`: Página com layouts HTML das páginas.
- `static/images/`: Imagens.
- `themes/airspace-hugo/`: Tema do airspace-hugo que foi modificado para nosso site. Infelizmente não é fácil descobrir o que foi mudado, e portanto não dá pra atualizar facilmente.
- `.github/workflows/`: Workflows, como o de deploy.

Sobre os arquivos soltos na raiz, o `config.toml` também é relevante para o site.
O `netlify.toml` deve morrer quando acabarmos o Milestone MVP.
O `.hugo_build.lock` é coisa do Hugo, e não sei pra que serve.

## Instalação

Instale o Hugo (veja <https://gohugo.io/getting-started/installing/>).
A versão do Hugo usada atualmente é `0.101.0`.

> Idealmente, deveríamos usar o Docker, para manter todas as versões compatíveis.
> Como não tem tanta coisa que pode quebrar, no entanto, talvez não faça muita diferença - vamos descobrir em breve.
> Sendo assim, colocarei os comandos abaixo como se estivesse no Linux usando o `hugo` direto.
> Caso você pretenda usar o Docker, basta mudar `hugo COMANDO` por
>
> ```shell
> docker run -v="$PWD:/src" -p 1313:1313 klakegg/hugo:0.101.0 COMANDO
> ```
>
> - `-v="$PWD:/src"`: Quer dizer manda a pasta atual ($PWD) para a pasta `/src` do Docker. Senão ele não tem os arquivos para compilar
> - `-p 1313:1313`: Quer dizer transmita a porta `1313` do Docker pra minha porta local `1313`. É necessário para o comando `server` para ver o site servido.
> - `klakegg/hugo:0.101.0`: A imagem.

## Desenvolvimento

Rode

```shell
hugo server
```

para ver o site em <http://localhost:1313>.
Atualizações nos arquivos serão refletidas automaticamente.

## Build

Para ver uma versão compilada do site, rode

```shell
hugo build
```

Esse comando irá gerar a pasta `public`, que não deve ser adicionada ao git.
Essa pasta contém o site gerado, que não depende mais do Hugo.
Essa é a versão que é colocado online.

## Deploy

> Atualmente estamos fazendo o deploy automático para o netlify em <https://cidamo.netlify.app>.
> Em breve vamos apagar essa versão, então não vamos explicar como funciona.

O deploy do site é feito com workflows do GitHub.
O arquivo [.github/workflows/deploy.yml](.github/workflows/deploy.yml) informa ao GitHub como:

1. Fazer o download desta pasta.
2. Instalar o Hugo na versão certa.
3. Rodar o build do Hugo.
4. Subir a pasta `public` para o branch `gh-pages`.

A branch `gh-pages` contém a versão compilada do site, e a partir desta branch, o GitHub gera o site.

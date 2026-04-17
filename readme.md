# ANN do zero em Python

Implementacao didatica de uma Rede Neural Artificial (ANN) sem frameworks de machine learning.

Objetivo: aprender os fundamentos de `Neuron`, `Layer`, `Network`, feedforward e backpropagation com codigo simples.

Baseado nos conceitos do livro *Classic Computer Science Problems*, de David Kopec.

![Rede neural biológica e artificial](./img/neurons.jpeg)

## Leitura rapida

- Guia enxuto de uso e estado atual: [`ann/read.md`](./ann/read.md)
- Este arquivo (`readme.md`): explicacao completa + tutorial passo a passo

## O que mudou no projeto (modernizacoes)

As melhorias abaixo ja estao implementadas:

- Suporte a `bias` por neuronio (agora entra no calculo da saida e no treino)
- `sigmoid` numericamente estavel para evitar overflow com valores extremos
- Caminhos de dataset robustos nos exemplos (via `Path(__file__)`)
- Reprodutibilidade inicial com `seed(42)` nos exemplos
- Suite de testes automatizados com `unittest`
- Estrutura de pacotes explicita com arquivos `__init__.py`
- Dependencias alinhadas: projeto usa apenas stdlib do Python

## Estrutura do projeto

```text
ANN/
├── ann/
│   ├── Core/
│   │   ├── util.py
│   │   ├── neuron.py
│   │   ├── layer.py
│   │   └── network.py
│   ├── data/
│   │   ├── iris.csv
│   │   └── wine.csv
│   ├── examples/
│   │   ├── iris_test.py
│   │   └── wine_test.py
│   └── read.md
├── tests/
│   ├── test_util.py
│   └── test_network.py
└── readme.md
```

## Como a rede funciona (simples)

1. Entrada: voce envia os atributos (ex.: medidas de uma flor).
2. Feedforward: os dados passam camada por camada.
3. Erro: a saida prevista e comparada com a esperada.
4. Backpropagation: o erro volta pelas camadas.
5. Atualizacao: pesos e `bias` sao ajustados.
6. Repeticao: apos varias iteracoes, a rede melhora.

## Componentes principais

### `ann/Core/util.py`

- `dot_product`: produto escalar
- `sigmoid`: ativacao (com estabilidade numerica)
- `derivative_sigmoid`: derivada da sigmoide
- `normalize_by_feature_scaling`: normaliza features para `[0, 1]`

### `ann/Core/neuron.py`

- Define o neuronio
- Guarda `weights`, `bias`, `output_cache`, `delta`
- Calcula saida com:

`z = dot_product(inputs, weights) + bias`

### `ann/Core/layer.py`

- Cria e organiza os neuronios da camada
- Calcula saida da camada
- Calcula deltas para camada de saida e oculta

### `ann/Core/network.py`

- Monta a arquitetura da rede
- Executa `outputs` (feedforward)
- Executa `backpropagate`
- Atualiza `weights` e `bias` no treino
- Valida previsoes com `validate`

## Tutorial de uso (passo a passo)

### 1) Pre-requisitos

- Python 3.11 ou superior
- Nenhuma dependencia externa obrigatoria

Opcional: criar ambiente virtual.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Rodar exemplos

A partir da raiz do projeto (pasta que contem `ann/`):

```bash
python -m ann.examples.iris_test
python -m ann.examples.wine_test
```

Cada comando imprime o total de acertos e a acuracia.

### 3) Rodar testes

Ainda na raiz do projeto:

```bash
python -m unittest discover -s tests -v
```

Se tudo estiver correto, os testes aparecem com status `ok`.

## Boas praticas para estudar e evoluir

- Comece por `ann/Core/neuron.py` e `ann/Core/layer.py`
- Depois leia `ann/Core/network.py` para ver o fluxo completo
- Execute os exemplos e altere arquitetura/taxa de aprendizado
- Rode os testes apos cada mudanca

## Referencia

- [Kopec, David - Classic Computer Science Problems](https://classicproblems.com/)
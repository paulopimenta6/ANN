# Guia rapido do projeto ANN

Este arquivo explica, de forma simples, o que este projeto faz, o que foi melhorado e como voce pode usar no dia a dia.

## O que este projeto e

Este projeto implementa uma Rede Neural Artificial (ANN) do zero em Python, sem frameworks de machine learning.

A ideia principal e aprender os fundamentos:

- neuronio
- camada
- feedforward
- backpropagation
- atualizacao de pesos

## Estrutura principal

- `ann/Core/util.py`: funcoes matematicas (produto escalar, sigmoid, normalizacao)
- `ann/Core/neuron.py`: estrutura de um neuronio
- `ann/Core/layer.py`: organiza varios neuronios em uma camada
- `ann/Core/network.py`: controla treino, validacao e fluxo completo da rede
- `ann/examples/iris_test.py`: exemplo com dataset Iris
- `ann/examples/wine_test.py`: exemplo com dataset Wine
- `tests/`: testes automatizados com `unittest`

## Atualizacoes realizadas no projeto

Estas melhorias ja foram aplicadas:

1. **Caminhos de dados mais robustos**
   - Os exemplos agora localizam os CSVs usando `Path(__file__)`.
   - Isso evita erro de caminho quando o script e executado de outro diretorio.

2. **Reprodutibilidade**
   - Foi adicionado `seed(42)` nos exemplos.
   - Assim, os resultados ficam mais consistentes entre execucoes.

3. **Testes automatizados**
   - Criados testes para utilitarios e para a rede.
   - Isso ajuda a garantir que mudancas futuras nao quebrem o comportamento.

4. **Bias nos neuronios**
   - Cada neuronio agora possui `bias`.
   - O `bias` participa do calculo da saida e tambem e atualizado no treino.

5. **Sigmoid numericamente estavel**
   - A funcao sigmoid foi melhorada para evitar overflow com valores extremos.

6. **Pacotes explicitos**
   - Foram adicionados arquivos `__init__.py` para melhorar compatibilidade com ferramentas.

7. **Dependencias alinhadas**
   - O projeto foi padronizado para uso de bibliotecas da stdlib (sem dependencias externas obrigatorias).

## Como executar os exemplos

Na raiz do projeto (pasta que contem `ann/`), execute:

```bash
python -m ann.examples.iris_test
python -m ann.examples.wine_test
```

## Como rodar os testes

Ainda na raiz do projeto:

```bash
python -m unittest discover -s tests -v
```

Se tudo estiver certo, voce vera os testes com status `ok`.

## Explicacao simples do fluxo da rede

1. Voce envia uma entrada (ex: medidas de flor).
2. A rede calcula uma saida passando por camadas (feedforward).
3. A saida e comparada com o valor esperado.
4. O erro e propagado para tras (backpropagation).
5. Pesos e bias sao ajustados para melhorar a previsao.
6. Repetindo isso varias vezes, a rede aprende.

## Dicas para evoluir o projeto

- Adicionar mais testes para casos de erro e limites.
- Criar um script unico de treino com argumentos de linha de comando.
- Medir mais metricas alem de acuracia.
- Registrar historico de treino por epoca.

## Resumo rapido

Se voce quer apenas usar agora:

1. Abra terminal na raiz do projeto.
2. Rode um exemplo com `python -m ann.examples.iris_test`.
3. Rode os testes com `python -m unittest discover -s tests -v`.
4. Edite os modulos em `ann/Core/` para estudar e experimentar.

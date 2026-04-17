from typing import List, Callable, Tuple
from math import exp, tanh

# Funcao para calcular o produto escalar entre dois vetores
# Definicao: https://en.wikipedia.org/wiki/Dot_product
def dot_product(xs: List[float], ys: List[float]) -> float:
    # Faz o produto escalar entre dois vetores.
    return sum(x*y for x, y in zip(xs, ys))

# Sera usada uma funcao de ativacao da rede neural. 
# Sera usada a uma funcao sigmoide, cujo objetivo e mapear qualquer valor real para o intervalo entre 0 e 1.
# Mais detalhes podem ser vistos em: 
# [1] https://pt.wikipedia.org/wiki/Fun%C3%A7%C3%A3o_sigmoide
# [2] https://en.wikipedia.org/wiki/Logistic_regression 
def sigmoid(x: float) -> float:
    # Implementacao numericamente estavel para evitar overflow com valores extremos.
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    exp_x: float = exp(x)
    return exp_x / (1.0 + exp_x)

# derivada da função sigmoid
def derivative_sigmoid(x: float) -> float:
    sig: float = sigmoid(x)
    return sig*(1-sig)     

# Funcao de ativacao tanh: mapeia para o intervalo [-1, 1].
def tanh_activation(x: float) -> float:
    return tanh(x)

# Derivada de tanh: 1 - tanh(x)^2
def derivative_tanh_activation(x: float) -> float:
    tanh_x: float = tanh_activation(x)
    return 1.0 - (tanh_x * tanh_x)

# Funcao de ativacao ReLU: retorna zero para negativos.
def relu_activation(x: float) -> float:
    return x if x > 0.0 else 0.0

# Derivada de ReLU.
def derivative_relu_activation(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0

# Funcao de ativacao Leaky ReLU: pequeno gradiente para negativos.
def leaky_relu_activation(x: float, alpha: float = 0.01) -> float:
    return x if x > 0.0 else alpha * x

# Derivada de Leaky ReLU.
def derivative_leaky_relu_activation(x: float, alpha: float = 0.01) -> float:
    return 1.0 if x > 0.0 else alpha


def resolve_activation_functions(
    name: str, leaky_alpha: float = 0.01
) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """Retorna (funcao_ativacao, derivada) com base no nome informado."""
    normalized_name = name.strip().lower()

    if normalized_name == "sigmoid":
        return sigmoid, derivative_sigmoid
    if normalized_name == "tanh":
        return tanh_activation, derivative_tanh_activation
    if normalized_name == "relu":
        return relu_activation, derivative_relu_activation
    if normalized_name == "leaky_relu":
        return (
            lambda x: leaky_relu_activation(x, alpha=leaky_alpha),
            lambda x: derivative_leaky_relu_activation(x, alpha=leaky_alpha),
        )
    raise ValueError(
        "Funcao de ativacao invalida. Use: sigmoid, tanh, relu ou leaky_relu."
    )

# Supoe-se que todas as linhas tem o mesmo tamanho
# e esta funcao e a feature scaling de cada coluna para que esteja no intervalo de 0 a 1
# ou seja, a normalizacao.
# Todo neuronio em nossa rede gera valores entre 0 a 1 como resultado da funcao de ativacao signoide. 
# Parece logico que uma escala entre 0 a 1 faria sentido para os atributos do conjunto de dados de entrada tambem.
def normalize_by_feature_scaling(dataset: List[List[float]]) -> None:
    for col_num in range(len(dataset[0])):
        column: List[float] = [row[col_num] for row in dataset]
        maximum: float = max(column)
        minimum: float = min(column)
        if maximum - minimum == 0:
            for row_num in range(len(dataset)):
                dataset[row_num][col_num] = 0.0
        else:
            for row_num in range(len(dataset)):
                dataset[row_num][col_num] = (dataset[row_num][col_num] - minimum) / (maximum - minimum)

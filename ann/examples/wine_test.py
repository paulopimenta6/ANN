import csv
import argparse
from typing import List
from pathlib import Path
from ..Core.util import normalize_by_feature_scaling, resolve_activation_functions
from ..Core.network import Network
from random import shuffle, seed

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Treina ANN no dataset Wine.")
    parser.add_argument(
        "--activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "tanh", "relu", "leaky_relu"],
        help="Funcao de ativacao da rede.",
    )
    parser.add_argument(
        "--leaky-alpha",
        type=float,
        default=0.01,
        help="Alpha da Leaky ReLU.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed aleatoria.")
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas.")
    args = parser.parse_args()

    seed(args.seed)
    # Listas para armazenar os dados do conjunto Wine
    wine_parameters: List[List[float]] = []
    wine_classifications: List[List[float]] = []
    wine_species: List[int] = []
    data_path = Path(__file__).resolve().parents[1] / "data" / "wine.csv"
    with data_path.open(mode='r') as wine_file:
        wines: List = list(csv.reader(wine_file, quoting=csv.QUOTE_NONNUMERIC))
        shuffle(wines) # Deixa as linhas em ordem aleatoria
        for wine in wines:
            # Le cada linha sequencialmente ate o penultimo elemento e adiciona os dados na lista
            parameters: List[float] = [float(n) for n in wine[1:14]] 
            # Adiciona os parametros do vinho na lista wine_parameters
            wine_parameters.append(parameters)
            species: int = int(wine[0])
            if species == 1:
                wine_classifications.append([1.0,0.0,0.0]) # One-hot encoding para a classe 1
            elif species == 2:
                wine_classifications.append([0.0,1.0,0.0]) # One-hot encoding para a classe 2
            else:
                wine_classifications.append([0.0,0.0,1.0]) # One-hot encoding para a classe 3
            wine_species.append(species) # Armazena o numero da classe do vinho extraido da primeira coluna
    
    # Normaliza os parametros dos dados do vinho evitando possiveis problemas no treinamento
    normalize_by_feature_scaling(wine_parameters) 
    activation_function, derivative_activation_function = resolve_activation_functions(
        args.activation, leaky_alpha=args.leaky_alpha
    )
    # Cria a rede neural com 13 entradas, 7 neuronios na camada oculta, 3 saidas e taxa de aprendizado de 0.9
    wine_network: Network = Network(
        [13, 7, 3],
        0.9,
        activation_function=activation_function,
        derivative_activation_function=derivative_activation_function,
    )

    # Interpreta a saida da rede neural retornando a classe do vinho
    # Retorna 1, 2 ou 3 dependendo do indice do maior valor na lista de saida 
    def wine_interpret_output(output: List[float]) -> int:    
        if max(output) == output[0]:
            return 1
        elif max(output) == output[1]:
            return 2
        else:
            return 3

    # Treina a rede neural com os primeiros 150 dados do conjunto de vinhos, 10 vezes
    wine_trainers: List[List[float]] = wine_parameters[0:150]
    wine_trainers_corrects: List[List[float]] = wine_classifications[0:150]
    for _ in range(args.epochs):
        wine_network.train(wine_trainers, wine_trainers_corrects)

    # Testa a rede neural com os ultimos 28 dados do conjunto de vinhos
    wine_testers: List[List[float]] = wine_parameters[150:178]
    wine_testers_corrects: List[int] = wine_species[150:178]
    wine_results = wine_network.validate(wine_testers, wine_testers_corrects, wine_interpret_output)
    print(
        f"{wine_results[0]} correct of {wine_results[1]} = {wine_results[2]*100}% "
        f"(activation={args.activation}, epochs={args.epochs}, seed={args.seed})"
    )
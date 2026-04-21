# layer.py
# This file is based on code from:
# "Classic Computer Science Problems in Python" by David Kopec
# Original source: https://github.com/davecom/ClassicComputerScienceProblemsInPython
#
# Copyright 2018 David Kopec
#
# Modifications Copyright 2026 Paulo Pimenta
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#

from typing import List, Callable
from .util import dot_product

class Neuron:
    def __init__(self, weights:List[float], learning_rate: float, activation_function: Callable[[float], float],
    derivative_activation_function: Callable[[float], float]) -> None:
        self.weights: List[float] = weights
        self.activation_function: Callable[[float], float] = activation_function
        self.derivative_activation_function: Callable[[float], float] = derivative_activation_function
        self.learning_rate: float = learning_rate
        self.output_cache: float = 0.0
        self.delta: float = 0.0

    def output(self, inputs:List[float]) -> float:
        self.output_cache = dot_product(inputs, self.weights)
        return self.activation_function(self.output_cache)
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:24:53 2025

@author: Thewo
"""

import numpy as np

class Perceptron:
    def __init__(self, n_entradas):
        # Inicializar pesos y bias (término independiente)
        self.pesos = np.random.uniform(-1, 1, n_entradas)
        self.bias = np.random.uniform(-1, 1)
    
    def calcular(self, entradas):
        # Calcular la suma ponderada de entradas más el bias
        return np.dot(self.pesos, entradas) + self.bias
    
    def actualizar_pesos(self, delta_pesos, delta_bias):
        # Actualizar pesos y bias
        try:
            self.pesos += delta_pesos
            self.bias += delta_bias
        except Exception as e:
            print(e)
            
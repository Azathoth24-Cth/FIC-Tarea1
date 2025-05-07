# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:24:27 2025

@author: Thewo
"""

from Perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
from Auxiliar import divisionDatos

class RedNeuronal:
    def __init__(self, topologia, tasa_aprendizaje=0.1):
        """
        Constructor de la red neuronal.
        
        Parámetros:
        topologia -- array que especifica el número de neuronas en cada capa.
                     Ejemplo: [2, 3, 2] sería 2 entradas, 1 capa oculta con 3 neuronas
                     y 1 capa de salida con 2 neuronas.
        """
        self.topologia = topologia
        self.n_entradas = topologia[0]
        self.n_salidas = topologia[-1]
        self.errores = []
        self.tasa_aprendizaje = tasa_aprendizaje
        # Crear todas las capas (incluyendo la de salida)
        self.capas = []
        for i in range(1, len(topologia)):
            # Crear una capa de perceptrones
            capa = [Perceptron(topologia[i-1]) for _ in range(topologia[i])]
            self.capas.append(capa)
    
    def _sigmoide(self, x):
        """Función de activación sigmoide"""
        # Para evitar overflow, se utiliza np.clip para limitar el rango de x
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def _derivada_sigmoide(self, x):
        """Derivada de la función sigmoide"""
        #para evitar overflow, se utiliza np.clip para limitar el rango de x
        x = np.clip(x, -500, 500)
        
        sig = self._sigmoide(x)
        return sig * (1 - sig)
    
    def calcular(self, entradas):
        """
        Propagación hacia adelante de las entradas a través de la red.
        
        Parámetros:
        entradas -- array de valores de entrada
        
        Retorna:
        Array con los valores de salida de la red
        """
        if len(entradas) != self.n_entradas:
            raise ValueError(f"Número de entradas incorrecto. Esperado: {self.n_entradas}, Recibido: {len(entradas)}")
        
        salida_capa = np.array(entradas)
        
        # Propagación a través de cada capa
        for capa in self.capas:
            nuevas_salidas = []
            for neurona in capa:
                # Calcular suma ponderada y aplicar función de activación
                suma_ponderada = neurona.calcular(salida_capa)
                activacion = self._sigmoide(suma_ponderada)
                nuevas_salidas.append(activacion)
            salida_capa = np.array(nuevas_salidas)
        
        return salida_capa
    
    def entrenar(self, entradas, objetivo):
        """
        Entrena la red con un par entrada-objetivo usando backpropagation
        
        Parámetros:
        entradas -- array de valores de entrada
        objetivo -- array con los valores esperados de salida
        """
        # 1. Propagación hacia adelante (forward pass)
        salida_capa = np.array(entradas)
        salidas_por_capa = [salida_capa]  # Guardamos las salidas de cada capa
        sumas_ponderadas = []  # Guardamos las sumas ponderadas de cada capa
        
        for capa in self.capas:
            nuevas_salidas = []
            nuevas_sumas = []
            
            for neurona in capa:
                suma_ponderada = neurona.calcular(salida_capa)
                activacion = self._sigmoide(suma_ponderada)
                nuevas_salidas.append(activacion)
                nuevas_sumas.append(suma_ponderada)
                
            salida_capa = np.array(nuevas_salidas)
            salidas_por_capa.append(salida_capa)
            sumas_ponderadas.append(np.array(nuevas_sumas))
        
        # 2. Cálculo del error
        error = objetivo - salida_capa
        self.errores.append(np.abs(error))
        
        # 3. Backpropagation (propagación hacia atrás del error)
        deltas = []
        
        # Capa de salida
        delta = error * self._derivada_sigmoide(sumas_ponderadas[-1])
        deltas.append(delta)
        
        # Capas ocultas (hacia atrás)
        for i in range(len(self.capas)-2, -1, -1):
            capa_siguiente = self.capas[i+1]
            suma_ponderada = sumas_ponderadas[i]
            
            error_capa = np.zeros(len(self.capas[i]))
            for j, neurona in enumerate(capa_siguiente):
                error_capa += neurona.pesos * deltas[-1][j]
                
            delta = error_capa * self._derivada_sigmoide(suma_ponderada)
            deltas.append(delta)
        
        deltas.reverse()  # Para que coincida con el orden de las capas
        
        # 4. Actualización de pesos
        for i in range(len(self.capas)):
            capa = self.capas[i]
            entrada_capa = salidas_por_capa[i]
            
            for j in range(len(capa)):
                neurona = capa[j]
                
                # Gradiente para los pesos
                gradiente_pesos = deltas[i][j] * entrada_capa
                
                # Gradiente para el bias (es igual a delta)
                gradiente_bias = deltas[i][j]
                
                # Actualización
                neurona.actualizar_pesos(
                    self.tasa_aprendizaje * gradiente_pesos,
                    self.tasa_aprendizaje * gradiente_bias
                )
        #if len(self.errores)>1:
        #    if np.abs(self.errores[-1]-self.errores[-2])<0.001:
        #        self.tasa_aprendizaje*=1.05
        return
    
    def entrenar_y_graficar(self, entradas, objetivos, epocas, titulo):
        x_train, y_train, x_test, y_test = divisionDatos(entradas, objetivos)
        errores_epoca = []
        errores_test = []
        for epoca in range(epocas):
            errores = []
            for x_tr, y_tr in zip(x_train, y_train):
                self.entrenar(x_tr, y_tr)
                errores.append(self.errores[-1])
            error_promedio = np.mean(errores)
            errores_epoca.append(error_promedio)
            err_test =[]
            for x, y in zip(x_test, y_test):
                predict = self.calcular(x)
                err_test.append(np.abs(predict-y))
            err_test = np.array(err_test)
            errores_test.append(np.mean(err_test))
            if epoca % (epocas//10) == 0:
                print(f"Época {epoca}, Error: {error_promedio:.6f}")
        # Gráfica
        plt.figure(figsize=(10, 6))
        plt.plot(range(epocas), errores_epoca, 'b-', label='error entrenamiento')
        plt.plot(range(epocas), errores_test,'r--', label = 'error prueba')
        plt.title(f"Evolución del Error - {titulo}")
        plt.xlabel("Época")
        plt.ylabel("Error Promedio")
        plt.grid(True)
        plt.legend()
        plt.show()
        
        
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 13:22:44 2025

@author: Thewo
"""
import numpy as np
import pandas as pd

def CargarDatos():
    ruta = "DatosConClases.xlsx"
    dataFrame = pd.read_excel(ruta)
    Muestras = []
    Componentes = []
    #se itera sobre las filasdel dataframe
    for header in dataFrame:
        if header=='Unnamed: 9':
            break
        Componentes.append(header)
        Columna = dataFrame[header]
        Columna = np.array(Columna)
        Muestras.append(Columna)
    Muestras = np.array(Muestras)
    Muestras = Muestras.T
    Etiquetas1 = dataFrame['Clase Entrega 1'].to_list()
    Etiquetas2 = dataFrame['Clase Entrega 2'].to_list()
    Etiquetas3 = dataFrame['Clase Entrega 3'].to_list()
    return Muestras, Componentes, Etiquetas1, Etiquetas2, Etiquetas3

def divisionDatos(x, y, rate=0.2,seed=42):
    #se busca tomar de manera aleatoria los conjuntos de x y y tratanto de respectar
    #el desbalance
    #se crean arreglos temporales donde se guarden los datos por clase
    #se buscan los y unicos
    Clases = set(y)
    #Se extraen las muestras por clase
    MuestrasPorClase = {}
    for clase in Clases:
        MuestrasPorClase[clase] = []
    for i in range(len(y)):
        MuestrasPorClase[y[i]].append(x[i])
    #se convierten a numpy arrays
    for clase in Clases:
        MuestrasPorClase[clase] = np.array(MuestrasPorClase[clase])
    #se inicializan los conjuntos de entrenamiento y prueba
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    #se itera sobre las clases para dividirlas
    for clase in Clases:
        #se obtiene el numero de muestras por clase
        num_muestras = len(MuestrasPorClase[clase])
        #se calcula el numero de muestras de prueba
        num_prueba = int(num_muestras * rate)
        #se genera un arreglo aleatorio para las muestras
        np.random.seed(seed)
        indices = np.random.permutation(num_muestras)
        #se dividen las muestras en entrenamiento y prueba
        x_train.extend(MuestrasPorClase[clase][indices[num_prueba:]])
        y_train.extend([clase] * (num_muestras - num_prueba))
        x_test.extend(MuestrasPorClase[clase][indices[:num_prueba]])
        y_test.extend([clase] * num_prueba)
    #se convierten a numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    #se mezclan los datos de entrenamiento y prueba
    indices_train = np.random.permutation(len(x_train))
    indices_test = np.random.permutation(len(x_test))
    x_train = x_train[indices_train]
    y_train = y_train[indices_train]
    x_test = x_test[indices_test]
    y_test = y_test[indices_test]

    return x_train, y_train, x_test, y_test

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 22:57:33 2025

@author: Thewo
"""
import numpy as np
import matplotlib.pyplot as plt
from RedNeuronal import RedNeuronal as RN
from PerceptronBolsillo import PerceptronBolsillo
from Auxiliar import CargarDatos, divisionDatos

def LMS_monocapa(x, y, mu, max_epochs, epsilon):
    # División de datos (asumiendo que divisionDatos está definida)
    x_train, y_train, x_test, y_test = divisionDatos(x, y, 0.2)
    
    # Identificar clases únicas y calcular umbral
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("Se requieren exactamente dos clases para clasificación binaria.")
    
    # Umbral adaptativo: punto medio entre las dos clases
    threshold =1.2815
    positive_class, negative_class = classes[1], classes[0]
    
    # Inicialización de pesos
    m = x_train.shape[1]
    w = np.random.randn(m) * 0.01
    
    # Historial de métricas
    train_mse = []
    test_mse = []
    test_acc = []
    sensitivity_list = []
    specificity_list = []
    
    for epoch in range(max_epochs):
        # Entrenamiento
        error = 0
        for i in range(len(x_train)):
            y_pred = np.dot(w, x_train[i])
            e = y_train[i] - y_pred
            w += mu * e * x_train[i]
            error += e**2
        train_mse.append(error / len(x_train))
        
        # Evaluación
        y_test_pred = np.dot(x_test, w)
        test_mse.append(np.mean((y_test - y_test_pred)**2))
        
        # Clasificación con umbral adaptativo
        y_test_class = np.where(y_test_pred >= threshold, positive_class, negative_class)
        
        # Calcular métricas de sensibilidad y especificidad
        TP = np.sum((y_test_class == positive_class) & (y_test == positive_class))
        FN = np.sum((y_test_class == negative_class) & (y_test == positive_class))
        TN = np.sum((y_test_class == negative_class) & (y_test == negative_class))
        FP = np.sum((y_test_class == positive_class) & (y_test == negative_class))
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

        accuracy = np.mean(y_test_class == y_test)
        test_acc.append(accuracy)
        
        if train_mse[-1] < epsilon:
            break
    
    # Gráfica de MSE
    plt.figure(figsize=(10, 6))
    plt.plot(train_mse, 'b-', label='Train MSE', linewidth=2)
    plt.plot(test_mse, 'g--', label='Test MSE', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.title('Curvas de MSE (Train y Test)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Gráfica de Accuracy, Sensitivity y Specificity
    plt.figure(figsize=(10, 6))
    plt.plot(test_acc, 'r-', label='Test Accuracy', linewidth=2)
    plt.plot(sensitivity_list, 'b--', label='Sensitivity', linewidth=2)
    plt.plot(specificity_list, 'g-.', label='Specificity', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Métricas')
    plt.title('Curvas de Accuracy, Sensitivity y Specificity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return w, train_mse, test_mse, test_acc

def Punto1(Muestras, Componentes, Etiquetas):
    #Implemente una red neuronal monocapa con algoritmo LMS, de tal manera que se diferencie
    #entre aves y otras especies.
    mu= 0.01       # Tasa de aprendizaje
    max_epochs = 5000
    epsilon = 1e-5   # Criterio de parada
    LMS_monocapa(Muestras, Etiquetas, mu, max_epochs, epsilon)
    return

def Punto2(Muestras, Componentes, Etiquetas):
    # Implementa una red neuronal monocapa con algoritmo de entrenamiento Perceptron (de
    # bolsillo), de tal manera que se diferencie entre aves y otras especies.
    print("\n" + "="*50)
    print("ENTRENANDO")
    print("="*50)
    entradas = Muestras
    objetivos_originales = Etiquetas
    # Codificar las etiquetas a 0 y 1 (asumiendo 1 es no-ave, 2 es ave)
    objetivos_codificados = np.array([1 if etiq == 2 else 0 for etiq in objetivos_originales])

    x_train, y_train, x_test, y_test = divisionDatos(entradas, objetivos_codificados)
    Modelo = PerceptronBolsillo(n_entradas=len(Componentes), tasa_aprendizaje=0.05, max_iter=10000)
    historial_errores = Modelo.train(x_train, y_train)

    y_pred_test = Modelo.predict(x_test)

    accuracy = np.mean(y_pred_test == y_test)

    tp = np.sum((y_pred_test == 1) & (y_test == 1))
    tn = np.sum((y_pred_test == 0) & (y_test == 0))
    fp = np.sum((y_pred_test == 1) & (y_test == 0))
    fn = np.sum((y_pred_test == 0) & (y_test == 1))

    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Accuracy en el conjunto de prueba: {accuracy:.4f}")
    print(f"Sensibilidad en el conjunto de prueba: {sensibilidad:.4f}")
    print(f"Especificidad en el conjunto de prueba: {especificidad:.4f}")

    # --- Generación de las gráficas ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(historial_errores) + 1), historial_errores, marker='o')
    plt.title('Historial de Error durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Error (1 - Accuracy)')
    plt.grid(True)

    metricas = ['Accuracy', 'Sensibilidad', 'Especificidad']
    valores = [accuracy, sensibilidad, especificidad]

    plt.figure(figsize=(8, 5))
    plt.bar(metricas, valores, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Métricas de Evaluación en el Conjunto de Prueba')
    plt.ylabel('Valor')
    plt.ylim([0, 1])
    for i, v in enumerate(valores):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.show()

    return

def Punto3(Muestras, Componentes, Etiquetas):
    #Implemente una red neuronal tipo Perceptron multicapa, con algoritmo de entrenamiento back-
    #propagation, de tal manera que la red diferencie entre aves y otras especies. Usted decide si usa
    #tasa de aprendizaje variable o fija.
    print("\n" + "="*50)
    print("ENTRENANDO")
    print("="*50)
    topologia = [9,7,5,3,1]
    red = RN(topologia, tasa_aprendizaje=0.5)
    red.entrenar_y_graficar(Muestras, Etiquetas, 5000, "Clasificación Aves -Otras Especies: backpropagation")
    return
Muestras, Componentes, Etiquetas1, Etiquetas2, Etiquetas3 = CargarDatos()
#Punto1(Muestras, Componentes, Etiquetas1)
Punto2(Muestras, Componentes, Etiquetas1)
#Punto3(Muestras, Componentes, Etiquetas1)

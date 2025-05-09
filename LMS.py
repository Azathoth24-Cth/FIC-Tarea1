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
from Auxiliar import CargarBaseDeDatosImagenes
import tensorflow as tf
from tensorflow.python.keras import layers, models,regularizers

from sklearn.model_selection import train_test_split
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
    print("\n" + "="*50)
    print("ENTRENANDO (Versión Diagrama de Flujo)")
    print("="*50)
    entradas = Muestras
    objetivos_originales = Etiquetas
    # Codificar las etiquetas a -1 y +1 (1 -> -1, 2 -> +1)
    objetivos_codificados = np.array([-1 if etiq == 1 else 1 for etiq in objetivos_originales])

    x_train, y_train, x_test, y_test = divisionDatos(entradas, objetivos_codificados, seed=42)
    Modelo = PerceptronBolsillo(n_entradas=len(Componentes), learning_rate=0.1, max_iter=1000)
    historial_errores = Modelo.train(x_train, y_train)

    y_pred_test = Modelo.predict(x_test)

    accuracy = np.mean(y_pred_test == y_test)

    tp = np.sum((y_pred_test == 1) & (y_test == 1))
    tn = np.sum((y_pred_test == -1) & (y_test == -1))
    fp = np.sum((y_pred_test == 1) & (y_test == -1))
    fn = np.sum((y_pred_test == -1) & (y_test == 1))

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
    plt.ylabel('Error Promedio por Época')
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
    
    plt.figure(figsize=(8, 5))
    plt.hist(historial_errores, bins=50, color='steelblue', edgecolor='black')
    plt.title('Distribución del Error durante el Entrenamiento')
    plt.xlabel('Error Promedio por Época')
    plt.ylabel('Frecuencia')
    plt.grid(True)
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
    red = RN(topologia, tasa_aprendizaje=1)
    red.entrenar_y_graficar(Muestras, Etiquetas, 5000, "Clasificación Aves -Otras Especies: backpropagation")
    return
def Punto4():
    #Implemente/entrene una red neuronal convolucional que permita diferenciar entre aves y otras
    #especies usando las imágenes proporcionadas en el siguiente enlace:
    #https://drive.google.com/drive/folders/1X_gZM_jRXcZS2kHxk158eksqvh1N0Ul4?usp=sharing
    #En este caso, la distribución de imagen por cada género corresponde a:
    #Artiodactyla: 643 imágenes
    #Carnivora: 546 imágenes
    #Cingulata: 469 imágenes
    #Pilosa: 204 imágenes
    #Rodentia: 1150 imágenes
    #Aves: 1272 imágenes
    # 1. Cargar y preprocesar datos
    X, y = CargarBaseDeDatosImagenes(target_size=(250, 250))  # Tamaño recomendado para CNNs
    X = X / 255.0  # Normalizar píxeles [0, 1]
    X = np.expand_dims(X, axis=-1)
    # 2. Dividir datos (80% train, 20% test)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, y_train, X_test, y_test = divisionDatos(X, y, seed=42)
    # 3. Definir arquitectura CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 1)),  # Primera capa convolucional
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(1, activation='sigmoid')  # Salida binaria (aves vs. otras)
    ])
    
    # 4. Compilar el modelo
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # 5. Entrenar y graficar
    history = model.fit(X_train, y_train, 
                        epochs=20, 
                        batch_size=32,
                        validation_data=(X_test, y_test))
    
    # 6. Graficar Pérdida (Loss) y Accuracy
    plt.figure(figsize=(12, 5))
    
    # Gráfica de Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Pérdida por Época')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (Loss)')
    plt.legend()

    # Gráfica de Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Precisión por Época')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    return model

Muestras, Componentes, Etiquetas1, Etiquetas2, Etiquetas3 = CargarDatos()
#Punto1(Muestras, Componentes, Etiquetas1)
Punto2(Muestras, Componentes, Etiquetas1)
#Punto3(Muestras, Componentes, Etiquetas1)
#Punto4()

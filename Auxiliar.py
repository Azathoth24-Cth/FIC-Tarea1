# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 13:22:44 2025

@author: Thewo
"""
import numpy as np
import pandas as pd

#para el punto #4
import os 
from PIL import Image 

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



def CargarBaseDeDatosImagenes():
    X = []
    Y = []
    
    # Get the path to the HW1 directory (sibling of current directory)
    base_dir = os.path.join(os.path.dirname(__file__), 'HW1')
    
    # Define the class folders and their corresponding labels
    class_folders = {
        'Aves': 0,
        'Artiodactyla': 1,
        'Carnivora': 1,
        'Cingulata': 1,
        'Pilosa': 1,
        'Rodentia': 1
    }
    
    # Iterate through each class folder
    for folder_name, label in class_folders.items():
        folder_path = os.path.join(base_dir, folder_name)
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            continue
            
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.jpg'):
                # Construct the full file path
                file_path = os.path.join(folder_path, filename)
                
                try:
                    # Open the image and convert it to RGB (in case it's grayscale)
                    img = Image.open(file_path).convert('RGB')
                    
                    # Convert image to numpy array and add to X
                    # Note: You might want to resize images to a consistent size here
                    X.append(np.array(img))
                    Y.append(label)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
    
    return np.array(X), np.array(Y)
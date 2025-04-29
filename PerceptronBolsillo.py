import numpy as np
import matplotlib.pyplot as plt

class PerceptronBolsillo:
    def __init__(self, n_entradas, tasa_aprendizaje=0.1, max_iter=1000):
        self.w = np.zeros(n_entradas + 1) # Weights including bias (w[0])
        self.learning_rate = tasa_aprendizaje
        self.max_iter = max_iter
        self.best_w = self.w.copy()
        self.best_accuracy = -1
        self.historial_errores = []

    def predict(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X] # Add bias term
        return np.where(X_b @ self.w >= 0, 1, 0) # Assuming labels 0 and 1

    def _accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def train(self, X, y):
        X_b = np.c_[np.ones(X.shape[0]), X] # Add bias term
        for epoch in range(self.max_iter):
            misclassified = False
            for idx, xi in enumerate(X_b):
                prediction = 1 if xi @ self.w >= 0 else 0 # Assuming labels 0 and 1
                if prediction != y[idx]:
                    misclassified = True
                    self.w += self.learning_rate * (y[idx] - prediction) * xi
                    current_accuracy = self._accuracy(X, y)
                    if current_accuracy > self.best_accuracy:
                        self.best_w = self.w.copy()
                        self.best_accuracy = current_accuracy

            self.historial_errores.append(1 - self._accuracy(X, y)) # Store error (1 - accuracy)
            if not misclassified:
                break
        self.w = self.best_w # Use the best weights found
        return self.historial_errores
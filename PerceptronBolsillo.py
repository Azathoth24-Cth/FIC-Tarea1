import numpy as np

class PerceptronBolsillo:
    def __init__(self, n_entradas, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.w = np.zeros(n_entradas + 1)  # +1 por el bias
        self.w_bolsillo = self.w.copy()
        self.menor_error = np.inf
        self.historial_errores = []

    def _agregar_bias(self, x):
        return np.insert(x, 0, 1)  # Inserta bias en posición 0

    def predict_single(self, x):
        x_b = self._agregar_bias(x)
        return 1 if np.dot(self.w, x_b) >= 0 else -1

    def predict(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X]
        return np.where(np.dot(X_b, self.w) >= 0, 1, -1)

    def evaluate_error(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred != y)

    def train(self, X, y):
        X_b = np.c_[np.ones(X.shape[0]), X]  # Agrega bias a todas las muestras
        y = np.array(y)

        for epoca in range(self.max_iter):
            errores = 0
            for i in np.random.permutation(len(X)):
                xi = X_b[i]
                yi = y[i]
                pred = 1 if np.dot(self.w, xi) >= 0 else -1
                if pred != yi:
                    self.w += self.learning_rate * yi * xi
                    errores += 1

            error_actual = self.evaluate_error(X, y)
            self.historial_errores.append(error_actual)

            if error_actual < self.menor_error:
                self.menor_error = error_actual
                self.w_bolsillo = self.w.copy()

            if errores == 0:
                break  # Convergencia

        self.w = self.w_bolsillo.copy()
        return self.historial_errores
#Clase Network
#~~~~~~~~~~~~
'''
Una clase que representa una red neuronal. Esta red neuronal se utiliza para tareas de aprendizaje automático, 
como reconocimiento de patrones o clasificación. La implementación sigue el concepto de una red neuronal 
alimentada hacia adelante (feedforward neural network).
'''
#### Bibliotecas
# Biblioteca estándar
import random

# Bibliotecas de terceros
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """El argumento 'sizes' es una lista que contiene el número de neuronas en las
        capas respectivas de la red. Por ejemplo, si la lista fuera [2, 3, 1], entonces
        sería una red de tres capas, con la primera capa que contiene 2 neuronas, la
        segunda capa 3 neuronas y la tercera capa 1 neurona. Los sesgos (biases) y pesos
        de la red se inicializan aleatoriamente utilizando una distribución gaussiana
        con media 0 y varianza 1. Es importante destacar que se asume que la primera capa
        es una capa de entrada, y por convención no se establecen sesgos para esas neuronas,
        ya que los sesgos solo se utilizan en el cálculo de las salidas de las capas posteriores.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Devuelve la salida de la red si 'a' es la entrada."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Entrena la red neuronal utilizando el descenso de gradiente estocástico por lotes
        miniatura. El 'training_data' es una lista de tuplas '(x, y)' que representan las
        entradas de entrenamiento y las salidas deseadas. Los otros parámetros son
        autoexplicativos. Si 'test_data' se proporciona, la red se evaluará con los datos de
        prueba después de cada época y se imprimirá el progreso parcial. Esto es útil para
        realizar un seguimiento del progreso, pero ralentiza sustancialmente el proceso."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Época {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Época {0} completada".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Actualiza los pesos y sesgos de la red aplicando el descenso de gradiente utilizando
        la retropropagación a un solo mini lote. El 'mini_batch' es una lista de tuplas '(x, y)',
        y 'eta' es la tasa de aprendizaje."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Devuelve una tupla '(nabla_b, nabla_w)' que representa el gradiente para la función de
        costo C_x. 'nabla_b' y 'nabla_w' son listas de numpy arrays, capa por capa, similares a
        'self.biases' y 'self.weights'."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Propagación hacia adelante (feedforward)
        activation = x
        activations = [x]  # Lista para almacenar todas las activaciones, capa por capa
        zs = []  # Lista para almacenar todos los vectores z, capa por capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Retropropagación (backward pass)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Tener en cuenta que la variable 'l' en el bucle a continuación se utiliza de manera
        # un poco diferente a la notación en el Capítulo 2 del libro. Aquí, 'l = 1' significa
        # la última capa de neuronas, 'l = 2' es la segunda última capa, y así sucesivamente.
        # Es una renumeración del esquema en el libro, utilizada aquí para aprovechar el hecho
        # de que Python puede usar índices negativos en listas.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Devuelve el número de entradas de prueba para las cuales la red neuronal produce el
        resultado correcto. Se asume que la salida de la red neuronal es el índice de la neurona
        en la última capa con la activación más alta."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Devuelve el vector de derivadas parciales \partial C_x / \partial a para las
        activaciones de salida."""
        return (output_activations-y)

#### Funciones misceláneas
def sigmoid(z):
    """La función sigmoide."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivada de la función sigmoide."""
    return sigmoid(z)*(1-sigmoid(z))
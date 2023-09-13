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
    '''
    Se definen 7 funciones para la clase Network:
    1. _init_
    2. feedforward
    3. rmsprop
    4. update_mini_batch
    5. backprop
    6. evaluate
    7. cost_derivative
    Estas funciones representan un flujo común de desarrollo de una red y ala vez partes
    escenciales.
    '''
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

        # El número de capas en la red es igual al tamaño de la lista 'sizes'
        self.num_layers = len(sizes)

        # 'sizes' es una lista que contiene el número de neuronas en cada capa
        self.sizes = sizes

        # Inicialización aleatoria de biases para las capas ocultas y de salida.
        # Se utiliza np.random.randn para generar valores aleatorios con distribución gaussiana.
        # sizes[1:] omite la primera capa ya que no se establecen sesgos para las neuronas de entrada.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]


        # Inicialización aleatoria de pesos para las conexiones entre neuronas en capas.
        # Se utiliza zip para combinar tamaños de capas consecutivas.
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    
    # Funciones Para implementar cross-entropy en capa softmax
    def softmax(self, z):
        """Función Softmax para calcular las probabilidades de clase"""
        exp_z = np.exp(z - np.max(z))  # Restar np.max(z) para evitar problemas
        return exp_z / np.sum(exp_z, axis=0)

    def cost_derivative(self, output_activations, y):
        """Devuelve el vector de derivadas parciales \partial C_x / \partial a para las
        activaciones de salida utilizando la entropía cruzada"""
        return output_activations - y
    

    def feedforward(self, a):
        """Devuelve la salida de la red si 'a' es la entrada."""
        # Comienza un bucle que iterará a través de las capas de la red.    
        for b, w in zip(self.biases, self.weights):
            # 1. Calcula el producto escalar (dot product) entre los pesos 'w' y la activación 'a' de la capa anterior.
            # 2. Luego, suma el sesgo 'b' a ese producto escalar.
            # 3. Luego, el resultado se pasa a través de una función de activación llamada 'softmax'.
            z = np.dot(w, a) + b
            a = self.softmax(z)
        return a


    def SGD_M(self, training_data, epochs, mini_batch_size, eta, momentum, test_data=None):
        """
        Entrena la red neuronal utilizando el descenso de gradiente estocástico con momento. 
        El 'training_data' es una lista de tuplas '(x, y)' que representan las
        entradas de entrenamiento y las salidas deseadas. Los otros parámetros son
        autoexplicativos. Si 'test_data' se proporciona, la red se evaluará con los datos de
        prueba después de cada época y se imprimirá el progreso parcial. Esto es útil para
        realizar un seguimiento del progreso, pero ralentiza sustancialmente el proceso.
        """
        
        # Si se proporciona 'test_data', se calcula la cantidad de ejemplos de prueba
        if test_data: n_test = len(test_data)
        
        # Se calcula la cantidad de ejemplos de entrenamiento
        n = len(training_data)
        
        # Inicializa los gradientes acumulativos de los sesgos y los pesos con ceros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
    
        # Inicia el momento en cero para los gradientes
        momenta_b = [np.zeros(b.shape) for b in self.biases]
        momenta_w = [np.zeros(w.shape) for w in self.weights]

        # Se inicia un bucle que recorrerá el número especificado de épocas
        for j in range(epochs):
            # Se barajan aleatoriamente los datos de entrenamiento en cada época
            random.shuffle(training_data)
        
            # Se divide el conjunto de datos de entrenamiento en mini batches
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            # Se itera a través de cada mini lote y se actualizan los pesos y sesgos de la red
            for mini_batch in mini_batches:
                delta_nabla_b, delta_nabla_w = self.update_mini_batch(mini_batch, eta)
            
                # Actualiza los gradientes acumulativos con el término de momento
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
                # Actualiza el momento de los gradientes
                momenta_b = [momentum * mb - eta * nb for mb, nb in zip(momenta_b, nabla_b)]
                momenta_w = [momentum * mw - eta * nw for mw, nw in zip(momenta_w, nabla_w)]
            

            # Si 'test_data' está presente, se evalúa el rendimiento de la red en los datos de prueba
            # y se imprime el progreso parcial. Si no se proporcionan datos de prueba, se imprime el 
            # progreso de la época.
            if test_data:
                print("Época {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Época {0} completada".format(j))

    def RMSprop(self, training_data, epochs, mini_batch_size, eta, decay_rate, epsilon, test_data=None):
        """
        Entrena la red neuronal utilizando el optimizador RMSprop.
        'training_data' es una lista de tuplas '(x, y)' que representan las entradas de entrenamiento y
        las salidas deseadas. Los otros parámetros son autoexplicativos. Si 'test_data' se proporciona,
        la red se evaluará con los datos de prueba después de cada época y se imprimirá el progreso
        parcial. Esto es útil para realizar un seguimiento del progreso, pero ralentiza sustancialmente
        el proceso.
        """
    
        # Si se proporciona 'test_data', se calcula la cantidad de ejemplos de prueba
        if test_data: n_test = len(test_data)
    
        # Se calcula la cantidad de ejemplos de entrenamiento
        n = len(training_data)
    
        # Inicializa los gradientes acumulativos de los sesgos y los pesos con ceros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
    
        # Inicializa el promedio móvil de los cuadrados de los gradientes pasados
        rmsprop_b = [np.zeros(b.shape) for b in self.biases]
        rmsprop_w = [np.zeros(w.shape) for w in self.weights]
    
        # Comienza un bucle que recorrerá el número especificado de épocas
        for j in range(epochs):
            # Baraja aleatoriamente los datos de entrenamiento en cada época
            random.shuffle(training_data)
        
            # Divide el conjunto de datos de entrenamiento en mini lotes
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
        
            # Itera a través de cada mini lote y actualiza los pesos y sesgos de la red con RMSprop
            for mini_batch in mini_batches:
                delta_nabla_b, delta_nabla_w = self.update_mini_batch(mini_batch, eta)
            
                # Actualiza el promedio móvil de los cuadrados de los gradientes pasados
                rmsprop_b = [(decay_rate * rb) + ((1 - decay_rate) * dnb**2) for rb, dnb in zip(rmsprop_b, delta_nabla_b)]
                rmsprop_w = [(decay_rate * rw) + ((1 - decay_rate) * dnw**2) for rw, dnw in zip(rmsprop_w, delta_nabla_w)]
            
                # Actualiza los pesos de la red con RMSprop
                self.weights = [w - (eta / (np.sqrt(rb) + epsilon)) * nw for w, rb, nw in zip(self.weights, rmsprop_w, delta_nabla_w)]
                # Actualiza los sesgos de la red con RMSprop
                self.biases = [b - (eta / (np.sqrt(rb) + epsilon)) * nb for b, rb, nb in zip(self.biases, rmsprop_b, delta_nabla_b)]

            # Si 'test_data' está presente, evalúa el rendimiento de la red en los datos de prueba
            # e imprime el progreso parcial. Si no se proporcionan datos de prueba, imprime el 
            # progreso de la época.
            if test_data:
                print("Época {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Época {0} completada".format(j))



    def update_mini_batch(self, mini_batch, eta):
        """Actualiza los pesos y sesgos de la red aplicando el descenso de gradiente utilizando
        la retropropagación a un solo mini lote. El 'mini_batch' es una lista de tuplas '(x, y)',
        y 'eta' es la tasa de aprendizaje."""
        # Inicializa los gradientes acumulativos de los sesgos y los pesos con ceros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Itera a través de cada par (x, y) en el mini batch
        for x, y in mini_batch:
            # Calcula los gradientes de los sesgos y los pesos utilizando backpropagation
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # Acumula los gradientes en nabla_b y nabla_w utilizando la suma de vectores
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Actualiza los pesos de la red utilizando el descenso de gradiente
        # Se multiplica cada gradiente (nabla_w) por la tasa de aprendizaje (eta) dividida por el tamaño del mini batch   
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        # Actualiza los sesgos de la red de manera similar a los pesos
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        return delta_nabla_b, delta_nabla_w


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
            activation = self.softmax(z)    # Utilizamos softmax para activar capas
            activations.append(activation)
        # Retropropagación (backward pass)
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Tener en cuenta que la variable 'l' en el bucle a continuación se utiliza de manera
        # un poco diferente a la notación en el Capítulo 2 del libro. Aquí, 'l = 1' significa
        # la última capa de neuronas, 'l = 2' es la segunda última capa, y así sucesivamente.
        # Es una renumeración del esquema en el libro, utilizada aquí para aprovechar el hecho
        # de que Python puede usar índices negativos en listas.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.softmax(z) * (1 - self.softmax(z))
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Devuelve el número de entradas de prueba para las cuales la red neuronal produce el
        resultado correcto. Se asume que la salida de la red neuronal es el índice de la neurona
        en la última capa con la activación más alta.
        """
        # Utiliza la función 'feedforward' para calcular la salida de la red para cada entrada de prueba 'x'
        # Luego, utiliza 'np.argmax' para encontrar el índice de la neurona con la activación más alta
        # Esto se compara con la etiqueta 'y' correspondiente de la entrada de prueba
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
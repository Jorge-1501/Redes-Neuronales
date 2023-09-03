#mnist_loader
#~~~~~~~~~~~~
'''
Un conjunto de funciones para cargar los datos de imágenes de MNIST. Para obtener detalles sobre las 
estructuras de datos que se devuelven, consulta las cadenas de documentación para las funciones 
``load_data`` y ``load_data_wrapper``. En la práctica, la función ``load_data_wrapper`` es la función 
que normalmente se llama en nuestro código de redes neuronales.
'''
#### Bibliotecas
# Biblioteca estándar
import gzip
import pickle

# Bibliotecas de terceros
import numpy as np

def load_data():
    """Devuelve los datos de MNIST como una tupla que contiene los datos de entrenamiento,
    los datos de validación y los datos de prueba.

    El "training_data" se devuelve como una tupla con dos entradas.
    La primera entrada contiene las imágenes de entrenamiento reales. Esta es una
    matriz numpy con 50,000 entradas. Cada entrada es, a su vez, una matriz numpy con
    784 valores, que representan los 28 * 28 = 784 píxeles en una sola imagen MNIST.

    La segunda entrada en la tupla "training_data" es una matriz numpy que contiene
    50,000 entradas. Esas entradas son simplemente los valores de los dígitos (0...9)
    correspondientes a las imágenes contenidas en la primera entrada de la tupla.

    Los datos de "validation_data" y "test_data" son similares, excepto que cada uno
    contiene solo 10,000 imágenes.

    Este es un formato de datos agradable, pero para su uso en redes neuronales, es útil
    modificar un poco el formato de "training_data". Eso se hace en la función de
    envoltura "load_data_wrapper()", consulta a continuación.
    """
    with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Devuelve una tupla que contiene ``(training_data, validation_data, test_data)``.
    Basado en "load_data", pero el formato es más conveniente para su uso en nuestra
    implementación de redes neuronales.

    En particular, "training_data" es una lista que contiene 50,000 pares (x, y).
    "x" es un numpy.ndarray de 784 dimensiones que contiene la imagen de entrada. "y" es
    un numpy.ndarray de 10 dimensiones que representa el vector unitario correspondiente
    al dígito correcto para "x".

    "validation_data" y "test_data" son listas que contienen 10,000 pares (x, y) cada una.
    En cada caso, "x" es un numpy.ndarray de 784 dimensiones que contiene la imagen de
    entrada, y "y" es la clasificación correspondiente, es decir, los valores de los
    dígitos (números enteros) correspondientes a "x".

    Obviamente, esto significa que estamos utilizando formatos ligeramente diferentes
    para los datos de entrenamiento y los datos de validación/prueba. Estos formatos
    resultan ser los más convenientes para su uso en nuestro código de redes neuronales.
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Devuelve un vector unitario de 10 dimensiones con un valor de 1.0 en la posición j-ésima
    y ceros en las demás posiciones. Esto se utiliza para convertir un dígito (0...9) en una
    salida deseada correspondiente de la red neuronal.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
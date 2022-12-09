# Neural Network
## Description
NeuralNetwork is a library understanding basic principles 
of artificial intelligence (AI) and an easy implementation of new training algorithms. 

## Usage
In this example we're going to use NeuralNetwork library for training a neural network (nn) to predict a XOR output.

```python 3.10
import NeuralNetwork as nn
import numpy as np

def random_points(n = 100):
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)

    return np.array([x, y]).T

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])

Y = np.array([
    [0],
    [1],
    [1],
    [0],
])

#
red_xor = nn.crear_modelo_de_red([2,3, 1])
#
funcion_de_activacion = nn.sigmoide
#
red_entrenada = nn.entrenar_red_neuronal(red_xor, 
                                        funcion_de_activacion, 
                                        nn.mean_squared_error,
                                        valor_de_prediccion=X,
                                        valor_real=Y, 
                                        epochs=5000, 
                                        tasa_de_aprendizaje=5.0)

x_test = random_points(n = 50000)
y_test = nn.predecir(x_test, red_entrenada, funcion_de_activacion)

plt.scatter(x_test[:,0], x_test[:,1], c = y_test, s = 25, cmap='GnBu')
plt.savefig('XOR_Fitted.jpg')
```

# Note
Please check Experimentos.ipynb file to learn more examples.


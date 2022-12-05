import numpy as np
from tqdm import tqdm
from numba import jit

# NOTA: Para el algorimo de backpropagation se necesitan las derivadas
# de las funciones de activación y una función de coste.

# __________ Funciones de activación __________ #

# Definimos la función sigmoide que es la función de activación. 

def sigmoide(x, derivada=False) -> np.float64:
    """Tiene como entrada el dato a evaluar."""
    # Si queremos que se retorne la derivada de la función sigmoide 
    # el parámetro derivada = True
    if derivada == True: 
        return x * (1 - x) # Derivada simple de la función sigmoide.
    
    elif derivada == False:
        return 1/(1 + np.exp(-x)) # Función de activación sin derivar.


def tan_h(x, derivada=False):
    if derivada == True:
        return (1 - (np.tanh(x))**2)
    
    elif derivada == False:
        return np.tanh(x)


# __________ Funciones de coste __________ #

def mean_squared_error(valor_de_prediccion, valor_real, derivada=False):
    """Esta función calcula el error cuadrático medio"""
    if derivada == True:
        # f'(x) = x2 - x1
        return valor_de_prediccion - valor_real
    
    elif derivada == False: 
        # f(x) = ((x2 - x1)**2)/2
        return np.mean((valor_de_prediccion - valor_real)**2)


# __________ Herramientas para las redes neuronales __________ #

# Definimos un creador de capas (layers).
def capa_de_neuronas(conexiones, numero_neuronas) -> np.array:
    """
    Entradas: Conexiones y Número de Neuronas por capa
        Dado un input de número de conexiones y número de neuronas de la capa
        esta función devuelve una variable bias que corresponde a un arreglo
        de números aleatorios uniformemente distribuidos entre (-1, 1) y un arreglo
        de pesos con las mismas características con el nombre de variable weights.
        
        El orden de retorno es: 
        >>> weights, bias
        
        bias = número de neuronas
        weights = conexiones * neuronas
    """
    # Para generar pesos en un rango entre (-1, 1) multiplicamos el arreglo por 2 y restamos 1.
    weights = np.random.rand(conexiones, numero_neuronas) * 2 - 1
    # Para generar un bias en un rango entre (-1, 1) multiplicamos el arreglo por 2 y restamos 1.
    bias = np.random.rand(1, numero_neuronas) * 2 -1 
    
    return weights, bias


# Creamos el modelo de la red basado en la topología que deseemos. 
def crear_modelo_de_red(topologia=[]) -> list:
    """
    Entrada: topología

        La topología hace referencia a la arquitectura de la red neuronal donde se tiene en cuenta:
        - numero entradas
        - numero de neuronas por capa escondida
        - neuronas de salida
        
        Ejemplo:
        
        crear_modelo_de_red(topologia=[3, 6, 6, 3])
        
        Lo que implica que la topología de esta red neuronal tiene 3 entradas,
        dos capas escondidas de 6 neuronas cada una y una capa de salida de 3 neuronas.
    
        Dada la topología crea un modelo de red neuronal, se debe prestar atención a las dimensiones
        de cada capa de la red para poder realizar el producto matricial. 
        
        Retorna una lista: 
        >>> red_neuronal
    
    """
    
    red_neuronal = []
    
    for indice_capa in range(len(topologia[:-1])): # Usando el [:-1] omite la última capa para evitar errores
        
        # La capa actual es equivalente al número de conexiones de la capa anterior 
        # y el número de neuronas de la capa actual para asegurarnos de que coincida 
        # el producto matricial (e.g. multiplicar una matriz de 3x2 por una de 2x3).
        
        conexiones_capa_anterior = topologia[indice_capa]
        numero_de_neuronas_capa_actual = topologia[indice_capa + 1]
        
        # La capa de neuronas requiere dos entradas: conexiones y número de neuronas
        red_neuronal.append(capa_de_neuronas(conexiones_capa_anterior, numero_de_neuronas_capa_actual)) 
        
    return red_neuronal


# Una vez entrenada la red realiza una predicción. 
def predecir(entradas, red_neuronal_entrenada, funcion_de_activacion=sigmoide) -> np.array:
    
    """ 
        Una vez entrenada la red, esta función permite realizar feed propagation
        lo que permite realizar predicciones. 
    """
    # La primera capa de la red cuenta como la primera salida de la primera capa.
    salida_por_capa = np.array(entradas) 
    
    for indice_capa in range(len(red_neuronal_entrenada)):
        
        # X es el dato a evaluar en la función de activación.
        # Este es calculado por el producto matricial de la salida de la capa anterior
        # por los pesos de la capa actual y sumando el bias. 
        # outcome * weights + bias.
        
        # Recordar que la salida de cada capa de neuronas es:
        # >>> weights, bias
        # Por lo que el índice 0 son los pesos y el índice 1 es el bias. 
        # Estos al ser arreglos de numpy permiten el método np.array().dot(np.array())
        # Para realizar el producto punto.
        
        pesos = red_neuronal_entrenada[indice_capa][0]
        bias = red_neuronal_entrenada[indice_capa][1]
        
        # Se realiza el cálculo de X:
        calculo_de_x = salida_por_capa.dot(pesos) + bias
        
        # Evalúa la función de activación en X
        salida_por_capa = funcion_de_activacion(calculo_de_x) 
    
    return salida_por_capa


# __________ Entrenamiento de la red neuronal __________ #
def entrenar_red_neuronal(red_neuronal_sin_entrenar, funcion_de_activacion=sigmoide, funcion_de_coste=mean_squared_error, valor_de_prediccion = [], valor_real = [], epochs = 100, tasa_de_aprendizaje = 0.1):
    """
    Entradas:
        Valor de prediccion = 
        Valor Real = 
        Epochs = Número de iteraciones por entrenamiento.
        Tasa de aprendizaje = Que tan rápido quiero que aprenda mi red. 
    """
    
    for epoch in tqdm(range(epochs)):
        
        # Salida sin activar, Salida activada
        salida_por_capa = [(None, valor_de_prediccion)]
        
        for indice_capa in range(len(red_neuronal_sin_entrenar)):
            
            pesos = red_neuronal_sin_entrenar[indice_capa][0]
            bias  = red_neuronal_sin_entrenar[indice_capa][1]
            
            salida_sin_activar = salida_por_capa[-1][1].dot(pesos) + bias
            salida_activada = funcion_de_activacion(salida_sin_activar)
            
            salida_por_capa.append((salida_sin_activar, salida_activada)) # indice 0 | indice 1
        
        # Este arreglo guarda el tamaño del paso para 
        # la optimización por descenso del gradiente
        arreglo_de_deltas = [] 
        
        # La función reversed() invierte los valores de la lista 
        # que genera range() para poder hacer el back propagation.
        for indice_reverso_de_capa in reversed(range(len(red_neuronal_sin_entrenar))): 
            
            # Tener en cuenta los valores de la variable salida por capa
            salida_sin_activar = salida_por_capa[indice_reverso_de_capa + 1][0] 
            salida_activada    = salida_por_capa[indice_reverso_de_capa + 1][1] # Recordar que este dato es una matriz
            
            # Debido a que el arreglo de deltas no tiene nada, se agrega algo para poder 
            # iterar sin problemas.
            
            if indice_reverso_de_capa ==  len(red_neuronal_sin_entrenar) - 1:
                
                # Se multiplica el error cuadratico medio con la derivada 
                # de la función de activación evaluada en la salida activada.
                #
                # Se usa insert() dado que al reversar los índices esta función revierte
                # esa inversión generando una lista ordenada. 
                arreglo_de_deltas.insert(0, (funcion_de_coste(salida_activada, valor_real, derivada=True) * funcion_de_activacion(salida_activada, derivada=True)))
            
            else:
                # pesos_retropropagacion = red_neuronal_sin_entrenar[indice_reverso_de_capa][0]
                arreglo_de_deltas.insert(0, arreglo_de_deltas[0].dot(pesos_retropropagacion.T) * funcion_de_activacion(salida_activada, derivada=True))
                
            pesos_retropropagacion = red_neuronal_sin_entrenar[indice_reverso_de_capa][0]    
            
            
            # pesos_actualizados = list(red_neuronal_sin_entrenar[indice_reverso_de_capa][0])
            # bias_actualizados = list(red_neuronal_sin_entrenar[indice_reverso_de_capa][1])
            
            pesos_actualizados = red_neuronal_sin_entrenar[indice_reverso_de_capa][0]  - salida_por_capa[indice_reverso_de_capa][1].T.dot(arreglo_de_deltas[0]) * tasa_de_aprendizaje
            bias_actualizados = red_neuronal_sin_entrenar[indice_reverso_de_capa][1]  - np.mean(arreglo_de_deltas[0], axis = 0, keepdims = True) * tasa_de_aprendizaje 

            red_neuronal_sin_entrenar[indice_reverso_de_capa] = tuple([pesos_actualizados, bias_actualizados])
            
        print(f'Epoch: {epoch}      |       Error: {mean_squared_error(salida_activada, valor_real)}')        
    return red_neuronal_sin_entrenar


# __________ Herramientas para leer datos __________ # 
def cargar_datos():
    return

if __name__ == '__main__':
    

    def random_points(n = 100):
        x = np.random.uniform(0.0, 1.0, n)
        y = np.random.uniform(0.0, 1.0, n)

        return np.array([x, y]).T


    red_xor = crear_modelo_de_red([2, 4, 1])

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
    # red_neuronal_sin_entrenar, funcion_de_activacion, valor_de_prediccion = [], valor_real = [], epochs = 100, tasa_de_aprendizaje = 0.1
    red_entrenada = entrenar_red_neuronal(red_xor, tan_h, mean_squared_error,valor_de_prediccion=X, valor_real=Y, epochs=10000, tasa_de_aprendizaje=0.08)
    
    x_test = random_points(n = 5000)
    # entradas, red_neuronal_entrenada, funcion_de_activacion
    y_test = predecir(x_test, red_entrenada, tan_h)
    # plt.scatter(x_test[:,0], x_test[:,1], c = y_test, s = 25, cmap='GnBu')
    # plt.savefig('XOR_Fitted.jpg')
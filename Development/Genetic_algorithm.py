import numpy as np
from NeuralNetwork import crear_modelo_de_red, obtener_caracteristicas_de_la_red

def mat_to_vector(mat_pop_weights):
     pop_weights_vector = []
     for sol_idx in range(mat_pop_weights.shape[0]):
         curr_vector = []
         for layer_idx in range(mat_pop_weights.shape[1]):
             vector_weights = np.reshape(mat_pop_weights[sol_idx, layer_idx], newshape=(mat_pop_weights[sol_idx, layer_idx].size))
             print(vector_weights)
             exit()
             curr_vector.extend(vector_weights)
         pop_weights_vector.append(curr_vector)
     return np.array(pop_weights_vector)
 
if __name__ == '__main__':
    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1],
    ])
    red_neuronal = crear_modelo_de_red([2, 3, 1])
    pesos_por_capa = obtener_caracteristicas_de_la_red(red_neuronal, 'activacion', X)
    print(pesos_por_capa)
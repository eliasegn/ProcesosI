################################################################################
# Matriz de Transición y Distribución Inicial
# Autor: Elías González Nieto
# Afil : Facultad de Ciencias - UNAM
# Curso : Procesos Estocásticos
################################################################################

################################################################################
# Librerías
import numpy as np 
import random
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from numpy.linalg import matrix_power
import scipy as sp
################################################################################

################################################################################
# Matrices de Transición
# La gráfica de la caminata aleatoria del borrachito
G = nx.Graph() 
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
# Posiciones de los vértices
pos = {
    0: (0, 0),
    1: (0.5, 0),
    2: (0.5, 0.5),
    3: (0, 0.5)
}

# Graficamos
plt.figure(figsize = (3,3))
nx.draw(G, pos, with_labels=True)
plt.show()

# Potencias de la matriz P
# Definimos variables simbólicas
p = sp.Symbol('p')
q = sp.Symbol('q')

# Definimos la matriz P
P = np.array([
    [0, p, 0, q],
    [q, 0, p, 0],
    [0, q, 0, p],
    [p, 0, q, 0]
])

# Imprimimos su cuadrado y su cubo
print('P^2 =', matrix_power(P,2), '\n\n P^3 =',matrix_power(P, 3))

################################################################################
# Primer ejemplo usando matriz de transición
# Definimos la matriz de transición
P0 = np.array([
    [0, 0, 0.9, 0.1],
    [0.1, 0.05, 0.8, 0.05],
    [0.025, 0.025, 0.9, 0.05],
    [0.1, 0.1, 0.7, 0.1]
])

# Definimos parámetros para guardar nuestro proceso
proceso0 = [random.randint(0,3)] # randint devuelve un entero aleatorio en el intervalo [a,b]
iter = 200 # número de iteraciones del proceso

# Simulamos nuestro proceso
for i in range(iter):
  estado_actual = proceso0[-1]
  xn = np.random.choice([0,1,2,3], p=P0[estado_actual])
  proceso0.append(int(xn))

# Imprimimos el proceso
print(proceso0)

# Graficamos el proceso
plt.figure()
plt.plot(proceso0, 'o', color = 'indigo')
plt.style.use('seaborn-v0_8-dark')
plt.grid()
plt.title('Cadena de Markov')
plt.xlabel('Iteración')
plt.ylabel('Estado')
plt.show()

################################################################################
# Clase de Cadena de Markov
class CadenaMarkov:

  # Definimos el constructor de la clase
  def __init__(self, P, pi_0):
    '''
    P : matriz de transición
    pi_0 : distribución inicial
    '''
    self.P = P
    self.pi_0 = pi_0
    self.proceso = [np.random.choice(range(len(pi_0)), p = pi_0)]

  # Simulamos la cadena igual que antes
  def simular(self, iter):
    for i in range(iter):
      estado_actual = self.proceso[-1]
      xn = np.random.choice(range(len(self.pi_0)), p = self.P[estado_actual])
      self.proceso.append(int(xn))
    return self.proceso

  # Graficamos la trayectoria generada de nuestro proceso
  def plot(self, co): # co es el color
    plt.figure(figsize=(7,4))
    plt.plot(self.proceso, 'o', color = co, ms=1)
    plt.style.use('seaborn-v0_8-dark')
    plt.grid()
    plt.title('Cadena de Markov')
    plt.xlabel('Iteración')
    plt.ylabel('Estado')
    plt.show()

# Generador aleatoria de distribuciones iniciales de n entradas
def generador_iniciales(n):
  pi_0 = np.random.rand(n) # Crea un array de n valores uniformes(0,1)
  pi_0 = pi_0/sum(pi_0) # Normaliza el vector
  return pi_0

# Ejemplo
P1 = np.array([
    [0.0, 0.2, 0.1, 0.7],
    [0.1, 0.4, 0.3, 0.2],
    [0.8, 0.2, 0.0, 0.0],
    [0.1, 0.1, 0.7, 0.1]
])

################################################################################
# Distribución Inicial
pi0 = generador_iniciales(4)
cadena1 = CadenaMarkov(P1, pi0)
cadena1.simular(200)
cadena1.plot('deeppink')

# Definimos la matriz de transición y ambas distribuciones iniciales
P2 = [[1,0],
      [0,1]]
pi1 = [1,0]
pi2 = [0,1]
iter = 200

# Graficamos y comparamos
cadena3 = CadenaMarkov(P2, pi1)
cadena4 = CadenaMarkov(P2, pi2)
cadena3.simular(iter)
cadena4.simular(iter)
cadena3.plot('deeppink')
cadena4.plot('olive')

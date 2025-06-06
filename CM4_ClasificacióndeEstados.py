################################################################################
# Clasificación de Estados
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
from numpy.linalg import matrix_power
################################################################################

################################################################################
# Estados Absorbentes
P_absorbente = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.3, 0.4, 0.3, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.1, 0.3, 0.0, 0.6],
])

estados = [0, 1, 2, 3]

# Verificamos estados absorbentes: P[i, i] == 1
absorbentes = [i for i in range(len(P_absorbente)) if P_absorbente[i, i] == 1]
print("Estados absorbentes:", [estados[i] for i in absorbentes])

################################################################################
# Evolución de Distribuciones
# Definimos la matriz
P_periodica = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
])

# Definimos una distribución inicial y vamos guardando cada distribución
dist = np.array([1.0, 0.0])  # Distribución inicial \pi_0
hist = [dist]  # Hacemos una lista de distribuciones

# Iteramos las 10 primeras distribuciones
for _ in range(10):
    dist = dist @ P_periodica
    hist.append(dist)

# Graficamos ambas distribuciones
hist = np.array(hist)
plt.figure(figsize=(7,4))
plt.style.use('seaborn-v0_8-dark')
plt.plot(hist[:, 0], label="Estado 0", color = 'indigo')
plt.plot(hist[:, 1], label="Estado 1", color = 'deeppink')
plt.title("Oscilación de probabilidades (periodicidad 2)")
plt.xlabel("Tiempo")
plt.ylabel("Probabilidad")
plt.legend()
plt.show()

################################################################################
# Tiempo Medio de Retorno
# Se edita la clase CadenaMarkov para agregar el Tiempo Medio de Retorno
class CadenaMarkov:

  # Definimos el constructor de la clase
  def __init__(self, P, pi_0):
    '''
    P : matriz de transición
    pi_0 : distribución inicial
    estados : espacio de estados
    proceso : trayectoria del proceso
    '''
    self.P = P
    self.pi_0 = pi_0
    self.estados = list(range(len(pi_0))) # Nuevo: Definimos el espacio de estados
    self.proceso = [np.random.choice(range(len(pi_0)), p = pi_0)]

  # Simulamos la cadena igual que antes
  def simular(self, iter):
    for i in range(iter):
      estado_actual = self.proceso[-1]
      xn = np.random.choice(range(len(self.pi_0)), p = self.P[estado_actual])
      self.proceso.append(int(xn))
    return self.proceso

  # Graficamos la trayectoria generada de nuestro proceso
  def plot(self, co):
    plt.figure(figsize=(7,4))
    plt.plot(self.proceso, '-', color = co, ms=1)
    plt.style.use('seaborn-v0_8-dark')
    plt.grid()
    plt.title('Cadena de Markov')
    plt.xlabel('Iteración')
    plt.ylabel('Estado')
    plt.show()

  # Función para calcular m_y, recibe iteraciones de cada trayectoria y número de trayectorias
  def Tiempos_retorno(self, y, iter, trayectorias):
    # Verificamos que el estado esté en el conjunto de estados
    if not y in self.estados:
      return None
    else:
      muestra = []
      # Generamos tantas trayectorias como indiquemos
      for _ in range(trayectorias):
        # Generamos una trayectoria que empiece en el estado que queremos
        cadena_y = CadenaMarkov(self.P, [1 if i == y else 0 for i in range(len(self.pi_0))])
        # Simulamos la trayectoria
        cadena_y.simular(iter)
        # Verificamos el tiempo de retorno de esa trayectoria y lo añadimos a la lista muestra
        for t in range(1, len(cadena_y.proceso)):
          if cadena_y.proceso[t] == y:
            muestra.append(t)
            break
      # Si todo sale bien, devolvemos la media muestral de la muestra de T_y
      if muestra:
        return sum(muestra) / len(muestra)
      else:
        return None

# Ejemplo
P1 = np.array([[0, 0.5, 0.5],
               [0.25, 0.5, 0.25],
               [.25, 0.25, 0.5]])

cadena_prueba = CadenaMarkov(P1, [1,0,0])
print('El tiempo medio de retorno a 0 es:', cadena_prueba.Tiempos_retorno(0, 1000, 1000), '\n a 1 es:', cadena_prueba.Tiempos_retorno(1, 1000, 1000),
      '\n a 2 es:', cadena_prueba.Tiempos_retorno(2, 1000, 1000))
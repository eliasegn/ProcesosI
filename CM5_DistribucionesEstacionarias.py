################################################################################
# Distribuciones Estacionarias
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
# Primer ejemplo
P0 = np.array([
    [0, 0, 0.9, 0.1],
    [0.1, 0.05, 0.8, 0.05],
    [0.025, 0.025, 0.9, 0.05],
    [0.1, 0.1, 0.7, 0.1]
])

pi_0 = np.array([1,0,0,0]) # Suponemos esta dist inicial

N = 10
distribuciones = []

for n in range(N):
    pi_n = np.dot(pi_0, P0) # Evolución de distribuciones
    distribuciones.append(pi_n)
    pi_0 = pi_n

distribuciones # Es una lista de listas

# Elegimos como inicial a la que se aproximaba antes
pi_00 = np.array([0.03048298, 0.02903141, 0.88624967, 0.05423595])
N = 10
distribuciones2 = []

for n in range(N):
    pi_n = np.dot(pi_00, P0) # Evolución de distribuciones
    distribuciones2.append(pi_n)
    pi_00 = pi_n

distribuciones2

################################################################################
# Segundo ejemplo
P = np.array([[2/3, 1/3], [1/4, 3/4]])
pi_01 = np.array([3/7, 4/7])
N = 10
distribuciones3 = []

for n in range(N):
    pi_n = np.dot(pi_01, P)
    distribuciones3.append(pi_n)
    pi_0 = pi_n

distribuciones3

################################################################################
# Clase CadenaMarkov
# Editamos la clase CadenaMarkov para que incluya el cálculo a largo plazo de las distribuciones
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

  def distribuciones(self, iter):
    # Definimos nuestra lista de distribuciones
    distribuciones = []
    pi_0 = self.pi_0
    # Iteramos como ya sabemos y guardamos las distribuciones
    for n in range(iter):
        pi_n = np.dot(pi_0, self.P)
        distribuciones.append(pi_n)
        pi_0 = pi_n
    return distribuciones

################################################################################
# Ejemplo 1
P1 = np.array([[0, 0.5, 0.5],
               [0.25, 0.5, 0.25],
               [.25, 0.25, 0.5]])

cadena_prueba = CadenaMarkov(P1, [1,0,0])
cadena_prueba.distribuciones(20)

# Histograma de la 20ava distribución
plt.figure(figsize=(7,4))
plt.style.use('seaborn-v0_8-dark')
plt.bar([0,1,2], cadena_prueba.distribuciones(20)[-1], color = 'indigo')
plt.grid()
plt.title('20ava Distribución')
plt.xlabel('Estado')
plt.ylabel('Probabilidad')
plt.show()

# Ejemplo 2
P = np.array([[2/3, 1/3], [1/4, 3/4]])
pi_0 = np.array([3/7, 4/7])

# Imprimos tiempos medios de retorno, recordando el Teorema Ergódico para CM
cadena_prueba2 = CadenaMarkov(P, pi_0)
print('El tiempo de retorno a 0 es:', cadena_prueba2.Tiempos_retorno(0, 1000, 1000), '\n a 1 es: ', cadena_prueba2.Tiempos_retorno(1, 1000, 1000))

# Vemos que tiene infinitas distribuciones estacionarias
cadena_prueba4 = CadenaMarkov(P2, [3/4, 0, 1/4])
cadena_prueba4.distribuciones(10)

################################################################################
# Clase CadenaMarkov con cálculo de distribución estacionaria
class CadenaMarkov:

  # Definimos el constructor de la clase
  def __init__(self, P, pi_0):
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

  def distribuciones(self, iter):
    # Definimos nuestra lista de distribuciones
    distribuciones = []
    pi_0 = self.pi_0
    # Iteramos como ya sabemos y guardamos las distribuciones
    for n in range(iter):
        pi_n = np.dot(pi_0, self.P)
        distribuciones.append(pi_n)
        pi_0 = pi_n
    return distribuciones

  def distribucion_estacionaria(self):
    # Por cada estado, calculamos el recíproco de su tiempo medio de recurrencia por el teorema ergódico para CM
    cadena_estado = CadenaMarkov(self.P, self.pi_0)
    est = [1/cadena_estado.Tiempos_retorno(j, 1000, 1000) for j in self.estados]
    return est
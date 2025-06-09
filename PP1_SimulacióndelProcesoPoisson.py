################################################################################
# Simulación del Proceso Poisson
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
# Graficos de procesos escalonados
# Tenemos una lista sencilla de tiempos de salto
import numpy as np
import matplotlib.pyplot as plt

llegadas = np.linspace(0, 10, 11) # Saltos
conteo = np.arange(1, len(llegadas) + 1) # Valor de la función

plt.figure(figsize=(7, 4))

# Dibujamos las líneas para la trayectoria
for i in range(len(llegadas) - 1):
    # Línea horizontal
    plt.plot([llegadas[i], llegadas[i+1]], [conteo[i], conteo[i]], color='navy')

    # Línea vertical punteada
    plt.plot([llegadas[i+1], llegadas[i+1]], [conteo[i], conteo[i+1]],  # Mantenemos fija a la x y movemos y
             color='navy', linestyle='--')

# Puntos para indicar el salto
plt.scatter(llegadas, conteo, color='indigo', zorder=3)

plt.title('Función Escalonada')
plt.grid()
plt.show()

################################################################################
# Simulación por tiempos número de ocurrencias
# Definimos el tiempo t y el número de saltos n
t = 50
n = 10

# Generamos n uniformes en el intervalo (0,t) y las ordenamos
uniformes = np.random.uniform(0, t, n)
uniformes.sort()

# Creamos un data.frame para los tiempos de recurrencia
pp1 = pd.DataFrame({'n':range(n+1), 'T_n':np.concatenate([[0],uniformes])}) # Agregamos 0 con 0

# Graficamos con las funciones step y scatter
plt.figure(figsize=(7,4))
plt.style.use('seaborn-v0_8-dark')
# Algo análogo a lo anterior
for i in range(len(pp1['T_n']) - 1):
    # Línea horizontal
    plt.plot([pp1['T_n'][i], pp1['T_n'][i+1]], [pp1['n'][i], pp1['n'][i]], color='navy')

    # Línea vertical punteada
    plt.plot([pp1['T_n'][i+1], pp1['T_n'][i+1]], [pp1['n'][i], pp1['n'][i+1]],  # Mantenemos fija a la x y movemos y
             color='navy', linestyle='--')
plt.scatter(pp1['T_n'], pp1['n'], color='indigo', zorder=3)
plt.plot()
plt.title('Proceso Poisson simulado')
plt.xlabel('Tiempo $t$')
plt.ylabel('Número de eventos $N(t)$')
plt.xlim(-1, t)
plt.ylim(-0.5,n+1)
plt.grid()
plt.show()

# Clase de PP por tiempo de ocurrencias
class ProcesoPoisson_Ocurrencias:
  def __init__(self, t, n):
    '''
    t : La cota que define el dominio del proceso
    n : El valor del proceso al tiempo t
    '''
    self.t = t
    self.n = n
    self.saltos = pd.DataFrame()

  # Simulamos como antes y lo agregamos a un data frame
  def simular(self):
    uniformes = np.random.uniform(0, self.t, self.n)
    uniformes.sort()
    self.saltos = pd.DataFrame({'n':range(self.n+1), 'T_n':np.concatenate([[0],uniformes])})
    return self.saltos

  # Graficamos igual que antes
  def plot(self):
    plt.figure(figsize=(7,4))
    plt.style.use('seaborn-v0_8-dark')
    for i in range(len(self.saltos['T_n']) - 1):
        # Línea horizontal
        plt.plot([self.saltos['T_n'][i], self.saltos['T_n'][i+1]], [self.saltos['n'][i], self.saltos['n'][i]], color='navy')

        # Línea vertical punteada
        plt.plot([self.saltos['T_n'][i+1], self.saltos['T_n'][i+1]], [self.saltos['n'][i], self.saltos['n'][i+1]],  # Mantenemos fija a la x y movemos y
                color='navy', linestyle='--')
    plt.scatter(self.saltos['T_n'], self.saltos['n'], color='indigo', zorder=3)
    plt.scatter(self.saltos['T_n'], self.saltos['n'], color='blue', zorder=1, s=15)
    plt.title('Proceso Poisson simulado')
    plt.xlabel('Tiempo $t$')
    plt.ylabel('Número de eventos $N(t)$')
    plt.grid()
    plt.show()

# Ejemplo1
ppt1n5 = ProcesoPoisson_Ocurrencias(1, 5)
ppt1n5.simular()
ppt1n5.plot()

# Ejemplo2
ppt10n50 = ProcesoPoisson_Ocurrencias(10, 50)
ppt10n50.simular()
ppt10n50.plot()

################################################################################
# Simulación por tiempo de interocurrencia
# Definimos el tiempo hasta el cual queremos nuestro proceso
t = 20

# Definimos la tasa del proceso
lamb = 1

# Definimos una lista de llegadas y la suma
llegadas = [0]

# Mientras la suma sea menor que t, seguimos generando exponenciales
while llegadas[-1] < t:
  ti = np.random.exponential(1/lamb)
  llegadas.append(llegadas[-1] + ti)

# N será el número de llegadas
N = len(llegadas)

# Imprimimos el número de llegadas y las llegadas
print(N, llegadas)

# Graficamos nuestro proceso
plt.figure(figsize=(7,4))
plt.style.use('seaborn-v0_8-dark')
# Algo análogo a lo anterior
niveles = list(range(len(llegadas) - 1))
for i in range(len(llegadas)-1):
    # Línea horizontal
    plt.plot([llegadas[i], llegadas[i+1]], [niveles[i], niveles[i]], color='navy')

    # Línea vertical punteada
    if i < len(niveles) - 1:
        plt.plot([llegadas[i+1], llegadas[i+1]], [niveles[i], niveles[i+1]], color='navy', linestyle='--')

plt.scatter(llegadas[:-1], niveles, color='indigo', zorder=3)
plt.title(f'Proceso Poisson de tasa {lamb}')
plt.xlabel('Tiempo $t$')
plt.ylabel('Número de eventos $N(t)$')
plt.ylim(0,N+1)
plt.grid()

# Clase para PP por tiempos de interocurrencia
class ProcesoPoisson:
  # Recibe la tasa lambda
  def __init__(self, lamb):
    '''
    lamb : tasa lambda del PP
    '''
    self.lamb = lamb
    self.proceso = pd.DataFrame()

  # Recibe un parámetro t y simulamos hasta ese tiempo
  def simular(self, t):
    llegadas = [0]
    # Mientras la suma no exceda t, genera una exponencial y la agrega a la lista
    while llegadas[-1] < t:
      ti = np.random.exponential(1/self.lamb)
      llegadas.append(llegadas[-1] + ti)
    # N es tal que N_t = N
    N = len(llegadas)
    self.proceso = pd.DataFrame({'n':range(N), 'T_n':llegadas})
    print(f'El proceso tiene {N} salto al tiempo {t}')
    return self.proceso

  # Graficamos el proceso igual que antes
  def plot(self):
    plt.figure(figsize=(7,4))
    for i in range(len(self.proceso['T_n'])-1):
        # Línea horizontal
        plt.plot([self.proceso['T_n'][i], self.proceso['T_n'][i+1]], [self.proceso['n'][i], self.proceso['n'][i]], color='navy')

        # Línea vertical punteada (si no es la última)
        if i < len(self.proceso['n']) - 1:
            plt.plot([self.proceso['T_n'][i+1], self.proceso['T_n'][i+1]], [self.proceso['n'][i], self.proceso['n'][i+1]], color='navy', linestyle='--')

    plt.scatter(self.proceso['T_n'], self.proceso['n'], color='indigo', zorder=3)
    plt.title(f'Proceso Poisson de tasa {self.lamb}')
    plt.xlabel('Tiempo $t$')
    plt.ylabel('Número de eventos $N(t)$')
    plt.grid()

# Ejemplo3
ppprueba = ProcesoPoisson(0.5)
ppprueba.simular(10)
ppprueba.plot()

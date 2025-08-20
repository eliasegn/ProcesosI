################################################################################
# Movimiento Browniano Geométrico
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
# Simulación del MBG

def MBG(S0, mu, sigma, T, N):
  '''
  S0: Precio inicial del activo
  mu: Tasa de retorno esperada
  sigma: Volatilidad del activo
  T: Tiempo total de simulación (en años)
  N: Número de pasos en la simulación
  '''
  dt = T / N # Tamaño del paso
  t = np.linspace(0, T, N + 1) # Vector de tiempo

  # Generamos el MBE
  W = np.random.normal(loc=0, scale=np.sqrt(dt), size=N)
  W = np.concatenate(([0], np.cumsum(W)))

  # Aplicamos método de Euler
  S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)
  return S

# Ejemplo de una trayectoria
# Definimos los parámetros del proceso
S_0 = 1
mu = 0.5
sigma = 0.2
T = 1
N = 1000

# Simulamos la trayectoria
S = MBG(S_0, mu, sigma, T, N)

# Graficamos la trayectoria
plt.figure(figsize=(10, 5))
plt.style.use('seaborn-v0_8-dark')
plt.plot(S, color = 'indigo', label = 'MBG')
plt.title(f'Movimiento Browniano Geométricocon $\mu = {mu}$, $\sigma = {sigma}$ y $S_0 = {S_0}$')
plt.legend()
plt.xlabel('Tiempo')
plt.ylabel('Precio del activo')
plt.grid()
plt.show()

# Función para graficar la trayectoria
def plot_trayectoria_MBG(S_0, mu, sigma, T, N):
  '''
  S0: Precio inicial del activo
  mu: Tasa de retorno esperada
  sigma: Volatilidad del activo
  T: Tiempo total de simulación (en años)
  N: Número de pasos en la simulación
  '''
  # Simulamos la trayectoria
  S = MBG(S_0, mu, sigma, T, N)
  # Graficamos la trayectoria
  plt.figure(figsize=(10, 5))
  plt.style.use('seaborn-v0_8-dark')
  plt.plot(S, color = 'indigo')
  plt.title(f'Movimiento Browniano Geométrico con $\mu = {mu}$, $\sigma = {sigma}$ y $S_0 = {S_0}$')
  plt.xlabel('Tiempo')
  plt.ylabel('Precio del activo')
  plt.grid()
  plt.show()
  return None

################################################################################
# Estimación Paramétrica

# Ejemplo 1
np.random.seed(4)
S_02 = 30
mu2 = 1
sigma2 = 0.1
T2 = 1
N2 = 10000
dt = T2 / N2

# Simulamos la trayectoria
S2 = MBG(S_02, mu2, sigma2, T2, N2)
Y2 = np.diff(np.log(S2))
n = len(Y2)

# Calculamos estimadores de mu y sigma
sigma_est2 = np.sqrt( (1 / (dt * n)) * np.sum((Y2 - np.mean(Y2)) ** 2) )
mu_est2 = (np.mean(Y2) / dt) + (0.5 * sigma_est2**2)

# Imprimimos los valores reales junto con los estimadores
print('Valor de mu:', mu2, '\nEstimador de mu:', mu_est2)
print('Valor de sigma:', sigma2, '\nEstimador de sigma:', sigma_est2)

# Segunda Parte: Cálculo para 1000 trayectorias
trayectorias = 1000

# Inicializamos listas donde guardaremos estimadores por trayectoria
mu_tray = np.zeros(trayectorias)
sigma_tray = np.zeros(trayectorias)

# Generamos trayectorias, calculamos los estimadores y los guardamos
for i in range(trayectorias):
  S_tray = MBG(S_02, mu_est2, sigma_est2, T2, N2)
  Y_tray = np.diff(np.log(S_tray))
  sigma_tray[i] = np.sqrt( (1 / (dt * n)) * np.sum((Y_tray - np.mean(Y_tray)) ** 2) )
  mu_tray[i] = (np.mean(Y_tray) / dt) + (sigma_tray[i]**2 / 2)

# Graficamos la evolución de los estimadores
plt.figure()
plt.style.use('seaborn-v0_8-dark')
plt.plot(mu_tray, color = 'green')
plt.plot(sigma_tray, color = 'red')
plt.axhline(y=mu2, color='lime', linestyle='--')
plt.axhline(y=sigma2, color='darkred', linestyle='--')
plt.legend(['$\hat{\mu}$', '$\hat{\sigma}$', '$\mu$', '$\sigma$'])
plt.grid()
plt.show()

# Imprimimos la media de los estimadores
print('Media de mu:', np.mean(mu_tray))
print('Media de sigma:', np.mean(sigma_tray))

################################
# Ajuste de una base de datos

# Importamos los datos
import pandas as pd
data = pd.read_csv('MB2_AAPL.txt', sep=',')
data = data.drop(index=0)
data.head()

# Recortamos a solo el cierre durante un año
data = data['Close']
data = data[:252]
data.head()

# Convertimos los datos a flotantes y redondeamos
data = np.array(data)
data = data.astype(float)
data = np.round(data, 2)

# Graficamos los datos
plt.figure(figsize=(10,5))
plt.style.use('seaborn-v0_8-dark')
plt.plot(data, color = 'indigo')
plt.title(f'Precio de las Acciones de Apple durante 2020')
plt.xlabel('Tiempo')
plt.ylabel('Precio')

# Función para calcular los estimadores
def estimadores(S, diferencia=1/252):
  # Calculamos diferencias logarítmicas
  S = np.array(S)
  Y = np.diff(np.log(S))
  N = len(S)

  # Estimadores
  estimador_sigmac = np.sqrt((1 / (N * diferencia)) * np.sum((Y - np.mean(Y))**2))
  estimador_mu = np.mean(Y) / diferencia + estimador_sigmac / 2

  # Devolvemos los estimadores
  return estimador_mu, estimador_sigmac

# Imprimimos los estimadores
mu_hat, sigma_hat = estimadores(data)
print('Estimador de mu:', mu_hat)
print('Estimador de sigma:', sigma_hat)

# Graficamos 5 trayectorias con los parámetros estimados
np.random.seed(123)

# Definimos los parámetros del proceso
S_inicial = data[0]
T = 1
N = 1000

# Hacemos 5 simulaciones
simulacion1 = MBG(S_inicial, mu_hat, sigma_hat, T, N)[:252]
simulacion2 = MBG(S_inicial, mu_hat, sigma_hat, T, N)[:252]
simulacion3 = MBG(S_inicial, mu_hat, sigma_hat, T, N)[:252]
simulacion4 = MBG(S_inicial, mu_hat, sigma_hat, T, N)[:252]
simulacion5 = MBG(S_inicial, mu_hat, sigma_hat, T, N)[:252]

# Graficamos las trayectorias junto con el proceso original
plt.figure(figsize=(10, 5))
plt.style.use('seaborn-v0_8-dark')
plt.plot(simulacion1, color = 'indigo', label='Simulación 1', linestyle = 'dashed')
plt.plot(simulacion2, color = 'deeppink', label='Simulación 2', linestyle = 'dashed')
plt.plot(simulacion3, color = 'crimson', label='Simulación 3', linestyle = 'dashed')
plt.plot(simulacion4, color = 'blueviolet', label='Simulación 4', linestyle = 'dashed')
plt.plot(simulacion5, color = 'orchid', label='Simulación 5', linestyle = 'dashed')
plt.plot(data, color = 'darkmagenta', label='Datos reales', linewidth=2.8)
plt.title(f'Precio de las Acciones de Apple durante 2020')
plt.legend()
#plt.xlim(0, 252)
plt.xlabel('Tiempo')
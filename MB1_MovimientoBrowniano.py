################################################################################
# Movimiento Browniano
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
# Simulación desde una Caminata Aleatoria

class CaminataAleatoria:

  # Definimos los datos iniciales
  def __init__(self, N, p, k_0):
    '''
    k_0: valor inicial de la caminata
    N: número de pasos
    p: probabilidad de que se mueva en la dirección positiva
    '''
    self.k_0 = k_0
    self.N = N
    self.p = p
    self.caminata_aleatoria = [k_0]

  # Función para simular la caminata
  def simular(self):
    for i in range(1, self.N):
        b = np.random.choice([-1, 1], p=[1 - self.p, self.p])
        self.caminata_aleatoria.append(self.caminata_aleatoria[-1] + b)
    return True

  # Función para imprimir la cadena
  def __str__(self):
    return str(self.caminata_aleatoria)

  # Función para graficar la cadena
  def plot(self):
    plt.style.use('seaborn-v0_8-dark')
    data = pd.DataFrame({'t': np.arange(1, self.N + 1), 'c': self.caminata_aleatoria})
    plt.figure(figsize=(7, 4))
    plt.plot(data['t'], data['c'], color='black')
    plt.grid()
    plt.title(f'Caminata aleatoria con p = {self.p}')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor del Proceso')
    plt.show()

# Ejemplo 1
ca1 = CaminataAleatoria(N=1000, p=0.5, k_0=0)
ca1.simular()
ca1.plot()

# Clase del MBE
# Heredamos de la clase CaminataAleatoria
class MovimientoBrowniano(CaminataAleatoria):
    def __init__(self, N, k_0=0):
      # Heredamos con p=0.5 (caminata aleatoria simple simétrica)
      super().__init__(N=N, p=0.5, k_0=k_0)
      self.Wt = np.array([k_0])

    def simular(self):
      super().simular()  # genera la caminata aleatoria

    def calcular_browniano(self, T=1):
      super().simular()
      t = np.linspace(0, T, self.N)
      # Normaliza la caminata
      W_tn = np.array(self.caminata_aleatoria) / np.sqrt(self.N)
      return t, W_tn

    def plot_browniano(self, T=1):
      t, W_tn = self.calcular_browniano(T)
      plt.figure(figsize=(7, 4))
      plt.plot(t, W_tn, color='navy')
      plt.grid()
      plt.title(r'Movimiento Browniano Estándar')
      plt.xlabel('tiempo')
      plt.ylabel(r'$X(t)$')
      plt.show()

# Ejemplo 2
mb1 = MovimientoBrowniano(N=1000)
mb1.plot_browniano()

# Ejemplo 3 
mb2 = MovimientoBrowniano(N=10000)
mb2.plot_browniano(1000)

# Clase del MB general
class MB(MovimientoBrowniano):

  def __init__(self, N, mu, sigma, k_0=0):
    super().__init__(N=N, k_0=k_0)
    self.mu = mu
    self.sigma = sigma
    self.w_0 = k_0
    self.Wt = None

  def simular_mbg(self):
    #self.simular()
    t, W_tn = super().calcular_browniano()
    self.Wt = self.w_0 + self.mu * t + self.sigma * W_tn
    #print(W_tn)
    return t, self.Wt

  def plot_browniano(self, T=1):
    t = np.linspace(0, T, self.N)
    plt.figure(figsize=(7, 4))
    plt.plot(t, self.Wt, color='navy')
    plt.grid()
    plt.title(r'Movimiento Browniano')
    plt.xlabel('tiempo')
    plt.ylabel(r'$W(t)$')
    plt.show()

# Ejemplo 4
mbg1 = MB(N=1000, mu=1.5, sigma=0.2)
mbg1.simular_mbg()
mbg1.plot_browniano()

# Ejemplo 5 con recta y = w_0 + \mu t
np.random.seed(123)
mbg2 = MB(N=1000, mu=1.5, sigma=0.2)
tiempo, W_t = mbg2.simular_mbg()

# Gráfica de la trayectoria y de la recta
plt.figure(figsize=(7, 4))
plt.plot(tiempo, W_t, color='navy',label = r'$W(t)$')
plt.plot(tiempo, 1.5*np.array(tiempo), color = 'indigo', label = r'Recta $y = 1.5x$')
plt.legend()
plt.grid()
plt.title(r'Movimiento Browniano')
plt.xlabel('tiempo')
plt.ylabel(r'$W(t)$')
plt.show()

# Ejemplo 6 con la recta y = w_0 + \mu t
np.random.seed(123)
mbg3 = MB(N=1000, mu=-1.5, sigma=0.2)
tiempo, W_t = mbg3.simular_mbg()

# Gráfica de la trayectoria y de la recta
plt.figure(figsize=(7, 4))
plt.plot(tiempo, W_t, color='navy', label = r'$W(t)$')
plt.plot(tiempo, -1.5*np.array(tiempo), color = 'indigo', label = r'Recta $y = -1.5x$')
plt.grid()
plt.legend()
plt.title(r'Movimiento Browniano')
plt.xlabel('tiempo')
plt.ylabel(r'$W(t)$')
plt.show()

################################################################################
# Simulación desde Incrementos Normales

# Código de prueba

# Definimos los parámetros
T = 1
browniano = [0]
N = 1000
t = np.linspace(0, T, N)
delta = T/N

# Aplicamos incrementos normales
for _ in t:
  X = np.random.normal()
  browniano.append(browniano[-1] + X*np.sqrt(delta))

# Graficamos
plt.figure(figsize=(7, 4))
plt.plot(t, browniano[:-1], color = 'navy')

# Clase para simular desde incrementos normales
class MBEINormales():
  
  def __init__(self, N):
    '''
    N : número de pasos
    Wt : lista de valores de la trayectoria del MBE
    '''
    self.N = N
    self.Wt = [0]

  def simular_browniano_in(self, T=1, N= 1000):
    # La partición y delta t
    t = np.linspace(0, T, N)
    delta = T/N

    # Incrementos normales
    for _ in t:
      X = np.random.normal()
      self.Wt.append(self.Wt[-1] + X*np.sqrt(delta))
    
    # Devolvemos el dominio y la trayectoria
    return t, self.Wt[:-1]

  def plot_browniano_in(self, T=1, N=1000):
    t, Wt = self.simular_browniano_in(T, N)
    plt.figure(figsize=(7, 4))
    plt.plot(t, Wt, color = 'navy')
    plt.title('Movimiento Browniano Estándar')
    plt.xlabel('tiempo')
    plt.ylabel(r'$W(t)$')
    plt.show()  

# Ejemplo 1
mbin1 = MBEINormales(N=1000)
mbin1.plot_browniano_in()

# Clase del MB por incrementos normales general
class MBIN_G(MBEINormales):

  def __init__(self, N, mu, sigma, w_0 = 0):
    '''
    N: número de pasos
    mu: parámetro del proceso MB
    sigma: parámetro del proceso MB
    w_0: valor inicial del proceso MB
    '''
    super().__init__(N)
    self.mu = mu
    self.sigma = sigma
    self.w0 = w_0
    self.Wt = [0]

  def simular_mb_in_g(self, T=1, N=1000):
    t, Wte = super().simular_browniano_in(T, N)
    # Aplicamos la definición de MB
    self.Wt = self.w0 + self.mu*t + self.sigma*np.array(Wte)
    return t, self.Wt

  def plot_mb_in_g(self, T=1, N=1000):
    t, Wt = self.simular_mb_in_g(T, N)
    plt.figure(figsize=(7, 4))
    plt.plot(t, Wt, color = 'navy')
    plt.title('Movimiento Browniano General')
    plt.xlabel('tiempo')
    plt.ylabel(r'$W(t)$')
    plt.show()

# Ejemplo 2
mbgin2 = MBIN_G(100, 1.5, 0.2, 0)
mbgin2.plot_mb_in_g()
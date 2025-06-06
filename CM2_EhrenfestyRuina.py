################################################################################
# Cadena de Ehrenfest y Ruina del Jugador
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
################################################################################

################################################################################
# Cadena de Ehrenfest

# Clase de Urna de Ehrenfest que toma como distribución inicial una lista
class UrnaEhrenfest:

  # Definimos datos iniciales
  def __init__(self, N, d, caja1 = list):
    '''
    N : número de iteraciones
    d : cantidad total de bolas
    caja1: lista de las etiquetas de las bolas en la caja 1
    '''
    self.N = N
    self.d = d
    self.caja1 = caja1
    self.caja2 = list(set(range(1, d+1)) - set(caja1))
    self.xn = [len(self.caja1)]
    self.x = len(self.caja1)

  # Función para simular
  def simular(self):
    for i in range(self.N):
      u = random.randint(1, self.d)
      if u in self.caja1:
        self.x -= 1
        self.caja1.remove(u)
        self.caja2.append(u)
      else:
        self.x += 1
        self.caja2.remove(u)
        self.caja1.append(u)
      self.xn.append(self.x)
    return True

  # Función para imprimir la cadena
  def __str__(self):
    return self.xn

  # Función para graficar
  def plot(self):
    plt.style.use('seaborn-v0_8-dark')
    data = pd.DataFrame({'t': np.arange(0, self.N + 1), 'ue': self.xn})
    plt.figure(figsize=(7, 4))
    plt.plot(data['t'], data['ue'], color='darkslategray', linewidth=1)
    plt.grid()
    plt.title(f'Urna de Ehrenfest con {self.d} bolas', fontsize=14)
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Valor del Proceso', fontsize=12)
    plt.show()

# Ejemplo con 1 bola
ue1 = UrnaEhrenfest(100, 1, [4,1])
ue1.simular()
ue1.plot()

# Clase heredada donde solo le decimos el número de bolas en la caja1
class UrnaEhrenfestAleatoria(UrnaEhrenfest):
    def __init__(self, N, d, caja1_size=None):
        # Si no se proporciona un tamaño para caja1, se elige aleatoriamente
        caja1_size = caja1_size if caja1_size is not None else random.randint(1, d)
        caja1 = random.sample(range(1, d+1), caja1_size)  # Elige aleatoriamente los elementos para caja1
        super().__init__(N, d, caja1)  # Llamada al constructor de la clase base

# Ejemplo de la nueva clase
ue2 = UrnaEhrenfestAleatoria(10000, 1000, 500)
ue2.simular()
ue2.plot()

# Ejemplos donde se ve el comportamiento hacia la media
ue3 = UrnaEhrenfestAleatoria(10000, 1000, 980)
ue3.simular()
ue3.plot()

ue4 = UrnaEhrenfestAleatoria(10000, 1000, 20)
ue4.simular()
ue4.plot()

################################################################################
# Ruina del Jugador

# Clase para la ruina del jugador
class Ruina1:

  # Definimos los datos iniciales
  def __init__(self, iter, p, d, k_0):
    '''
    k_0 : capital inicial
    iter : número de iteraciones
    p : probabilidad de ganar
    d : monto máximo
    '''
    self.k_0 = k_0 
    self.iter = iter 
    self.p = p 
    self.d = d 
    self.caminata_aleatoria = [k_0]

  # Función para simular la caminata
  def simular(self):
    for i in range(1, self.iter):
        b = np.random.choice([-1, 1], p=[1 - self.p, self.p])
        self.caminata_aleatoria.append(self.caminata_aleatoria[-1] + b)
    return True

  # Función para imprimir la cadena
  def __str__(self):
    return str(self.caminata_aleatoria)

  # Función para graficar la cadena
  def plot(self):
    plt.style.use('seaborn-v0_8-dark')
    data = pd.DataFrame({'t': np.arange(1, self.iter + 1), 'c': self.caminata_aleatoria})
    plt.figure(figsize=(7, 4))
    plt.plot(data['t'], data['c'], color='darkslategray', linewidth=1)
    plt.grid()
    plt.title(f'Ruina del Jugador con p = {self.p}', fontsize=16)
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Valor del Proceso', fontsize=12)
    plt.show()

# Clase con el tiempo de paro incorporado
class RuinadelJugador(Ruina1):

  def __init__(self, iter, p, d, k_0):
    # El mismo constructor de la clase Ruina1
      super().__init__(iter, p, d, k_0)

  def simular(self):
    for i in range(1, self.iter):
      # Acá agregamos la condición de paro
      if self.caminata_aleatoria[-1] == 0 or self.caminata_aleatoria[-1] == self.d:
        self.caminata_aleatoria.append(self.caminata_aleatoria[-1])
        continue
      b = np.random.choice([-1, 1], p=[1 - self.p, self.p])
      self.caminata_aleatoria.append(self.caminata_aleatoria[-1] + b)
    return True

# Ejemplo
ruina2 = RuinadelJugador(10000, 0.5, 100, 50)
ruina2.simular()
ruina2.plot()

# Visualización del cambio en la función u respecto de p y el valor inicial
def u_k(p, d, k):
  if p == 0.5:
    return (d-k)/d
  else:
    q = 1-p
    num = (q/p) ** k - (q/p) ** d
    den = 1 - (q/p) ** d
    return num/den

# DataFrame con algunos valores específicos de u
datos_uk = pd.DataFrame({'k': np.arange(0, 51), 'u_k1': [u_k(0.1, 50, k) for k in np.arange(0, 51)], 'u_k2': [u_k(0.3, 50, k) for k in np.arange(0, 51)]
                      , 'u_k3': [u_k(0.5, 50, k) for k in range(0, 51)], 'u_k4': [u_k(0.55, 10500, k) for k in np.arange(0, 51)], 'u_k5': [u_k(0.7, 50, k) for k in np.arange(0, 51)],
                         'u_k6': [u_k(0.9, 50, k) for k in range(0, 51)], 'u_k7': [u_k(0.45, 50, k) for k in range(0, 51)]})

# Gráfico del cambio en u respecto al valor inicial
plt.figure(figsize=(7, 4))
plt.plot(datos_uk['k'], datos_uk['u_k1'], color='olive', linewidth=1, label='p = 0.1')
plt.plot(datos_uk['k'], datos_uk['u_k2'], color='forestgreen', linewidth=1, label='p = 0.3')
plt.plot(datos_uk['k'], datos_uk['u_k7'], color='lime', linewidth=1, label='p = 0.45')
plt.plot(datos_uk['k'], datos_uk['u_k3'], color='red', linewidth=1, label='p = 0.5')
plt.plot(datos_uk['k'], datos_uk['u_k4'], color='purple', linewidth=1, label='p = 0.55')
plt.plot(datos_uk['k'], datos_uk['u_k5'], color='indigo', linewidth=1, label='p = 0.7')
plt.plot(datos_uk['k'], datos_uk['u_k6'], color ='deeppink', linewidth=1, label='p = 0.9')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Probabilidad de perder respecto al valor inicial')
plt.xlabel('Capital Inicial')
plt.ylabel('Probabilidad de ganar 1')
plt.style.use('seaborn-v0_8-dark')
plt.xlim(0, 50)
plt.ylim(-0.2, 1.2)
plt.grid()
plt.show()

# DataFrame para guardar el cambio de u respecto de p
datos_uk2 = pd.DataFrame({'p': np.linspace(0.01, 0.98, 98), 'u_k1': [u_k(p, 50, 5) for p in np.linspace(0.01, 0.98, 98)], 'u_k2': [u_k(p, 50, 25) for p in np.linspace(0.01, 0.98, 98)],
            'u_k3': [u_k(p, 50, 45) for p in np.linspace(0.01, 0.98, 98)]})

# Gráficos para visualizar el cambio de u respecto de p
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# Primer subplot
axes[0].plot(datos_uk2['p'], datos_uk2['u_k1'], color='blue')
axes[0].set_title('Probabilidad de perder respecto a valor de p con k=5')
axes[0].set_xlabel(r'$p$')
axes[0].set_ylabel(r'u_k')
axes[0].plot([0.5 for _ in range(100)], [i for i in np.linspace(0,1,100)], '--')
axes[0].grid()

# Segundo subplot
axes[1].plot(datos_uk2['p'], datos_uk2['u_k2'], color='green')
axes[1].set_title('Probabilidad de perder respecto a valor de p con k=25')
axes[1].set_xlabel(r'$p$')
axes[1].set_ylabel(r'u_k')
axes[1].plot([0.5 for _ in range(100)], [i for i in np.linspace(0,1,100)], '--', color='lime')
axes[1].grid()

# Tercer subplot
axes[2].plot(datos_uk2['p'], datos_uk2['u_k3'], color='red')
axes[2].set_title('Probabilidad de ruina respecto a valor de p con k=45')
axes[2].set_xlabel(r'$p$')
axes[2].set_ylabel(r'u_k')
axes[2].plot([0.5 for _ in range(100)], [i for i in np.linspace(0,1,100)], '--', color = 'brown')
axes[2].grid()

plt.tight_layout()
plt.show()



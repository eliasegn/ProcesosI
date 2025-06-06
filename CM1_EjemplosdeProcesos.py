################################################################################
# Ejemplos de Procesos Estocásticos
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
# Proceso a Ensayos Independientes
n = 30 # número de días
p = 0.75  # probabildad de correr

# Generar 30 v.a. bin(p)
xp = np.random.binomial(n=1, p=p, size=n)

# generamos la v.a. binomial asociada
yp = np.cumsum(xp)

# Los días que hemos corrido en total
yp = np.cumsum(xp)

print("He corrido los días:", xp)
print(f"En total, hasta el día {n} he corrido {yp[-1]} veces.")

# Creamos un DataFrame con Pandas
datos_xp = pd.DataFrame({
    'x': np.arange(1, len(xp) + 1),
    'equis': xp
    'y' : yp
})

# Graficamos al proceso x
plt.figure(figsize=(7,4))
plt.style.use('seaborn-v0_8-dark')
plt.grid()
sns.scatterplot(data=datos_xp, x='equis', y='y', color='darkcyan')
plt.ylim(-1, 2)
plt.title("Día a día")
plt.grid()
plt.xlabel("Día")
plt.ylabel("¿Corrió?")
plt.show()

# Graficamos al proceso y
plt.figure(figsize=(7, 4))
plt.style.use('seaborn-v0_8-dark')
sns.scatterplot(data=datos_xp, x='x', y='y', color='darkorange')
plt.title("Acumulado")
plt.grid()
plt.xlabel("Día")
plt.ylabel("Total")
plt.show()

# Automatización del proceso de generación del proceso
def procesosxy(n, p):
    xpr = np.random.binomial(n=1, p=p, size=n)
    ypr = np.cumsum(xpr)
    return pd.DataFrame({'x': xpr, 'y': ypr})

# Función para visualizar
def visualizador_xy(n, p):
    pro1 = procesosxy(n, p)
    datospro1 = pd.DataFrame({
        'x': pro1['x'],
        'y': pro1['y'],
        't': np.arange(1, len(pro1) + 1)
    })

    # Primer gráfico: Día a día
    plt.figure(figsize=(7, 4))
    plt.style.use('seaborn-v0_8-dark')
    plot1 = sns.scatterplot(data=datospro1, x='t', y='x', color='navy')
    plt.title('Día a día')
    plt.grid()
    plt.xlabel('Día')
    plt.ylabel('¿Corrió?')
    plt.show()

    # Segundo gráfico: Acumulado
    plt.figure(figsize=(7, 4))
    plot2 = sns.scatterplot(data=datospro1, x='t', y='y', color='darkgreen')
    plt.title('Acumulado')
    plt.grid()
    plt.xlabel('Día')
    plt.ylabel('Total')
    plt.show()

# Llamamos la función
visualizador_xy(100, 0.3)
###############################################################################

###############################################################################
# Caminata Aleatoria en Z
# Parámetros iniciales
caminata_aleatoria = [0]  # 0 es la Posición inicial
N = 50  # Número de pasos
p = 0.5  # Probabilidad de avanzar

# Función para simular
for i in range(1, N):
    b = np.random.choice([-1, 1], p=[1 - p, p])  # Escoge aleatoriamente avanzar o retroceder
    caminata_aleatoria.append(caminata_aleatoria[-1] + b)

# Creamos un DataFrame para los datos
data = pd.DataFrame({'t': np.arange(1, N + 1), 'c': caminata_aleatoria})

# Graficar
plt.figure(figsize=(7, 4))
plt.style.use('seaborn-v0_8-dark')
plt.plot(data['t'], data['c'], color='indigo', linewidth=1)
plt.title('Caminata aleatoria con p = 0.5')
plt.xlabel('Tiempo')
plt.ylabel('Valor del Proceso')
plt.grid(True)
plt.show()

# Clase de Caminata Aleatoria
class CaminataAleatoria:

  # Definimos los datos iniciales
  def __init__(self, N, p, k_0=0):
    '''
    N : número de pasos de la caminata
    p : probabilidad de avanzar un paso
    k_0 : punto de inicio (0 por default)
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
    plt.plot(data['t'], data['c'], color='black', linewidth=1)
    plt.grid()
    plt.title(f'Caminata aleatoria con p = {self.p}', fontsize=16)
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Valor del Proceso', fontsize=12)
    plt.show()

# Caminta aleatoria simple simétrica
ca1 = CaminataAleatoria(N=50, p=0.5, k_0=0)
ca1.simular()
ca1.plot()

# Caminata con proba de transición 1/2 y empezando en 900 con 1000 pasos
ca2 = CaminataAleatoria(1000, p=0.5, k_0=900)
ca2.simular()
ca2.plot()

# Estudio de la esperanza del proceso
# Caminata con p = 0.8 y recta de esperanza
import numpy as np
cm3 = CaminataAleatoria(1000, p=0.8, k_0=0)
cm3.simular()
plt.figure(figsize=(7, 4))
plt.style.use('seaborn-v0_8-dark')
plt.plot(cm3.caminata_aleatoria, color = 'blue', label = 'Caminata')
plt.title('Caminata aleatoria con p = 0.8')
plt.plot(np.linspace(0,1000,100000), np.linspace(0,1000,100000)*(2*0.8-1), color = 'red', label = 'Esperanza')
plt.legend()
plt.grid()
plt.show()

# Caminta con p = 0.2 y recta del proceso
cm4 = CaminataAleatoria(1000, p=0.2, k_0=0)
cm4.simular()
plt.figure(figsize=(7, 4))
plt.style.use('seaborn-v0_8-dark')
plt.plot(cm4.caminata_aleatoria, color = 'blue', label = 'Caminata')
plt.title('Caminata aleatoria con p = 0.2')
plt.plot(np.linspace(0,1000,100000), np.linspace(0,1000,100000)*(2*0.2-1), color = 'red', label = 'Esperanza')
plt.legend()
plt.grid()
plt.show()

# Caminata aleatoria simétrica simple
sim = CaminataAleatoria(1000, p=0.5, k_0=0)
sim.simular()
sim.plot()

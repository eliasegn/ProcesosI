#################################################################################
# EJEMPLOS DE CADENAS DE MARKOV
#################################################################################
# Librerías
library(ggplot2)
library(glue)
#################################################################################

# Ruina del Jugador

ruina_del_jugador <- function(ci, N, p){
  proceso <- c(ci)
  
  while(proceso[length(proceso)] != 0 && proceso[length(proceso)] != N){
    x <- sample(c(-1,1), 1, prob=c(1-p, p))
    proceso <- c(proceso, proceso[length(proceso)]+x)
  }
  return(proceso)
  
}

# Definimos los parámetros iniciales
ci <- 800 # capital inicial
N <- 1000 # capital máximo
p <- 0.6 # probabilidad de éxito

jugador1 <- ruina_del_jugador(ci, N, p)
jugador1 <- data.frame(x = seq(1, length(jugador1)), y = jugador1)

ggplot(jugador1, aes(x = x, y = y)) +
  geom_point(color = "navy", size = 1) +
  labs(title = "Ruina del Jugador", x = "Tirada", y = "Capital") +
  theme_minimal()

install.packages("tinytex")
tinytex::install_tinytex()
N

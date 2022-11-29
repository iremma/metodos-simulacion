from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------
#          Variables globales
#----------------------------------------------------------------------


x_1 = 70 #numero de unidades del producto 1
x_2 = 70 #numero de unidades del producto 2
T_simulacion = 5
#T_simulacion = 5*30*24 #tiempo transcurrido en la simulacion en horas
Tp = 7*24 #cada cuanto pide
lista = {'tc': 0,  # tiempo en el que ha llegado un cliente
         'tpc': 0, # tiempo en el que se ha comprado un pedido
         'tp': 0}  # tiempo en el que ha llegado un pedido
R = 0 #beneficio esperado
P1 = 1000 #numero de unidades max del producto 1
P2 = 1500 #numero de unidades max del producto 2
Nc = 0 #numero de clientes satisfechos
Nnc = 0 # numero de clientes no satisfechos
t0 = 0 # numero de tiempo que el intervalo esta a cero
C = 0 #coste total por pedidos
H = 0 #coste total por almacenamiento
lambda_poisson = 0.5
demanda = [1, 2, 3, 4] #posibles demandas del producto
probab_1 = [0.3, 0.4, 0.2, 0.1] #probabilidades de demanda del producto 1
probab_2 = [0.2, 0.2, 0.4, 0.2] #probabilidades de demanda del producto 2
r_1 = 2.5 #coste al publico del producto 1
r_2 = 3.5 #coste al publico del producto 2
h = 0.0002 #precio sumado por producto y unidad de tiempo
mu = 48 #tiempo de media que tarda un pedido
sigma = 3.5 #desviacion tipica de lo que tarda un pedido
K = 100 #coste fijo del proveedor
n_descuent_1 = 600 #unidades mayores a estas obtienen descuento en producto 1
p1_1 = 1 # precio si son menos de 600 uds del producto 1
p1_2 = 0.75 # precio si son mas de 600 uds del producto 1
n_descuent_2 = 800
p2_1 = 1.5
p2_2 = 1.25
Lref = 48 #tiempo media de llegada del pedido
lim_penal = 3 #a partir de estas horas de retraso del pedido, penalizacion
penal = 0.0003 # la penalizacion en el precio si llega tarde/pronto 
t_real = 0
var_aux = 0 # instante en el que el almacen se vacia completamnete
L = 0
ts = 0 #tiempo de simulacion

#Vectores para la representación gráfica de los niveles de inventario de los dos tipos de producto a lo largo del tiempo

tiempos_1 = [0]
niveles_1 = [70]
tiempos_2 = [0]
niveles_2 = [70]

# datos_grafica [producto 1 o 2] [tiempo (0) o nivel(1)] [i]  
datos_grafica = ["",
                 [[0],[70]], # tiempo y nivel producto 1
                 [[0],[70]]  # tiempo y nivel producto 2
                ]

#----------------------------------------------------------------------
#             Metodos
#----------------------------------------------------------------------

def rutina_llegada_cliente(ts):
  global H, h, t_real, x_1, x_2, Nc, Nnc, var_aux, R, Y, y_1, y_2
  global r_1, r_2, var_aux, T_simulacion, tiempos_1, tiempos_2
  global niveles_1, niveles_2
  
  #Aumenta el coste de almacenamiento
  H += (ts-t_real)*h*(x_1+x_2)
  t_real = ts
  
  #Generamos demanda del cliente
  demanda_1 = np.random.choice(demanda, 1, p=probab_1)[0]
  demanda_2 = np.random.choice(demanda, 1, p=probab_2)[0]
  
  print("------------------------------------------------------------")
  print(f"Llega nuevo cliente\n  - demanda producto 1: {demanda_1}\n  - demanda producto 2: {demanda_2}")
  print(f"Estado del almacen: x_1:{x_1}, x_2:{x_2} ")
  
  #Si hay suficiente almacenado, esta satisfecho
  if demanda_1<=x_1 and demanda_2<=x_2 :
    R += demanda_1*r_1 + demanda_2*r_2 #sube el beneficio
    x_1 -= demanda_1   #baja el inventario
    x_2 -= demanda_2
    Nc += 1 #cliente satisfecho
    print("> Cliente satisfecho!")
  #Si no hay suficiente almacenado de algun producto, no esta satisfecho
  else:
    if(demanda_1<=x_1):
      R += demanda_1*r_1 
      x_1 -= demanda_1
    elif(demanda_2<=x_2):
      R += demanda_2*r_2 
      x_2 -= demanda_2
    Nnc += 1 #cliente no satisfecho
    print("> Cliente no satisfecho")
  
  print(f"Estado del almacen: x_1:{x_1}, x_2:{x_2} ")
    
  # Si se ha vaciado del todo (y antes no estaba vacio) guardamos el tiempo actual
  if x_2 == 0 and x_1 == 0 and var_aux == 0 :
    var_aux = t_real
    
  #Generamos el tiempo que tarda en llegar el siguiente cliente
  Y = stats.poisson.rvs(lambda_poisson, size=1)[0] 
  
  # si el cliente llega antes de acabar la simulacion, se simula
  if Y+t_real < T_simulacion:
    lista['tc'] = t_real+Y 
    
  datos_grafica[1][0].append(t_real)
  datos_grafica[1][1].append(x_1)
  datos_grafica[2][0].append(t_real)
  datos_grafica[2][1].append(x_2)
  
  #tiempos_1.append(t_real)
  #niveles_1.append(x_1)
  #tiempos_2.append(t_real)
  #niveles_2.append(x_2)
  
  print(f"tiempo actual: {t_real}")

def rutina_llegada_pedido(ts):
  global H, K, h, t_real, C, t0, var_aux, x_1, x_2, y_1, y_2
  global p1_1, p1_2, p2_1, p2_1, penal, var_aux
  global tiempos_1, tiempos_2, niveles_1, niveles_2
  
  #Aumenta el coste de almacenamiento
  H += (ts-t_real)*h*(x_1+x_2)
  t_real = ts
  
  print("------------------------------------------------------------")
  print("El pedido ha llegado")
  print(f"Estado del almacen (antes): x_1:{x_1}, x_2:{x_2} ")
  
  #Aumenta el nivel de inventario
  x_1 += y_1
  x_2 += y_2
  
  print(f"Estado del almacen (desp.): x_1:{x_1}, x_2:{x_2} ")
  
  #Si son muchas unidades, descuento en el precio
  Ci_1 = K + y_1 * p1_1 if y_1<=n_descuent_1 else K + y_1 * p1_2
  Ci_2 = K + y_2 * p2_1 if y_2<=n_descuent_2 else K + y_2 * p2_2
  
  #Si llega tarde, penalizacion en el coste total
  C += (Ci_1+Ci_2)*(1-penal) if L-Lref>lim_penal else (Ci_1+Ci_2)*(1+penal)
  
  #Ya no quedan productos por llegar
  y_1 = 0
  y_2 = 0
  
  datos_grafica[1][0].append(t_real)
  datos_grafica[1][1].append(x_1)
  datos_grafica[2][0].append(t_real)
  datos_grafica[2][1].append(x_2)
  
  #tiempos_1.append(t_real)
  #niveles_1.append(x_1)
  #tiempos_2.append(t_real)
  #niveles_2.append(x_2)
  
  # Si estaba vacio el invenario (se vacio en el instante var_aux),
  # aumenta el tiempo que ha estado vacio.
  if var_aux > 0:
    t0 += t_real - var_aux
    var_aux = 0
  
  print(f"> El almacen ha estado vacio este tiempo: {t0-t_real} ")
  print(f"tiempo actual: {t_real}")

def rutina_compra_pedido(ts):
  global H, x_1, x_2, t_real, y_1, y_2, h, t_real
  global P1, P2, lista, T_simulacion
  
  print("------------------------------------------------------------")
  print("Realizamos pedido al proveedor")
  print(f"Estado del almacen: x_1:{x_1}, x_2:{x_2} ")

  # Aumenta el coste de almacenamiento
  H += (ts-t_real)*h*(x_1+x_2)
  t_real = ts
  
  #Cantidad a pedir es lo que falta para llenar el almacen
  y_1 = P1 - x_1
  y_2 = P2 - x_2

  #Generamos cuanto va a tardar en llegar el pedido
  L = np.random.normal(mu, sigma, 1)[0]

  # actualizamos el tiempo de llegada del pedido y el tiempo de siguiente compra
  lista['tp'] = t_real + L if L+t_real < T_simulacion else lista['tp']
  lista['tpc'] = t_real + Tp if t_real+Tp < T_simulacion else lista['tpc']
  
  print(f"> El pedido tardara este tiempo: {L} ")
  print(f"> Se ha pedido: x_1:{y_1} x_2:{y_2} ")
  print(f"tiempo actual: {t_real}")

def simul_main():
  
  # Iniciamos siumlacion
  ts = 0
  lista['tc'] = 4000
  lista['tp'] = 4000
  lista['tpc'] = Tp

  print("===========================================================================")
  print("                          SIMULACION INICIADA                              ")
  print("===========================================================================")
  #Generamos el tiempo que tarda en llegar el primer cliente
  Z = stats.poisson.rvs(lambda_poisson, size=1)[0]

  #Si el tiempo si pasa del limite T, la simulación se acaba
  if(Z > T_simulacion): return -1

  print('** Llega el primer cliente')
  rutina_llegada_cliente(Z)
  
  #Repetir si siguen llegando clientes o siguen llegando pedidos
  while lista['tc']!=4000 or lista['tp']!=4000:
    
    #Si el siguiente evento es la llegada de un cliente
    if lista['tc'] <= lista['tpc'] and lista['tc'] <= lista['tp']:
      ts = lista['tc']
      lista['tc'] = 4000
      print("** Llega un cliente")
      rutina_llegada_cliente(ts)

    #Si el siguiente evento es la compra de un pedido 
    if(lista['tpc']<=lista['tc'] and lista['tpc']<=lista['tp']):
      ts = lista['tpc']
      lista['tpc'] = 4000
      print("** Realizo pedido")
      rutina_compra_pedido(ts)

    #Si el siguiente evento es una llegada de pedido
    if(lista['tp']<=lista['tc'] and lista['tp']<=lista['tpc']):
      ts = lista['tp']
      lista['tp'] = 4000
      print("** Llega un pedido")
      rutina_llegada_pedido(ts)
  benef = R-C-H                    #Beneficios
  cl_satisf = Nc / (Nc + Nnc) *100 #Porcentaje de clientes satisfechos
  t0_tot = t0 / T_simulacion       #Tiempo que ha estado el almacen vacio
  
  print("===========================================================================")
  print("                          FIN DE LA SIMULACION                             ")
  print("===========================================================================")
  print("beneficio->",R,"coste pedidos->",C," coste almacenamiento->",H,"\n")
  
  print(f"beneficio: {benef}")
  print(f"% clientes satisfechos: {cl_satisf}")
  print(f"tiempo con el almacen vacio: {t0_tot}")
  return benef, cl_satisf, t0_tot


#----------------------------------------------------------------------
#          Simulacion
#----------------------------------------------------------------------

simul_main()

# PLOT RESULTS
fig = plt.figure()
plt.plot(datos_grafica[1][0],datos_grafica[1][1], color='red', label="Producto 1")
plt.plot(datos_grafica[2][0],datos_grafica[2][1], color='blue', label="Producto 2")
plt.legend()
plt.title(f"Simulacion de {T_simulacion} h")
fig.savefig("sim.png")

#plt.axis([200, 250, 0, 1000])

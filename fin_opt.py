import numpy as np
import math as m

L = 0.100 # m
w = 0.190 # m
q = 900 # W
u = 20 #m/s
H = 0.030 #m
T_inf = 47 #degC
T_b = 50
t_grease = 0.0001
k_grease = 0.79
t_case = 0.001
k_case =150.0

#6061-T6 Al
k_b = 200# W/mK

#Air
v = 1.68E-5 #m^2/s
k_s = 0.026726 #W/mK
Pr = 0.67
gamma = 1.8

h = (0.664*k_s*(Pr)**(1/3)*np.sqrt(u/L))/np.sqrt(v)
s = 2*gamma*np.sqrt((v*L)/u)
#Nu = 2.656*gamma*(Pr)**(1/3)
Nu = 8.235
alpha = np.sqrt(k_b/(k_s*Nu))
t = H/(alpha)
N = m.floor(w/(s+t))
lamb = H/(s*alpha)
omega = H/(k_b*w*L)
R_lossless = omega/lamb**2

if lamb>= 1:
	zeta = 2*lamb + 1
else:
	zeta = (np.sqrt(lamb)*(lamb+1))/np.tanh(lamb)
	print('lamb greater than 1')

R_hs = R_lossless*zeta

R_case = t_grease/(k_grease*L*w)
R_grease = t_case/(k_case*L*w)
R_tot = R_hs + R_grease + R_case
T_b = (R_tot*q)+T_inf #base temperature


print(T_b, R_tot, H, s, t, N, h)



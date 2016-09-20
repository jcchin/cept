import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab

mass_flow = 0.306 #kg/s
d_inner = 0.352 #m
d_outer = 0.371 #m
density = 1.033 #kg/m^3
L = 0.13
air_viscosity = 1.94E-5
Pr = 0.7174
k = 0.0285

X = [0] * 129
H = [0] * 129
T = [0] * 129
i = 0

T_ambient = 50
q = 308

for L in np.arange(.001,0.13, 0.001):
    c = np.pi*d_inner
    flow_area = (np.pi/4)*(d_outer**2-d_inner**2)
    velocity = mass_flow * density * (1/flow_area)

    Re = velocity*L/air_viscosity
    bl_t = (5.47*L)/(Re**.5)
    therm_bl_t = bl_t * Pr ** -0.33
    therm_bl_t_in = therm_bl_t * 39.37 #thermal boundary layer inches

    h = 0.53*Re**0.5*Pr**0.5*0.0285/L #http://www.rpi.edu/dept/chem-eng/WWW/faculty/plawsky/Comsol%20Modules/BLThermal/ThermalBoundaryLayer.html

    X[i] = L
    H[i] = h
    T[i] = therm_bl_t_in

    i = i+1

avg_h = np.mean(H)
#avg_hbar = (0.53*((mass_flow*density)/(c*air_viscosity))**0.5*Pr**0.5*k*(m.log(L)-m.log(0.01)))/L

Ts = q/(avg_h*c*0.13) + T_ambient

print(Ts, avg_h)
plt.figure()
plt.plot(X,H)
plt.xlabel('Length (m)')
plt.ylabel('Heat Transfer Coefficient (W/m^2K)')
pylab.show()

plt.figure()
plt.plot(X,T)
plt.xlabel('Length (m)')
plt.ylabel('Thermal Boundary Layer Thickness (in)')
pylab.show()

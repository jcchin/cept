#!/usr/bin/env python3

# tracks lumped capacitance temperatures of
# X57 high lift motors and inverters

import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab
from itertools import accumulate
import scipy.integrate as integrate


A_conv_motor = .0474  # m^2
h_conv_motor = 30  # W/m^2-C
len_motor = 0.065
eff_motor = 0.95

A_conv_inv = 0.05  # m^2 - estimate of available cabin surface area for external convection
h_conv_inv = 50  # W/m^2-C estimate of cabin external convection coefficient (computed for Lc = 0.8m, u_0=50 m/s)
len_inv = 0.08
eff_inv = 0.97


#free convection parameters
rho_0 = 1.055  # kg/m^3
mu_air = 1.81E-05  # Pa-s, viscosity of ambient air
k_air = 0.026  # W/m-C, therm. conductivity of air

k_wall_nacelle = 200  # W/m-C module wall conductivity (Al 6061)
t_wall_nacelle = .003  # m module wall thickness
U_conv_motor = A_conv_motor/((1/h_conv_motor)+(t_wall_nacelle/k_wall_nacelle))  # W/C total heat transfer coefficient from module to cabin
U_conv_motor = 30.  # override value 5 W/m^2-C


cp_air = 1005  # J/kg-C specific heat of air (average)
cp_al = 900  # J/kg-C specific heat of aluminum grid and external plate
m_nacelle_air = 1.  # m^3
m_inv = 0.5  # kg, grid and external plate area per 160 cell module.
m_motor = 1.0  # kg mass of cells per 160 cell module

mcp_nacelle = m_nacelle_air*cp_air
mcp_motor = m_inv*cp_al
mcp_inv = m_motor*cp_al

T_0 = 30  # C module and aircraft equilibrium temperature at start (HOT DAY)

ts = 0.5  # time step, s

import csv
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list

with open('mission.csv') as f:
    reader = csv.DictReader(f, delimiter=',',) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(float(v)) # append the value into the appropriate list
                                 # based on column name k

mc = list(accumulate(columns['segmentDuration'])) #uncomment if using 'segment duration'
#mc = columns['elapsedTime']  # uncomment if using 'elapsedTime'
time = [0]
time.extend(mc[:-1])
altitude = columns['altitude']
DEP_pwr = columns['DEP_pwr']
U0 = columns['U0']  # speed

#http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
alt_table = [-1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000] # meters
temp_table = [49., 49., 8.5, 2., -4.49, -10.98, -17.47, -23.96, -30.45] # Celsius (21.5 @ 0 in reality, using 49 to be conservative)

def lookup(x, schedule):
             #Taxi            # TO Checklist              Cruise Runup              HLP Runup              Flight go/no-go                Ground roll             Climb to 1000 feet           Cruise Climb                 Cruise                Descent to 1000 feet        Final approach            Go around to 1000 feet          Approach pattern               Final Approach           Rollout and turnoff                 Taxi
    #conds = [x < mc[0], (x > mc[0]) & (x < mc[1]), (x > mc[1]) & (x < mc[2]), (x > mc[2]) & (x < mc[3]), (x > mc[3]) & (x < mc[4]), (x > mc[4]) & (x < mc[5]),(x > mc[5]) & (x < mc[6]), (x > mc[6]) & (x < mc[7]), (x > mc[7]) & (x < mc[8]), (x > mc[8]) & (x < mc[9]), (x > mc[9]) & (x < mc[10]), (x > mc[10]) & (x < mc[11]), (x > mc[11]) & (x < mc[12]), (x > mc[12]) & (x < mc[13]), (x > mc[13]) & (x < mc[14]), (x > mc[14]) & (x < mc[15])]
    conds = []#[x < mc[0]]
    for i,bp in enumerate(mc):
        conds.extend(((x > mc[i-1]) & (x < mc[i]),))
    return np.select(conds, schedule)

# initiate transient variable arrays
times = np.arange(0, mc[-1], ts)

solverTime = []
Ambient_Temp = []  # np.zeros(t_len)  # ambient temp --> per altitude model

Q_gen = []  # rate of heat absorbed by battery module
dTdt = []  # rate of battery module temperature change
Module_Hconv = []  # module convective heat transfer coefficient
Module_Ra = []
Module_Nusselt = []
Delta_T = []

Q_cool = []  # module cooling rate log
Q_net = []  # module heat accumulation log

# states:
#0 = Motor_Temp     # motor (bulk) temperature
#1 = Inv_Temp  # inverter (bulk) temperature
#2 = Nacelle_Temp # nacelle skin (bulk) temperature
def dxdt(temp, t):

    dep_power = lookup(t, DEP_pwr)
    alt = np.interp(t, time, altitude)
    T_0 = np.interp(alt, alt_table, temp_table)

    Pmotor = 1000./12.*dep_power  # W, total power requested by propulsion
    Qmotor = (1.-eff_motor)*Pmotor  # W, instantaneous heat dissipated by the motor
    Qinv = (1.-eff_motor)*(Pmotor+Qmotor)  # W, inverter heat diss

    # convection model - adjacent plates
    #Tfilm = 0.5*(temp[0]+temp[2]+273.)  # guesstimate of freeconv film temperature
    #Beta = 1./Tfilm
    #Ra_module = (rho_0*9.81*Beta*cp_air*(s_module**4)*(temp[0]-temp[2])+.01)/(mu_air*k_air*h_module)
    #Nu_module = (Ra_module/24.)*((1-np.exp(-35./Ra_module))**0.75)

    #hconv_module = Nu_module*k_air/Lc_module  # free convection hconv
    #A_conv_module = Lc_module**2  # free convection featureless surface

    # hconv_module = 30. #forced convection hconv, given sink design
    # A_conv_module = .06 #m^2, finned surface with n=10, 1cm X 30 cm fins

    U_conv_motor = h_conv_motor*A_conv_motor
    U_conv_inv = h_conv_inv*A_conv_inv

    #convection rates from module and cabin from current time step T's
    Qconv_motor = U_conv_motor*(temp[0]-T_0)
    Qconv_inv = U_conv_inv*(temp[1]-T_0)

    #temperature corrections
    dT_motor = (Qmotor - Qconv_motor)/mcp_motor  # cabin with module and avionics heat load
    dT_inv = (Qinv - Qconv_inv)/mcp_inv  # module heat loss to convection

    # save off other useful info (not integrated states)
    # note: the time breakpoints are "solverTime", not "times"
    #       to allow forsolver adaptive time stepping
    solverTime.extend([t])
    Ambient_Temp.extend([T_0])
    # dTdt.extend([dT_module])
    # Module_Hconv.extend([hconv_module])
    # Module_Ra.extend([Ra_module])
    # Module_Nusselt.extend([Nu_module])
    # Delta_T.extend([temp[0] - temp[2]])
    Q_gen.extend([Qmotor])

    return [dT_motor, dT_inv]

#Integrate
Temp_states = integrate.odeint(dxdt, [T_0,T_0], times, hmax=0.5)

# Print Results
print('Max Comp Temp: %f deg C' % max(Temp_states[:,0]))
# print('Battery system thermal mass: %f deg C/J' % (12.*mcp_module))
# print('Cabin air thermal mass: %f deg C/J' % mcp_cabin_air)
# print('Cabin participating structure thermal mass: %f deg C/J' % mcp_cabin)
# print('Module overall HTC: %f W/deg C' % U_conv_module)
# print('Cabin overall HTC: %f W/deg C' % U_conv_cabin)
# print('Average module heat absorption: %f W' % np.mean(Q_gen))
# print('Peak module heat absorption: %f W' % np.max(Q_gen))
# print('Average module-cabin delta-T: %f deg C' % np.mean(Delta_T))
# print('Minimum module-cabin delta_T: %f deg C' % np.min(Delta_T))
# print('Maximum module-cabin delta_T: %f deg C' % np.max(Delta_T))


fig1 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig1.add_subplot(111)
ax3 = fig1.add_subplot(111)
ax4 = ax2.twinx()
ax5 = fig1.add_subplot(111)

ax1.plot(times, Temp_states[:,0], 'r-', label='Motor Temperature')
# ax2.plot(times, Temp_states[:,2], 'g-', label='Cabin Temperature')
# ax3.plot(solverTime, Ambient_Temp, 'b-', label='Ambient Temperature')
ax4.plot(solverTime, Q_gen, 'g-', label='Motor Heat (W)')
ax5.plot(solverTime, Ambient_Temp, 'b-', label='Ambient Temp (C)')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Temperature (deg C)', color='k')
ax4.set_ylabel('Module absorbed heat rate(W)', color='k')

legend = ax1.legend(loc='upper center', shadow=True)
#legend = ax2.legend(loc='upper center', shadow=True)
legend = ax4.legend(loc='upper center', shadow=True)

plt.show()

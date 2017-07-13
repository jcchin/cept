#!/usr/bin/env python3

# tracks lumped capacitance temperatures of
# X57 high lift motors and inverters

import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab
from itertools import accumulate
import scipy.integrate as integrate


A_conv_motor = .0474  # m^2  # availabe area for convection on the motor
Al_cond = 215  # W/m*K      # aluminum conductivity
Al_thick = 0.01             # aluminum motor thickness
h_conv_motor = 30  # W/m^2-C   # heat transfer coeff
r_motor = 0.75  # K/W    #thermal resistance from the stator to the outer metal
len_motor = 0.065
eff_motor = 0.95

A_conv_inv = 0.05  # m^2
h_conv_inv = 50  # W/m^2-C
r_sink = 0.75  # K/W
len_inv = 0.08
eff_inv = 0.97

inlet_D = 0.03 # inlet diameter
inlet_A = np.pi*(inlet_D/2.)**2.
R_hs = 0.4   # heat sink thermal resistance (computed in heatSinkSMA.py) match CFM

paste_thick = 0.004       #thermal paste from stator end windings to SMA drum
paste_cond = 0.7  # W/m-K   #thermal conductivity
R_paste = paste_thick/paste_cond  # K/W   #thermal resistance

#free convection parameters
rho_0 = 1.055  # kg/m^3
mu_air = 1.81E-05  # Pa-s, viscosity of ambient air
k_air = 0.026  # W/m-C, therm. conductivity of air

k_wall_nacelle = 200  # W/m-C module wall conductivity (Al 6061)
t_wall_nacelle = .003  # m module wall thickness
U_conv_motor = A_conv_motor/((1/h_conv_motor)+(t_wall_nacelle/k_wall_nacelle))  # W/C total heat transfer coefficient from module to cabin
U_conv_motor = 30.  # override value 30 W/m^2-C


cp_air = 1005  # J/kg-C specific heat of air (average)
cp_al = 900  # J/kg-C specific heat of aluminum grid and external plate
m_nacelle_air = 1.  # m^3
m_inv = 1.0  # kg, grid and external plate area per 160 cell module.
m_motor = 2.0  # kg mass of cells per 160 cell module

mcp_nacelle = m_nacelle_air*cp_air  #thermal mass
mcp_motor = m_inv*cp_al
mcp_inv = m_motor*cp_al

T_0 = 35.  # C module and aircraft equilibrium temperature at start (HOT DAY)

ts = 0.5  # time step, s

# - read data from mission.csv -------------
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
# ---------------------------------------
#http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
alt_table = [-1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000] # meters
temp_table = [35., 35., 35., 8.5, 2., -4.49, -10.98, -17.47, -23.96]#, -30.45] # Celsius (21.5 @ 0 in reality, using 49 to be conservative)
rho_table = [1.347, 1.225, 1.112, 1.007, 0.9093, 0.8194, 0.7364, 0.6601, 0.59] #1.055  # inlet density, kg/m^3

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
#2 = Motor_Temp with SMA cooling path
def dxdt(temp, t):

    dep_power = lookup(t, DEP_pwr)
    vel = np.interp(t,time,U0)
    alt = np.interp(t, time, altitude)
    T_0 = np.interp(alt, alt_table, temp_table)
    rho_0 = np.interp(alt, alt_table, rho_table)

    #print(inlet_A * vel) #CFM

    Pmotor = (1000./12.)*dep_power  # W, total power requested by propulsion
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
    Qconv_motor = U_conv_motor*((r_motor+R_paste)*(temp[0]-T_0))
    Qconv_inv = U_conv_inv*((r_sink)*(temp[1]-T_0))
    Qconv_motor2 = U_conv_motor*((r_motor+R_paste)*(temp[2]-T_0))
    Q_cool = (temp[2]-T_0)/R_hs
    if (temp[2]<70):
        Q_cool = 0.
    #temperature corrections
    dT_motor = (Qmotor - Qconv_motor)/mcp_motor  # cabin with module and avionics heat load
    dT_inv = (Qinv - Qconv_inv)/mcp_inv  # module heat loss to convection
    dT_motor2 = (Qmotor - Qconv_motor2 - Q_cool)/mcp_motor

    # save off other useful info (not integrated states)
    # note: the time breakpoints are "solverTime", not "times"
    #       to allow forsolver adaptive time stepping
    solverTime.extend([t])
    Ambient_Temp.extend([T_0])

    if Pmotor < 10000:
        Delta_T.extend([dT_motor])
    Q_gen.extend([Qmotor])

    return [dT_motor, dT_inv, dT_motor2]

#Integrate
Temp_states = integrate.odeint(dxdt, [T_0,T_0,T_0], times, hmax=0.5)

t_limit = (100.-40.)/max(Delta_T)

# Print Results
print('Max Comp Temp: %f deg C' % max(Temp_states[:,0]))
print('Temp Over Time: %f seconds' % t_limit)


fig1 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig1.add_subplot(111)
ax3 = fig1.add_subplot(111)
ax4 = ax2.twinx()
ax5 = fig1.add_subplot(111)

ax1.plot(times, Temp_states[:,0], 'r-', label='Motor Temperature')
# ax2.plot(times, Temp_states[:,2], 'g-', label='Cabin Temperature')
ax3.plot(times, Temp_states[:,1], 'k-', label='Inverter Temperature')
ax3.plot(times, Temp_states[:,2], 'c-', label='Cool Motor Temperature')
ax4.plot(solverTime, Q_gen, 'g-', label='Motor Heat (W)')
ax5.plot(solverTime, Ambient_Temp, 'b-', label='Ambient Temp (C)')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Temperature (deg C)', color='k')
ax4.set_ylabel('Module absorbed heat rate(W)', color='k')

legend = ax1.legend(loc='upper center', shadow=True)


plt.show()

#!/usr/bin/env python3

#  Calculate transient duct temperature across entire mission profile
#  Consider altitude and power profile
#  Duct temperature state integrated implicitly using Euler integration
#  (no explicit differential equations)

#  Major Assumptions
# ------------------
#  Duct assumed to be a circle of end radius (conservative)
#  0-Dimensional Transient (dT/dt), no spatial thermal gradient

#  Methods
# ------------------
#  Heating derived from joule heating
#  Cooling derived from natural convection
#  Thermal inertia from wire heat capacity
#  Thermal rate of change driven by Q_in - Q_out
#  Assume a control volume around all wires for heat balance
#  Convection coefficient held constant h = 7.0 W/m-K (conservative)
#  (Doesn't vary much, saves numerical instability)


import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab
from itertools import accumulate
import scipy.integrate as integrate

# Mission Inputs
app_alt = 457.2  # approach altitude, in meters (1500 ft)
cruise_alt = 2438.4  # m, --> 8,000 ft

# mission time segments in seconds
m1 = 600  # Taxi from NASA
m2 = 120  # Take off checklist
m3 = 30  # Cruise Runup
m4 = 30  # HLP Runup
m5 = 30  # flight go/no-go
m6 = 10  # ground role
m7 = 90  # climb to 1000'
m8 = 540  # cruise climb
m9 = 300  # cruise
m10 = 450  # descent to 1500'
m11 = 180  # final approach
m12 = 90  # go around to 1500'
m13 = 90  # approach pattern
m14 = 180  # final approach
m15 = 60  # role-out turn off
m16 = 600  # taxi to NASA

#carbon fiber specs
t_fg = 0.0006096  # (meters) fiberglass duct thickness
k_fg = 0.04  # (W/mK) fiberglass thermal conductivity

# air properties
k_air = 0.0285  # stagnant air thermal conductivity
t_air = 0.003937  # air gap inside duct

bus_voltage = 416  # Operating Voltage
motor_efficiency = 0.95
inverter_efficiency = 0.95

# wire specs
RperL = 0.00096 #RperL = 0.005208 #.003276  # (ohm/m) resistance of AWG 10 wire

alt_derating = 0.925
d = 0.01905  # (m) duct diameter
circ_tpe = 0.040132 # jacket circumference

HC = 240.
ts = 0.1  # time step for Euler Integration
hC = 3.3
hD = 3.3

T_cruise0 = 49.
T_dep0 = 49.
T_ambient = 49.
#--------------------------------------------------

# cumulative mission time breakpoints for altitude table lookup
mc = list(accumulate([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16]))

time =       [0,  mc[1], mc[2], mc[3], mc[4], mc[5], mc[6],   mc[7],      mc[8],      mc[9],  mc[10], mc[11],  mc[12],  mc[13], mc[14],  mc[15]]
altitude =   [0.,    0.,    0.,    0.,    0.,    0.,    0., app_alt, cruise_alt, cruise_alt, app_alt,     0., app_alt, app_alt,     0.,     0.]  # m
DEP_pwr =    [0.,    0.,    0.,   30.,    0.,   30.,   15.,      0.,         0.,         0.,     15.,    15.,      0.,     15.,     0.,     0.]
Cruise_pwr = [2.5,   0.,   30.,    0.,    0.,   30.,   30.,     30.,       22.5,        15.,      0.,    30.,    22.5,      0.,    3.8,    2.5]


def lookup(x, schedule):
             #Taxi            # TO Checklist              Cruise Runup              HLP Runup              Flight go/no-go                Ground roll             Climb to 1000 feet           Cruise Climb                 Cruise                Descent to 1000 feet        Final approach            Go around to 1000 feet          Approach pattern               Final Approach           Rollout and turnoff                 Taxi
    conds = [x < mc[0], (x > mc[0]) & (x < mc[1]), (x > mc[1]) & (x < mc[2]), (x > mc[2]) & (x < mc[3]), (x > mc[3]) & (x < mc[4]), (x > mc[4]) & (x < mc[5]),(x > mc[5]) & (x < mc[6]), (x > mc[6]) & (x < mc[7]), (x > mc[7]) & (x < mc[8]), (x > mc[8]) & (x < mc[9]), (x > mc[9]) & (x < mc[10]), (x > mc[10]) & (x < mc[11]), (x > mc[11]) & (x < mc[12]), (x > mc[12]) & (x < mc[13]), (x > mc[13]) & (x < mc[14]), (x > mc[14]) & (x < mc[15])]
    return np.select(conds, schedule)

#http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
alt_table = [-1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000] # meters
temp_table = [49., 49., 8.5, 2., -4.49, -10.98, -17.47, -23.96, -30.45] # Celsius (21.5 @ 0 in reality, using 49 to be conservative)

# initiate transient variable arrays
times= np.arange(0, mc[-1], ts)

# perform engineering calculations in a 'for' loop
# across the entire mission length updating state values for each time step
# (Euler Integration of Duct Temperature State)
def dxdt(temp,t):

    cruise_power = lookup(t, Cruise_pwr)  # kW
    dep_power = lookup(t, DEP_pwr)  # kW

    alt = np.interp(t, time, altitude)
    T_ambient = 49.#np.interp(alt, alt_table, temp_table)

    cruise_bus_power = cruise_power/(motor_efficiency*inverter_efficiency)  # kW
    cruise_bus_current = (1000.*cruise_bus_power/bus_voltage)/alt_derating  # ****
    q_prime_cruise = cruise_bus_current**2. * RperL

    # DEP BUS
    dep_bus_power = dep_power/(motor_efficiency*inverter_efficiency)  # kW
    dep_bus_current = (1000*dep_bus_power/bus_voltage)/alt_derating  # ****
    q_prime_dep = dep_bus_current**2 * RperL

    q_prime_outC = circ_tpe * (temp[0]-T_ambient) * hC #T_cruise
    q_prime_outD = circ_tpe * (temp[1]-T_ambient) * hD #T_dep

    Temp_ROCC = (q_prime_cruise - q_prime_outC) / HC
    Temp_ROCD = (q_prime_dep - q_prime_outD) / HC

    return [Temp_ROCC, Temp_ROCD]


Temp_wires = integrate.odeint(dxdt,[T_cruise0, T_dep0],times)

# Print Results

plt.figure()

plt.plot(times,Temp_wires[:,0], 'b', label = 'Cruise Ducts', lw=1.5)
plt.plot(times,Temp_wires[:,1], 'g', label = 'DEP Ducts', lw=1.5)

plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel('Time (s)')
plt.ylabel('Wire Temperature (C)',fontsize=18)
plt.rcParams['xtick.labelsize'] = 24
# plt.ylabel('Current rating per conductor (Amp)')
plt.axvline(mc[1], color='k', linestyle='--',lw=0.5)
plt.axvline(mc[2], color='k', linestyle='--',lw=0.5)
plt.axvline(mc[3], color='k', linestyle='--',lw=0.5)
plt.text(250, 55,'Taxi', fontsize=8)
plt.axvline(mc[4], color='k', linestyle='--',lw=0.5)
plt.axvline(mc[5], color='k', linestyle='--',lw=0.5)
plt.text(mc[2]-50, 72,'Take-Off', fontsize=8)
plt.axvline(mc[6], color='k', linestyle='--',lw=0.5)
plt.text(mc[6]+150, 40,'10kft Climb', fontsize=8)
plt.axvline(mc[7], color='k', linestyle='--',lw=0.5)
plt.text(mc[7]+50, 65,'Cruise', fontsize=8)
plt.axvline(mc[8], color='k', linestyle='--',lw=0.5)
plt.text(mc[8]+150, 60,'Descent', fontsize=8)
plt.axvline(mc[9], color='k', linestyle='--',lw=0.5)
plt.text(mc[9]+150, 60,'Go-Around', fontsize=8)
plt.axvline(mc[10], color='k', linestyle='--',lw=0.5)
plt.axvline(mc[11], color='k', linestyle='--',lw=0.5)
plt.axvline(mc[12], color='k', linestyle='--',lw=0.5)
plt.axhline(max(Temp_wires[:,0]), color='k', linestyle='--',lw=1)
plt.axhline(74, color='g', linestyle='--',lw=1)
plt.text(2400, 74,['Temp Limit: %2.0f' % 76], fontsize=16)
plt.text(2400, max(Temp_wires[:,0]),['Max Temp: %2.2f' % max(Temp_wires[:,0])], fontsize=16)
pylab.show()

# a = np.asarray([ Time, Temp, Duct_Temp ])
# np.savetxt("temp_data.csv", a, delimiter=",")

# output = np.column_stack((Time.flatten(),Temp.flatten(),Duct_Temp.flatten()))
# np.savetxt('temp_data.csv',output,delimiter=',')


# import plotly.plotly as py
# from plotly import tools
# import plotly.graph_objs as go


# # Create traces
# trace0 = go.Scatter(
#     x = Time,
#     y = Temp_cruise,
#     mode = 'lines',
#     name = 'Cruise'
# )

# trace1 = go.Scatter(
#     x = Time,
#     y = Temp_dep,
#     mode = 'lines',
#     name = 'DEP'
# )

# data = [trace0, trace1]

# py.iplot(data, filename='wire_test')

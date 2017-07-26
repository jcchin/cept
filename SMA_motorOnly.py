#!/usr/bin/env python3

# tracks lumped capacitance temperatures of
# X57 high lift motors and inverters

import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab
from itertools import accumulate
import scipy.integrate as integrate

Al_cond = 215  # W/m*K      # aluminum conductivity
Al_t = 4E-3  # m

Iron_cond = 79.5  # W/m*K
Iron_t = 3.5E-3  # m

cp_copper = 385  # J/kg
rho_copper = 8940  # kg/m^3
A_Cu = np.pi*(138.5E-3**2 - 130.0E-3**2)
HC = cp_copper*rho_copper*A_Cu  # J/(K*m)
C = np.pi*.134

h_conv_motor = 30  # W/m^2-C   # heat transfer coeff
T_0 = 35.
ts = 0.5  # time step, s

# R_tot = ((Iron_t/Iron_cond)+(Al_t/Al_cond)+ (1/h_conv_motor))/C
# q_cool = (T_0 - T_amb)/R_tot
eff_motor = 0.95
L_motor = 0.04  # m

# - read data from mission.csv -------------
import csv
from collections import defaultdict

columns = defaultdict(list)  # each value in each column is appended to a list

with open('mission.csv') as f:
    reader = csv.DictReader(f, delimiter=',',)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items():  # go over each column name and value
            columns[k].append(float(v))  # append the value into the appropriate list
                                 # based on column name k

mc = list(accumulate(columns['segmentDuration']))  # uncomment if using 'segment duration'
#mc = columns['elapsedTime']  # uncomment if using 'elapsedTime'
time = [0]
time.extend(mc[:-1])
altitude = columns['altitude']
DEP_pwr = columns['DEP_pwr']
U0 = columns['U0']  # speed
# ---------------------------------------
#http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
alt_table = [-1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]  # meters
temp_table = [35., 35., 8.5, 2., -4.49, -10.98, -17.47, -23.96, -30.45]  # Celsius (21.5 @ 0 in reality, using 49 to be conservative)
rho_table = [1.347, 1.225, 1.112, 1.007, 0.9093, 0.8194, 0.7364, 0.6601, 0.59]  # 1.055  # inlet density, kg/m^3


def lookup(x, schedule):
             #Taxi            # TO Checklist              Cruise Runup              HLP Runup              Flight go/no-go                Ground roll             Climb to 1000 feet           Cruise Climb                 Cruise                Descent to 1000 feet        Final approach            Go around to 1000 feet          Approach pattern               Final Approach           Rollout and turnoff                 Taxi
    #conds = [x < mc[0], (x > mc[0]) & (x < mc[1]), (x > mc[1]) & (x < mc[2]), (x > mc[2]) & (x < mc[3]), (x > mc[3]) & (x < mc[4]), (x > mc[4]) & (x < mc[5]),(x > mc[5]) & (x < mc[6]), (x > mc[6]) & (x < mc[7]), (x > mc[7]) & (x < mc[8]), (x > mc[8]) & (x < mc[9]), (x > mc[9]) & (x < mc[10]), (x > mc[10]) & (x < mc[11]), (x > mc[11]) & (x < mc[12]), (x > mc[12]) & (x < mc[13]), (x > mc[13]) & (x < mc[14]), (x > mc[14]) & (x < mc[15])]
    conds = []  # [x < mc[0]]
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


def dxdt(temp, t):

    dep_power = lookup(t, DEP_pwr)
    alt = np.interp(t, time, altitude)
    T_amb = np.interp(alt, alt_table, temp_table)

    Pmotor = (1000./12.)*dep_power  # W, total power requested by propulsion
    Qmotor = (1.-eff_motor)*Pmotor  # W, instantaneous heat dissipated by the motor
    q_prime_motor = ((1.-eff_motor)*Pmotor)/L_motor  # W/m, instantaneous heat dissipated by the motor
    R_tot = ((Iron_t/Iron_cond)+(Al_t/Al_cond) + (1/h_conv_motor))  # /C # Km/W
    q_prime_cool = (temp[0] - T_amb)/R_tot

    R_tot_open = (1/(h_conv_motor + ((Iron_t/Iron_cond)+(Al_t/Al_cond) + (1/h_conv_motor))**-1))/C
    q_prime_cool_open = (temp[1] - T_amb)/R_tot_open

    dT_motor = (q_prime_motor - q_prime_cool)/HC  # cabin with module and avionics heat load
    dT_motor_open = (q_prime_motor - q_prime_cool_open)/HC

    solverTime.extend([t])
    Ambient_Temp.extend([T_amb])

    if Pmotor < 10000:
        Delta_T.extend([dT_motor])

    Q_gen.extend([Qmotor])

    return [dT_motor, dT_motor_open]

Temp_states = integrate.odeint(dxdt, [T_0, T_0], times, hmax=0.5)

# Print Results
# print('Max Comp Temp: %f deg C' % max(Temp_states[:, 0]))
# print('Temp Over Time: %f seconds' % t_limit)


fig1 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig1.add_subplot(111)
# ax3 = fig1.add_subplot(111)
ax4 = ax2.twinx()
ax5 = fig1.add_subplot(111)

ax1.plot(solverTime, Q_gen, 'g-', label='Motor Heat (W)')

ax4.plot(times, Temp_states[:,1], 'c-', label='Open Motor Temperature')
# ax3.plot(times, Temp_states[:,1], 'k-', label='Inverter Temperature')
# ax3.plot(times, Temp_states[:,2], 'c-', label='Cool Motor Temperature')
ax4.plot(times, Temp_states[:,0], 'r-', label='Motor Temperature')
ax4.plot(solverTime, Ambient_Temp, 'b-', label='Ambient Temp')

ax1.set_xlabel('Time (s)')
ax4.set_ylabel('Temperature (deg C)', color='k')
ax1.set_ylabel('Module absorbed heat rate(W)', color='g')

legend = ax4.legend(loc='upper center', shadow=True)


plt.show()









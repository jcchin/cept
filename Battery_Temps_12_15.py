
# 1. Test fit cell cp and efficiency to match EPS data for 60 cell block
# 2. Use generated cell heating model with X-57 current schedule, scaling down
#      from full battery system to 60 cell block
# 3. Determine steady cooler requirement to keep batteries below critical temp: model
#      first as constant Q (heat pump, simpler to model), then as Qconv ~ h*(Tbatt-Tenv).

import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab
from itertools import accumulate

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

mass_cell = 45 # grams
Cp_cell = 1.02 # J/(gm*C)
eff_cell = 0.05
N_cell = 640  #32s20p module @ 3.6 V cell nominal
V_cell=3.6 #V
Ns = 32
Np = 20

T_0 = 35 # C
T0_50 = T_0
T0_150 = T_0
T0_350 = T_0
T0_h10 = T_0
T0_h25 = T_0

T_sink = 35 #C
dT = 10 #C

hconv = 0 #W/C


V_module = Ns*V_cell # volts:  32s @ 3.6V/cell
V_X57 = 461

N_scale = 0.125 #scale from full vehicle to module (8x modules for full aircraft)

ts = 0.5 # time step, s

mc = list(accumulate([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16]))

t_len = int(mc[-1] * 1./ts)
Time = np.zeros(t_len)

Temp = np.zeros(t_len)
Temp50 = np.zeros(t_len)
Temp150 = np.zeros(t_len)
Temp350 = np.zeros(t_len)
Temph10 = np.zeros(t_len)
Temph25 = np.zeros(t_len)

Current = np.zeros(t_len)
Q_cell = np.zeros(t_len)
Q_module = np.zeros(t_len)
Q_cool = np.zeros(t_len)
Q_net = np.zeros(t_len)

i = 0

for t in np.arange(0, mc[-1], ts):

    if t <= mc[0]:  # Taxi from NASA
        dep_power = 0.0  # kW
        cruise_power = 10.0  # kW

    elif t > mc[0] and t <= mc[1]:  # TO Checklist
        dep_power = 0.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[1] and t <= mc[2]:  # Cruise Runup
        dep_power = 0.0  # kW
        cruise_power = 120.0  # kW

    elif t > mc[2] and t <= mc[3]:  # HLP Runup
        dep_power = 30.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[3] and t <= mc[4]:  # Flight go/no-go
        dep_power = 0.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[4] and t <= mc[5]:  # Ground roll
        dep_power = 30.0  # kW
        cruise_power = 120.0  # kW

    elif t > mc[5] and t <= mc[6]:  # Climb to 1000 feet
        dep_power = 15.0  # kW
        cruise_power = 120.0  # kW

    elif t > mc[6] and t <= mc[7]:  # Cruise Climb
        dep_power = 0.0  # kW
        cruise_power = 120.0  # kW

    elif t > mc[7] and t <= mc[8]:  # CruiseW
        dep_power = 0.0  # kW
        cruise_power = 90.0  # kW

    elif t > mc[8] and t <= mc[9]:  # Descent to 1000 feet
        dep_power = 0.0  # kW
        cruise_power = 60.0  # kW

    elif t > mc[9] and t <= mc[10]:  # Final approach
        dep_power = 15.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[10] and t <= mc[11]:  # Go around to 1000 feet
        dep_power = 15.0  # kW
        cruise_power = 120.0  # kW

    elif t > mc[11] and t <= mc[12]:  # Approach pattern
        dep_power = 0.0  # kW
        cruise_power = 90.0  # kW

    elif t > mc[12] and t <= mc[13]:  # Final Approach
        dep_power = 15.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[13] and t <= mc[14]:  # Rollout and turnoff
        dep_power = 0.0  # kW
        cruise_power = 15.0  # kW

    #elif t>2955 and t<=3555:
    else:               # Taxi to NASA
        dep_power = 0.0  # kW
        cruise_power = 10.0  # kW



    I_X57 = 1000*cruise_power/V_X57
    #I_module = (1000*(cruise_power + dep_power))/(V_module*Np)  #module electrical power @V_module
    I_module = I_X57/2

    Qconv = 25 * (T0_h25 - T_sink)
    Q = eff_cell * I_module * V_module  #module dissipated thermal power

    T_module = T_0 + Q*ts/(N_cell*mass_cell*Cp_cell)
    T50 = T0_50 + (Q - 50)*ts/(N_cell*mass_cell*Cp_cell)
    T150 = T0_150 +  (Q - 150)*ts/(N_cell*mass_cell*Cp_cell)
    T350 = T0_350 + (Q - 350)*ts/(N_cell*mass_cell*Cp_cell)
    Th10 = T0_h10 + (Q - 10 * (T0_h10 - T_sink))*ts/(N_cell*mass_cell*Cp_cell)
    Th25 = T0_h25 + (Q - 25 * (T0_h25 - T_sink))*ts/(N_cell*mass_cell*Cp_cell)



    Time[i] = t
    Temp[i] = T_module
    Temp50[i] = T50
    Temp150[i] = T150
    Temp350[i] = T350
    Temph10[i] = Th10
    Temph25[i] = Th25

    Current[i] = I_module
    Q_module[i] = Q
    Q_cell[i] = Q/N_cell
    Q_cool[i] = Qconv
    Q_net[i] = Q - Qconv

    T_0 = T_module
    T0_50 = T50
    T0_150 = T150
    T0_350 = T350
    T0_h10 = Th10
    T0_h25 = Th25




    i = i + 1

    # Print Results


print('Max Temp: %f' % max(Temp))
print('Average Cooling PWR: %f' % np.mean(Q_cool))
print('Average Module Qgen: %f' % np.mean(Q_module))

fig, ax1 = plt.subplots()

plt.axhline(40, color='k', linestyle='--',lw=0.5)
plt.axhline(50, color='k', linestyle='--',lw=0.5)
plt.axhline(60, color='k', linestyle='--',lw=0.5)
plt.axhline(70, color='k', linestyle='--',lw=0.5)
plt.axhline(80, color='k', linestyle='--',lw=0.5)
plt.axhline(90, color='k', linestyle='--',lw=0.5)




ax1.plot(Time, Temp, 'b-')
ax1.plot(Time, Temph25, 'g-')



ax1.set_xlabel('time (s)')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('Module Temp (C)', color='b')

for tl in ax1.get_yticklabels():
    tl.set_color('b')


"""
ax2 = ax1.twinx()
ax2.plot(Time, Q_module, 'r-')
ax2.set_ylabel('Module Qgen (W)', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
"""

#plt.axvline(mc[1], color='k', linestyle='--',lw=0.5)
#plt.axvline(mc[2], color='k', linestyle='--',lw=0.5)
#plt.axvline(mc[3], color='k', linestyle='--',lw=0.5)
#plt.text(250, 50,'Taxi', fontsize=8)
#plt.axvline(mc[4], color='k', linestyle='--',lw=0.5)
#plt.axvline(mc[5], color='k', linestyle='--',lw=0.5)
#plt.text(mc[3]-50, 50,'Take-Off', fontsize=8)
#plt.axvline(mc[6], color='k', linestyle='--',lw=0.5)

#plt.text(mc[5]+150, 160,'Climb', fontsize=8)
#plt.axvline(mc[7], color='k', linestyle='--',lw=0.5)
#plt.text(mc[6]+50, 75,'Cruise', fontsize=8)
#plt.axvline(mc[8], color='k', linestyle='--',lw=0.5)
#plt.text(mc[7]+150, 75,'Descent', fontsize=8)
#plt.axvline(mc[9], color='k', linestyle='--',lw=0.5)
#plt.text(mc[8]+150, 75,'Go-Around', fontsize=8)
#plt.axvline(mc[10], color='k', linestyle='--',lw=0.5)
#plt.axvline(mc[11], color='k', linestyle='--',lw=0.5)
#plt.axvline(mc[12], color='k', linestyle='--',lw=0.5)
#plt.axhline(max(Temp), color='k', linestyle='--',lw=0.5)
plt.text(2500, max(Temp)+2,['Max Temp (adiabatic): %f' % max(Temp)], fontsize=8)
plt.text(2500, max(Temph25)+2,['Max Temp (h = 25 W/C): %f' % max(Temph25)], fontsize=8)

plt.show()

import plotly.plotly as py
from plotly import tools
import plotly.graph_objs as go


# Create traces
trace0 = go.Scatter(
    x = Time,
    y = Temp,
    mode = 'lines',
    name = 'Adiabatic'
)

trace1 = go.Scatter(
    x = Time,
    y = Temph10,
    mode = 'lines',
    name = 'Cooling h=10'
)

trace2 = go.Scatter(
    x = Time,
    y = Temph25,
    mode = 'lines',
    name = 'Cooling h=25'
)

data = [trace0, trace1, trace2]

py.iplot(data, filename='battery2')

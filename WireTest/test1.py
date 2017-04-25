#!/usr/bin/env python3

# 50A
# 30 sec

import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab
from itertools import accumulate

# K factor = 0.23 BTU-in / hr – ft2 – °F
# Btu in / (h ft^2 F) * 0.14 = W / (m K)
# 1" thick 7/8" ID 3' length https://www.mcmaster.com/#5556k31/=165irzj

# wire dimensions (NEAT) .22" x .57" Thermoplastic Elastomer Jacket
# wire dimensions (New) .19" x.66" -55 to 200C, 10awg 7 x 3 x 50/40 silver plated copper
# http://www.methode.com/power/power-cabling/ppc.html#.WJDSZrYrKHo

# TPE info
#https://plastics.ulprospector.com/generics/53/c/t/thermoplastic-elastomer-tpe-properties-processing/sp/24
#140C max temp, 200C melting point (Vicat Softening)

#insulation specs
t_fg = 0.00254  # (meters) fiberglass duct thickness
k_fg = 0.23*0.14  # (W/mK) fiberglass thermal conductivity
k_tpe = 0.2 # (W/m K) Thermoplastic elastomer
# http://www.efunda.com/materials/polymers/properties/polymer_datasheet.cfm?MajorID=TPE&MinorID=1

# test conditions
bus_voltage = 416  # Operating Voltage
current = 55.3  # amps
P_time = 21.*60.  # seconds powered on
C_time = 10.*60. # seconds cooling open-air
# air properties
k_air = 0.0285  # (W/m K) stagnant air thermal conductivity

# wire specs
RperL = 0.00096  # 0.005208  # .003276  # (ohm/m) resistance of AWG 10 wire
bundle_derating = 1.  # 0.5125

d_cu = 2.59E-3  # copper diameter 10AWG
A_cu = 4. * (np.pi*(d_cu)**2/4.)  # total Cu area
Cp_cu = 385.  # (J/kg*K) Copper Specific Heat
rho_cu = 8960.  # (kg/m^3) Copper Density
HC_cu = A_cu*Cp_cu*rho_cu  # (J/K*m) Heat Capacity per length of Copper Wire

circ_tpe = 0.040132  # (.22*2 + .57*2)inch -> m# jacket circumference
A_tpe = 8.09E-5 - A_cu # total area - A_cu
Cp_tpe = 580. # (J/kg*K) specific heat TPE
rho_tpe = 1200.  # (kg/m^3)
HC_tpe = A_tpe*Cp_tpe*rho_tpe  # (J/K*m) Heat Capacity per length of Copper Wire
HC = (HC_cu + HC_tpe)  * 1.75

HC = 240.
#HC = 322.
print(HC)

ts = 0.1  # time step for Euler Integration
q_conv = 0.  # starting heat rate (q) out
T_ins = 24.  # starting wire temperature (Celcius)
T_open = 22.5  # starting wire temperature (Celcius)
T_ambient = 22.5
#--------------------------------------------------

# initiate transient variable arrays
ttime = (P_time + C_time)
t_len = int(ttime/ts)


Time = np.zeros(t_len)
Temp = np.zeros(t_len)
Temp_open = np.zeros(t_len)
Temp_ins = np.zeros(t_len)

# (Euler Integration of Duct Temperature State)
# all calculations are per length (ohm/m, W/m, etc..)
for i,t in enumerate(np.arange(0, ttime, ts)):

    q_prime_in = current**2 * RperL
    #hi = 2.3  # fully insulated
    hi = -0.1031 * (ts/600.) + 3.3583
    ho = 17.29  # (W/m^2K) open-air cooling rate

    if(t > P_time): # turn off power, turn on fans
        q_prime_in = 0.
        hi = 2.3  # (W/m^2K) fan cooling rate
        ho = 17.29
    q_prime_outi = circ_tpe * (T_ins-T_ambient) * hi

    #print(q_prime_in)
    Ins_Temp_ROC = (q_prime_in-q_prime_outi)/HC
    T_ins = T_ins + (ts*Ins_Temp_ROC)


    q_prime_outo = circ_tpe * (T_open-T_ambient) * ho

    Open_temp_ROC = (q_prime_in-q_prime_outo)/HC
    T_open = T_open + (ts*Open_temp_ROC)

    # save instantaneous states to array
    Time[i] = t
    Temp_open[i] = T_open
    Temp_ins[i] = T_ins



# Print Results
plt.figure()
plt.xlabel('Time (m)')
plt.ylabel('Wire Temperature (C)',fontsize=18)
plt.rcParams['xtick.labelsize'] = 24

import csv
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list

with open('Cal4c.csv') as f:
    reader = csv.DictReader(f, delimiter=',',) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k

TestTime = np.asarray(columns['Time'],dtype=np.float32)/60.
O2 = np.asarray(columns['O2'])
O4 = np.asarray(columns['O4'])
I2 = np.asarray(columns['I2'])
I4 = np.asarray(columns['I4'])

skip = 415  # cut out beginning
end = 31*60 # cut out end
I2 = I2[skip:]
I4 = I4[skip:]
O2 = O2[skip:]
O4 = O4[skip:]
TestTime = TestTime[skip:] - float(skip)/60. #sync timestep
I2 = I2[:end]
I4 = I4[:end]
O2 = O2[:end]
O4 = O4[:end]
TestTime = TestTime[:end]

plt.plot(Time/60.,Temp_ins, 'r', label = 'Model Insulated', lw=1.5)
plt.plot(TestTime,I2, 'k', label = 'Observed Insulated', lw=1.5)
#plt.plot(TestTime,I4, 'k--', label = 'Test I4', lw=1.5)
plt.plot(Time/60.,Temp_open, 'g', label = 'Model Open', lw=1.5)
plt.plot(TestTime,O2, 'g', label = 'Observed Open', lw=1.5)
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
#plt.plot(TestTime,O4, 'g--', label = 'Test O2', lw=1.5)


pylab.show()

# output = np.column_stack((Time.flatten(),Temp.flatten(),Wire_Temp.flatten()))
# np.savetxt('temp_data.csv',output,delimiter=',')

import plotly.plotly as py
from plotly import tools
import plotly.graph_objs as go


# Create traces
trace0 = go.Scatter(
    x = Time/60,
    y = Temp_ins,
    mode = 'lines',
    name = 'InsModel'
)

trace1 = go.Scatter(
    x = TestTime,
    y = I2,
    mode = 'lines',
    name = 'InsTest'
)

trace2 = go.Scatter(
    x = Time/60,
    y = Temp_open,
    mode = 'lines',
    name = 'InsModel'
)

trace3 = go.Scatter(
    x = TestTime,
    y = O2,
    mode = 'lines',
    name = 'InsTest'
)
data = [trace0, trace1, trace2, trace3]

py.iplot(data, filename='test1')

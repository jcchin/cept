#!/usr/bin/env python3

#  Calculate transient duct temperature across entire mission profile
#  Consider altitude and power profile
#  Duct temperature state integrated implicitly using Euler integration
#  (no explicit differential equations)

#  Major Assumptions
# ------------------
#  Duct is exposed to ambient air at all points (***important! Either need ambient air inside duct, or inside wing)
#  Wire insulation ignored (conservative)
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

# mission time segments in seconds
#m1 = 600  # Taxi from NASA
#m2 = 120  # Take off checklist
m1 = 5 # up
m2 = 30  # Cruise Runup
m3 = 30  # HLP Runup
m4 = 5   # down
m5 = 30  # flight go/no-go
m6 = 5 # up
m7 = 10  # ground role
m8 = 90  # climb to 1500'
m9 = 540  # cruise climb
m10 = 300  # cruise
m11 = 450  # descent to 1500'
m12 = 5 # down
m13 = 180  # final approach
m14 = 5 # up
m15 = 90  # go around to 1500'
m16 = 90  # approach pattern
m17 = 180  # final approach
#m15 = 60  # role-out turn off
#m16 = 600  # taxi to NASA

#insulation specs
t_fg = 0.00254  # (meters) fiberglass duct thickness
k_fg = 0.23*0.14  # (W/mK) fiberglass thermal conductivity
k_tpe = 0.2 # (W/m K) Thermoplastic elastomer
# http://www.efunda.com/materials/polymers/properties/polymer_datasheet.cfm?MajorID=TPE&MinorID=1

# air properties
k_air = 0.0285  # stagnant air thermal conductivity
t_air = 0.003937  # air gap inside duct

bus_voltage = 416  # Operating Voltage
motor_efficiency = 0.95
inverter_efficiency = 0.95

# wire specs
RperL = 0.00096 #0.005208 #.003276  # (ohm/m) resistance of AWG 10 wire

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


ts = 0.1  # time step for Euler Integration
q_conv = 0.  # starting heat rate (q) out
T_insC = 20.1  # starting wire temperature (Celcius)
T_openC = 18.1  # starting wire temperature (Celcius)
T_insD = 20.1  # starting wire temperature (Celcius)
T_openD = 20.1  # starting wire temperature (Celcius)
T_ambient = 15.9
#--------------------------------------------------
HCO = 2.*(HC_cu + HC_tpe)*2.3
HCI = HCO + 50.
hI = 8.0  # (W/m^2K) insulated
hO = 24.  # open-air
#q_ins_out = 3. # heat leaked through insulation
# cumulative mission time breakpoints for altitude table lookup
mc = list(accumulate([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17]))

# initiate transient variable arrays
t_len = int(mc[-1] * 1./ts)
Time = np.zeros(t_len)
Temp = np.zeros(t_len)
Temp_openC = np.zeros(t_len)
Temp_insC = np.zeros(t_len)
Temp_openD = np.zeros(t_len)
Temp_insD = np.zeros(t_len)

# perform engineering calculations in a 'for' loop
# across the entire mission length updating state values for each time step
# (Euler Integration of Duct Temperature State)
for i,t in enumerate(np.arange(0, mc[-1], ts)):

    if t <= mc[0]:                  # up
        dep_power = 0.0  # kW
        cruise_power = 15.  # kW

    elif t > mc[0] and t <= mc[1]:  # Cruise Runup
        dep_power = 5.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[1] and t <= mc[2]:  # HLP Runup
        dep_power = 30.0  # kW
        cruise_power = 5.0  # kW

    elif t > mc[2] and t <= mc[3]:  # down
        dep_power = 10.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[3] and t <= mc[4]:  # Flight go/no-go
        dep_power = 0.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[4] and t <= mc[5]:  # up
        dep_power = 15.0  # kW
        cruise_power = 15.0  # kW

    elif t > mc[5] and t <= mc[6]:  # Ground roll
        dep_power = 30.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[6] and t <= mc[7]:  # Climb to 1500 feet
        dep_power = 15.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[7] and t <= mc[8]:  # Cruise Climb
        dep_power = 0.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[8] and t <= mc[9]:  # Cruise
        dep_power = 0.0  # kW
        cruise_power = 22.5  # kW

    elif t > mc[9] and t <= mc[10]:  # Descent to 1500 feet
        dep_power = 0.0  # kW
        cruise_power = 15.0  # kW

    elif t > mc[10] and t <= mc[11]:  # down
        dep_power = 5.0  # kW
        cruise_power = 5.0  # kW

    elif t > mc[11] and t <= mc[12]:  # Final approach
        dep_power = 15.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[12] and t <= mc[13]:  # up
        dep_power = 15.0  # kW
        cruise_power = 15.0  # kW

    elif t > mc[13] and t <= mc[14]:  # Go around to 1500 feet
        dep_power = 15.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[14] and t <= mc[15]:  # Approach pattern
        dep_power = 0.0  # kW
        cruise_power = 22.5  # kW

    elif t > mc[15] and t <= mc[16]:  # down
        dep_power = 5.0  # kW
        cruise_power = 5.0  # kW

    else:               # Final Approach
        dep_power = 15.0  # kW
        cruise_power = 0.0  # kW

    # CRUISE BUS
    cruise_bus_power = cruise_power/(motor_efficiency*inverter_efficiency)  # kW
    cruise_bus_current = 1000.*cruise_bus_power/bus_voltage  # ****
    q_prime_cruise = cruise_bus_current**2. * RperL

    q_prime_outoCI = circ_tpe * (T_insC-T_ambient) * hI
    if(t/60. > 10.):
        q_prime_outoCI = 3.7

    Ins_Temp_ROCC = ((2.*q_prime_cruise)-q_prime_outoCI)/HCI
    T_insC = T_insC + (ts*Ins_Temp_ROCC)

    q_prime_outoCO = circ_tpe * (T_openC-T_ambient) * hO

    Open_temp_ROCC = (2.*q_prime_cruise-q_prime_outoCO)/HCO
    T_openC = T_openC + (ts*Open_temp_ROCC)

    # DEP BUS
    # dep_bus_power = dep_power/(motor_efficiency*inverter_efficiency)  # kW
    # dep_bus_current = 1000*dep_bus_power/bus_voltage  # ****
    # q_prime_dep = dep_bus_current**2 * RperL

    # Ins_Temp_ROCD = 2*q_prime_dep/HC
    # T_insD = T_insD + (ts*Ins_Temp_ROCD)

    # q_prime_outoD = circ_tpe * (T_openD-T_ambient) * h

    # Open_temp_ROCD = (q_prime_dep-q_prime_outoD)/HC
    # T_openD = T_openD + (ts*Open_temp_ROCD)

    # save instantaneous states to array
    Time[i] = t
    Temp_openC[i] = T_openC
    Temp_insC[i] = T_insC
    Temp_openD[i] = T_openD
    Temp_insD[i] = T_insD


# Print Results
# print('Max Duct Temp: %f' % max(Duct_Temp))
plt.figure()

#plt.plot(Time,Temp, 'r', label = 'Wire')
plt.plot(Time/60.,Temp_openC, 'b', label = 'Open-air Cruise', lw=1.5)
plt.plot(Time/60.,Temp_insC, 'r', label = 'Insulated Cruise', lw=1.5)

#plt.plot(Time/60.,Temp_openD, 'c', label = 'Open-air DEP', lw=1.5)
#plt.plot(Time/60.,Temp_insD, 'm', label = 'Insulated DEP', lw=1.5)
# plt.plot(Time,Temp_cruise, 'r', label = 'Cruise Wires')
# plt.plot(Time,Temp_dep, 'b', label = 'DEP wires')
# plt.plot(Time,I_cruise, 'r', label = 'Cruise Wires')
# plt.plot(Time,I_dep, 'b', label = 'DEP wires')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel('Time (m)')
plt.ylabel('Wire Temperature (C)',fontsize=18)
plt.rcParams['xtick.labelsize'] = 24
# plt.ylabel('Current rating per conductor (Amp)')
plt.axvline(mc[1]/60., color='k', linestyle='--',lw=0.5)
plt.axvline(mc[2]/60., color='k', linestyle='--',lw=0.5)
plt.axvline(mc[3]/60., color='k', linestyle='--',lw=0.5)
# plt.text(250, 55,'Taxi', fontsize=8)
plt.axvline(mc[4]/60., color='k', linestyle='--',lw=0.5)
plt.axvline(mc[5]/60., color='k', linestyle='--',lw=0.5)
# plt.text(mc[2]-50, 72,'Take-Off', fontsize=8)
plt.axvline(mc[6]/60., color='k', linestyle='--',lw=0.5)
# plt.text(mc[6]+150, 40,'10kft Climb', fontsize=8)
plt.axvline(mc[7]/60., color='k', linestyle='--',lw=0.5)
# plt.text(mc[7]+50, 65,'Cruise', fontsize=8)
plt.axvline(mc[8]/60., color='k', linestyle='--',lw=0.5)
# plt.text(mc[8]+150, 60,'Descent', fontsize=8)
plt.axvline(mc[9]/60., color='k', linestyle='--',lw=0.5)
# plt.text(mc[9]+150, 60,'Go-Around', fontsize=8)
plt.axvline(mc[10]/60., color='k', linestyle='--',lw=0.5)
plt.axvline(mc[11]/60., color='k', linestyle='--',lw=0.5)
plt.axvline(mc[12]/60., color='k', linestyle='--',lw=0.5)
# plt.axhline(max(Duct_Temp), color='k', linestyle='--',lw=1)
# plt.axhline(78, color='g', linestyle='--',lw=1)
# plt.text(2400, 78,['Temp Limit: %2.0f' % 78], fontsize=16)
# plt.text(2400, max(Duct_Temp)+2,['Max Temp: %2.2f' % max(Duct_Temp)], fontsize=16)
plt.axvline(mc[13]/60., color='k', linestyle='--',lw=0.5)
plt.axvline(mc[14]/60., color='k', linestyle='--',lw=0.5)
plt.axvline(mc[15]/60., color='k', linestyle='--',lw=0.5)
plt.axvline(mc[16]/60., color='k', linestyle='--',lw=0.5)

import csv
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list

with open('Cal7c.csv') as f:
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

skip = 165
#deltaI = I2[2] - I4[2] #calibrate for different baseline temp
#deltaO = O2[2] - O4[2]
I2 = I2[skip:]
I4 = I4[skip:]
O2 = O2[skip:]
O4 = O4[skip:]
TestTime = TestTime[skip:] - float(skip)/60.

plt.plot(TestTime,I2, 'k', label = 'Test I2', lw=1.5)
plt.plot(TestTime,I4, 'k--', label = 'Test I4', lw=1.5)
plt.plot(TestTime,O2, 'g', label = 'Test O2', lw=1.5)
plt.plot(TestTime,O4, 'g--', label = 'Test O2', lw=1.5)

pylab.show()

# a = np.asarray([ Time, Temp, Duct_Temp ])
# np.savetxt("temp_data.csv", a, delimiter=",")

# output = np.column_stack((Time.flatten(),Temp.flatten(),Duct_Temp.flatten()))
# np.savetxt('temp_data.csv',output,delimiter=',')


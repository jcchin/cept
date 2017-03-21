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
#num_conduct = 6 #4
#bundle_derating = 0.5125
# num_conduct = 8  # 4 conducters per wire, *2 positive and return lines
# bundle_derating = 0.375
#num_bus = 2 # number of power busses

alt_derating = 0.925
d = 0.01905  # (m) duct diameter
circ_tpe = 0.040132 # jacket circumference
# cd = 2.05E-3 #2.59E-3
# AperStrand = num_conduct * num_bus * (np.pi*(cd)**2/4.)  # total Cu area
# Cp = 385.  # Copper Specific Heat
# rho = 8890.  # Copper Density
# HC = AperStrand*Cp*rho  # Heat Capacity of Copper Wire
HC = 240.
ts = 0.1  # time step for Euler Integration
hC = 3.3
hD = 3.3
#q_conv = 0.  # starting heat rate (q) out
#T_wire = 49.  # starting wire temperature
#T_duct = 49.  # starting duct temperature (Celcius)
T_cruise = 49.
T_dep = 49.
T_ambient = 49.
#--------------------------------------------------

# cumulative mission time breakpoints for altitude table lookup
mc = list(accumulate([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16]))

time =     [0, mc[5],   mc[6],      mc[7],      mc[8],   mc[9], mc[10],  mc[11],  mc[12], mc[13], mc[15]]
altitude = [0,     0, app_alt, cruise_alt, cruise_alt, app_alt,     0, app_alt, app_alt,      0,      0]  # m

#http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
alt_table = [-1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000] # meters
temp_table = [49., 49., 8.5, 2., -4.49, -10.98, -17.47, -23.96, -30.45] # Celsius (21.5 @ 0 in reality, using 49 to be conservative)
#gravity_table = [9.81,9.807,9.804,9.801,9.797,9.794,9.791,9.788,9.785]
#pressure_table = [11.39,10.13,8.988,7.95,7.012,6.166,5.405,4.722,4.111]
#kin_viscosity_table = [1.35189E-05,1.46041E-05,1.58094E-05,1.714E-05,1.86297E-05,2.02709E-05,2.21076E-05,2.4163E-05,2.64576E-05]

# http://www.engineeringtoolbox.com/air-properties-d_156.html
#temperature_table = [-100,-50,0,20,40,60,80,100,120,140,160,180,200]
#k_table = [0.016,0.0204,0.0243,0.0257,0.0271,0.0285,0.0299,0.0314,0.0328,0.0343,0.0358,0.0372,0.0386]
#Pr_table = [0.74,0.725,0.715,0.713,0.711,0.709,0.708,0.703,0.7,0.695,0.69,0.69,0.685]

#http://www.engineeringtoolbox.com/dry-air-properties-d_973.html
#temperature_alpha = [-73,-23,27,77,127]
#alpha_table = [0.00001017,0.00001567,0.00002207,0.00002918,0.00003694]

# initiate transient variable arrays
t_len = int(mc[-1] * 1./ts)
Time = np.zeros(t_len)
Temp_cruise = np.zeros(t_len)
Temp_dep = np.zeros(t_len)

# perform engineering calculations in a 'for' loop
# across the entire mission length updating state values for each time step
# (Euler Integration of Duct Temperature State)
for i,t in enumerate(np.arange(0, mc[-1], ts)):

    if t <= mc[0]:  # Taxi from NASA
        dep_power = 0.0  # kW
        cruise_power = 2.5  # kW

    elif t > mc[0] and t <= mc[1]:  # TO Checklist
        dep_power = 0.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[1] and t <= mc[2]:  # Cruise Runup
        dep_power = 0.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[2] and t <= mc[3]:  # HLP Runup
        dep_power = 30.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[3] and t <= mc[4]:  # Flight go/no-go
        dep_power = 0.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[4] and t <= mc[5]:  # Ground roll
        dep_power = 30.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[5] and t <= mc[6]:  # Climb to 1000 feet
        dep_power = 15.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[6] and t <= mc[7]:  # Cruise Climb
        dep_power = 0.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[7] and t <= mc[8]:  # CruiseW
        dep_power = 0.0  # kW
        cruise_power = 22.5  # kW

    elif t > mc[8] and t <= mc[9]:  # Descent to 1000 feet
        dep_power = 0.0  # kW
        cruise_power = 15.0  # kW

    elif t > mc[9] and t <= mc[10]:  # Final approach
        dep_power = 15.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[10] and t <= mc[11]:  # Go around to 1000 feet
        dep_power = 15.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[11] and t <= mc[12]:  # Approach pattern
        dep_power = 0.0  # kW
        cruise_power = 22.5  # kW

    elif t > mc[12] and t <= mc[13]:  # Final Approach
        dep_power = 15.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[13] and t <= mc[14]:  # Rollout and turnoff
        dep_power = 0.0  # kW
        cruise_power = 3.8  # kW

    #elif t>2955 and t<=3555:
    else:               # Taxi to NASA
        dep_power = 0.0  # kW
        cruise_power = 2.5  # kW

    # else:
    #     print('error')
    #     quit()
    alt = np.interp(t, time, altitude)
    T_ambient = 49.#np.interp(alt, alt_table, temp_table)
    #kinematic_viscosity = np.interp(alt, alt_table, kin_viscosity_table)
    #k_air = np.interp(T_ambient, temperature_table, k_table)
    #Pr = np.interp(T_ambient, temperature_table, Pr_table)

    # CRUISE BUS
    # cruise_bus_power = cruise_power/(motor_efficiency*inverter_efficiency)  # kW
    # cruise_bus_current = 1000*cruise_bus_power/bus_voltage  # ****
    # cruise_current_per_conductor = cruise_bus_current/(num_conduct)
    # cruise_current_rating_per_conductor = cruise_current_per_conductor/(bundle_derating*alt_derating)
    # q_prime_cruise = cruise_current_per_conductor**2 * RperL

    cruise_bus_power = cruise_power/(motor_efficiency*inverter_efficiency)  # kW
    cruise_bus_current = (1000.*cruise_bus_power/bus_voltage)/alt_derating  # ****
    q_prime_cruise = cruise_bus_current**2. * RperL


    # DEP BUS
    dep_bus_power = dep_power/(motor_efficiency*inverter_efficiency)  # kW
    dep_bus_current = (1000*dep_bus_power/bus_voltage)/alt_derating  # ****
    q_prime_dep = dep_bus_current**2 * RperL
    # Temp_ROC_dep = q_prime_dep/(HC)
    # T_wire_dep = T_wire_dep + (ts*Temp_ROC_dep)

    q_prime_outC = circ_tpe * (T_cruise-T_ambient) * hC
    q_prime_outD = circ_tpe * (T_dep-T_ambient) * hD

    Temp_ROCC = (q_prime_cruise - q_prime_outC) / HC
    Temp_ROCD = (q_prime_dep - q_prime_outD) / HC
    T_cruise = T_cruise + (ts*Temp_ROCC)
    T_dep = T_dep + (ts*Temp_ROCD)

    # save instantaneous states to array
    Time[i] = t
    #Temp[i] = T_wire
    #Duct_Temp[i] = T_duct
    Temp_cruise[i] = T_cruise
    Temp_dep[i] = T_dep

# Print Results

plt.figure()

plt.plot(Time,Temp_cruise, 'b', label = 'Cruise Ducts', lw=1.5)
plt.plot(Time,Temp_dep, 'g', label = 'DEP Ducts', lw=1.5)

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
plt.axhline(max(Temp_cruise), color='k', linestyle='--',lw=1)
plt.axhline(74, color='g', linestyle='--',lw=1)
plt.text(2400, 74,['Temp Limit: %2.0f' % 76], fontsize=16)
plt.text(2400, max(Temp_cruise),['Max Temp: %2.2f' % max(Temp_cruise)], fontsize=16)
pylab.show()

# a = np.asarray([ Time, Temp, Duct_Temp ])
# np.savetxt("temp_data.csv", a, delimiter=",")

# output = np.column_stack((Time.flatten(),Temp.flatten(),Duct_Temp.flatten()))
# np.savetxt('temp_data.csv',output,delimiter=',')


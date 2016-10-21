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

# Mission Inputs
app_alt = 304.8  # approach altitude, in meters (1000 ft)
cruise_alt = 3048.  # m, --> 10,000 ft

# mission time segments in seconds
m1 = 600  # Taxi from NASA
m2 = 120  # TAke off checklist
m3 = 60  # engine run up
m4 = 30  # flight go/no-go
m5 = 15  # ground role
m6 = 60  # climb to 1000'
m7 = 720  # cruise climb
m8 = 300  # cruise
m9 = 600  # descent to 1000'
m10 = 120  # final approach
m11 = 60  # go around to 1000'
m12 = 90  # approach pattern
m13 = 120  # final approach
m14 = 60  # role-out turn off
m15 = 600  # taxi to NASA

#carbon fiber specs
t_cf = 0.0006096  # (meters) carbon fiber duct thickness
k_cf = 21.  # (W/mK) carbon fiber thermal conductivity

# air properties
k_air = 0.0285  # stagnant air thermal conductivity
t_air = 0.003937  # air gap inside duct

bus_voltage = 416  # Operating Voltage
motor_efficiency = 0.95
inverter_efficiency = 0.95

# wire specs
RperL = .003276  # (ohm/m) resistance of AWG 10 wire
# num_conduct = 4
# bundle_derating = 0.5125
num_conduct = 8  # 4 conducters per wire, *2 positive and return lines
num_bus = 2 # number of power busses
bundle_derating = 0.375
alt_derating = 0.925
d = 0.01905  # (m) duct diameter
AperStrand = num_conduct * num_bus * (np.pi*(2.59E-3)**2/4.)  # total Cu area
Cp = 385.  # Copper Specific Heat
rho = 8890.  # Copper Density
HC = AperStrand*Cp*rho  # Heat Capacity of Copper Wire

ts = 0.1  # time step for Euler Integration
q_conv = 0.  # starting heat rate (q) out
T_wire = 49.  # starting wire temperature
T_duct = 49.  # starting duct temperature (Celcius)
#--------------------------------------------------

# cumulative mission time breakpoints for altitude table lookup
mc = list(accumulate([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15]))

time =     [0, mc[4],   mc[5],      mc[6],      mc[7],   mc[8], mc[9],  mc[10],  mc[11], mc[12], mc[14]]
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
Temp = np.zeros(t_len)
Temp_cruise = np.zeros(t_len)
Q_prime = np.zeros(t_len)
Cruise_current = np.zeros(t_len)
Duct_Temp = np.zeros(t_len)

i = 0  # counter

# perform engineering calculations in a 'for' loop
# across the entire mission length updating state values for each time step
# (Euler Integration of Duct Temperature State)
for t in np.arange(0, mc[-1], ts):

    if t <= mc[0]:  # Taxi from NASA
        dep_power = 0.0  # kW
        cruise_power = 2.5  # kW

    elif t > mc[0] and t <= mc[1]:  # TO Checklist
        dep_power = 0.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[1] and t <= mc[2]:  # Engine Runup
        dep_power = 30.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[2] and t <= mc[3]:  # Flight go/no-go
        dep_power = 0.0  # kW
        cruise_power = 0.0  # kW

    elif t > mc[3] and t <= mc[4]:  # Ground roll
        dep_power = 30.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[4] and t <= mc[5]:  # Climb to 1000 feet
        dep_power = 22.5  # kW
        cruise_power = 30.0  # kW

    elif t > mc[5] and t <= mc[6]:  # Cruise Climb
        dep_power = 0.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[6] and t <= mc[7]:  # CruiseW
        dep_power = 0.0  # kW
        cruise_power = 22.5  # kW

    elif t > mc[7] and t <= mc[8]:  # Descent to 1000 feet
        dep_power = 0.0  # kW
        cruise_power = 15.0  # kW

    elif t > mc[8] and t <= mc[9]:  # Final approach
        dep_power = 22.5  # kW
        cruise_power = 0.0  # kW

    elif t > mc[9] and t <= mc[10]:  # Go around to 1000 feet
        dep_power = 22.5  # kW
        cruise_power = 30.0  # kW

    elif t > mc[10] and t <= mc[11]:  # Approach pattern
        dep_power = 0.0  # kW
        cruise_power = 30.0  # kW

    elif t > mc[11] and t <= mc[12]:  # Final Approach
        dep_power = 22.5  # kW
        cruise_power = 0.0  # kW

    elif t > mc[12] and t <= mc[13]:  # Rollout and turnoff
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
    T_ambient = np.interp(alt, alt_table, temp_table)
    #kinematic_viscosity = np.interp(alt, alt_table, kin_viscosity_table)
    #k_air = np.interp(T_ambient, temperature_table, k_table)
    #Pr = np.interp(T_ambient, temperature_table, Pr_table)

    # CRUISE BUS
    cruise_bus_power = cruise_power/(motor_efficiency*inverter_efficiency)  # kW
    cruise_bus_current = 1000*cruise_bus_power/bus_voltage  # ****
    cruise_current_per_conductor = cruise_bus_current/num_conduct
    cruise_current_rating_per_conductor = cruise_current_per_conductor/(bundle_derating*alt_derating)

    q_prime_cruise = cruise_current_rating_per_conductor**2 * RperL
    # Temp_ROC_cruise = q_prime_cruise/HC
    # T_wire_cruise = T_wire_cruise + (ts*Temp_ROC_cruise)


    # DEP BUS
    dep_bus_power = dep_power/(motor_efficiency*inverter_efficiency)  # kW
    dep_bus_current = 1000*dep_bus_power/bus_voltage  # ****
    dep_current_per_conductor = dep_bus_current/num_conduct
    dep_current_rating_per_conductor = dep_current_per_conductor/(bundle_derating*alt_derating)

    q_prime_dep = dep_current_rating_per_conductor**2 * RperL
    # Temp_ROC_dep = q_prime_dep/HC
    # T_wire_dep = T_wire_dep + (ts*Temp_ROC_dep)

    q_prime_in = (8*(q_prime_cruise+q_prime_dep))


    Temp_ROC = q_prime_in/HC
    T_wire = T_wire + (ts*Temp_ROC)

    # r_cond_air = t_air/k_air
    # r_cond_cf = t_cf/k_cf

    # T_duct = T_wire - (total_q_prime*(r_cond_cf+r_cond_air))

    # T_film = ((T_duct+T_ambient)/2)+273 #K
    # alpha = np.interp(T_film, temperature_alpha, alpha_table)

    # Ra = (9.81*(1/T_film) * (T_wire- T_ambient) * d**3)/ (kinematic_viscosity*38.3E-6)

    # print(Ra, T_duct)

    # if Ra < 100:
    #     C = 1.02
    #     n = 0.148
    # elif Ra < 10000 and Ra >= 100:
    #     C = 0.850
    #     n = 0.188
    # elif Ra < 10**7 and Ra >= 10000:
    #     C = 0.480
    #     n = 0.250
    # else:
    #     C = 0.125
    #     n = 0.333

    # Nu = C* Ra ** n

    h = 7.0  # (W/m^2K) doesn't vary signficantly
    r_conv = 1/h  # (m^2 K / W)
    duct_circ = np.pi * d # duct_circumference (assume circuluar for worst case)
    q_prime_out = duct_circ * (T_duct-T_ambient) * h

    q_net = q_prime_in-q_prime_out

    Duct_temp_ROC = q_net/(HC)
    T_duct = T_duct + (ts*Duct_temp_ROC)

    # save instantaneous states to array
    Time[i] = t
    Temp[i] = T_wire
    Duct_Temp[i] = T_duct
    # Temp_cruise[i] = T_wire_cruise
    # Temp_dep[i] = T_wire_dep
    # Q_prime[i] = total_q_prime
    # Cruise_current[i] =cruise_current_rating_per_conductor

    i = i+1  # increment time

# Print Results
print('Max Duct Temp: %f' % max(Duct_Temp))
plt.figure()
plt.plot(Time,Temp, 'r', label = 'Wire temp w/o amb cooling')
plt.plot(Time,Duct_Temp, 'b', label = 'CFiber duct temp w amb cooling')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel('Time (s)')
plt.ylabel('Wire Temperature (C)')
plt.axvline(mc[1], color='k', linestyle='--',lw=0.5)
plt.axvline(mc[2], color='k', linestyle='--',lw=0.5)
plt.axvline(mc[3], color='k', linestyle='--',lw=0.5)
plt.text(250, 170,'Taxi', fontsize=8)
plt.axvline(mc[4], color='k', linestyle='--',lw=0.5)
plt.axvline(mc[5], color='k', linestyle='--',lw=0.5)
plt.text(mc[3]-50, 170,'Take-Off', fontsize=8)
plt.axvline(mc[6], color='k', linestyle='--',lw=0.5)
plt.text(mc[5]+150, 160,'10kft Climb', fontsize=8)
plt.axvline(mc[7], color='k', linestyle='--',lw=0.5)
plt.text(mc[6]+50, 170,'Cruise', fontsize=8)
plt.axvline(mc[8], color='k', linestyle='--',lw=0.5)
plt.text(mc[7]+150, 160,'Descent', fontsize=8)
plt.axvline(mc[9], color='k', linestyle='--',lw=0.5)
plt.text(mc[8]+150, 160,'Go-Around', fontsize=8)
plt.axvline(mc[10], color='k', linestyle='--',lw=0.5)
plt.axvline(mc[11], color='k', linestyle='--',lw=0.5)
plt.axvline(mc[12], color='k', linestyle='--',lw=0.5)
plt.axhline(max(Duct_Temp), color='k', linestyle='--',lw=0.5)
plt.text(3000, max(Duct_Temp)+2,['Max Temp: %f' % max(Duct_Temp)], fontsize=8)
pylab.show()

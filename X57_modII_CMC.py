#!/usr/bin/env python3
#Andrew Smith and Jeffrey Chin
#8/22/2017
#This script tracks bulk temperatures of an isothermal CMC heat sink for the X57 mod II
#Uses linearized flow coefficients for hydraulic resistance of JMX57 and CMC fin row
#Thermohydraulic model from "X57_inverter_aavid83249_V3.xmcd"
#CMC cooler model specific to high density extruded fin row per aavid catalog #83249-China

import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab
from itertools import accumulate
from scipy import interpolate
import scipy.integrate as integrate

#reference air properties ground - should try to use table values from film temperature, if possible
cp_air = 1005  # J/kg-C specific heat of air (average)
Pr = 0.71
k_air = 0.026

#cmc cooler mass props - source:  6061 properties at room temperature, 2x parallel aavid 83249 + spacer
cp_sink = 900  # J/kg-C #specific heat of 6061
m_cmc_sink = 2.62  # kg, mass of a single CMC fin row plus heat spreader
m_motor = 14.34  # kg, mass of motor
mcp_cmc_sink = m_cmc_sink*cp_sink  # thermal mass of a single CMC fin row + heat spreader
mcp_motor = m_motor*cp_sink

#conservative motor and CMC efficiencies- source: Joby Aviation
eff_motor = 0.96
eff_cmc = 0.94

#CMC fin row definition - from aavid catalog
nchan = 76  # number of channels per cooler
hchan = 0.022  # m, cmc channel height
lchan = .305  # m, cmc channel length
wchan = 1.727E-03  # m, cmc channel width
ARchan = wchan/hchan  # channel aspect ratio - for computing laminar Nusselt number
fd_chan = 0.04  # unitless, channel friction coefficient for turbulent darcy-weisbach solution
dh_cmc = 4*wchan*hchan/(2*(wchan + hchan))  # m, cmc cooler channel hydraulic diameter
A_conv_cmc = nchan*lchan*hchan  # m^2, total wetted area of a single cmc cooler
A_flow_cmc = nchan*wchan*hchan  # m^2, cmc cooler flow area
T_paste = 0.006  # degC/W

#flow characterization - combined linearized flow coeffieients for both fin rows - source: aavid 83245 + "x57_inverter_aavid83249_V3.xmcd"
cf_jmx57 = 5.334E-04  # m^4*s/kg
cf_cmc_finrow = 2/(6.719E-05)  # m^4*s/kg

cf_laminar = 3.405E-05  # s*m^4/kg,   coefficient to [inlet pressure]
cf_turb = 2.021E-03  # m^3.5/kg^0.5, coefficient to sqrt([inlet pressure])

T_init = 45  # hot day initial conditions
#T_init = 25 #cooler day initial conditions

ts = 10  # time step, s

import csv
from collections import defaultdict

columns = defaultdict(list)  # each value in each column is appended to a list

with open('mission.csv') as f:
    reader = csv.DictReader(f, delimiter=',',)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        for (k, v) in row.items():  # go over each column name and value
            columns[k].append(float(v))  # append the value into the appropriate list
                                 # based on column name k

mc = list(accumulate(columns['segmentDuration']))  # uncomment if using 'segment duration'
#mc = columns['elapsedTime']  # uncomment if using 'elapsedTime'
time = [0]
time.extend(mc[:-1])
altitude = columns['altitude']

Cruise_pwr = columns['Cruise_pwr']
U0 = columns['U0']

#US Standard Atmosphere 1976 calculator.  (https://www.digitaldutch.com/atmoscalc/index.htm) with a +29.55 deg C offset
alt_table = [-1000, -500, 0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5550, 6000, 6500, 7000]  # meters
temp_table = [51., 47.8, 41.3, 44.55, 38.05, 34.8, 31.55, 28.3, 25.05, 21.8, 18.55, 15.3, 12.05, 8.8, 5.55, 2.3, -0.95]  # Celsius (Using 40 C at ground to be conservative)
rho_table = [1.224, 1.16659, 1.0576, 1.111, 1.006, .956, .9088, 0.863, .819, 0.777, 0.736, 0.697, 0.659, 0.624, 0.589, 0.557, 0.525]  # density, kg/m^3
mu_table = [1.99E-05, 1.97E-05, 1.96E-05, 1.94E-05, 1.93E-05, 1.91E-05, 1.89E-05, 1.877E-05, 1.86E-05, 1.85E-05, 1.83E-05, 1.81E-05, 1.79E-05, 1.78E-05, 1.76E-05, 1.747E-05, 1.73E-05]  # dynamic viscosity, Pa*s
k_table = [.0279, .0277, .0274, 0.0274, .027, 0.0267, .0265, 0.0263, .0261, .0259, .0256, .0254, .0252, .0250, .0247, .02545, .0243]  # thermal conductivity, W/m-deg C


def lookup(x, schedule):  # choose the next closest index from the table, (rather than interpolate intermediate values) this allows for step values in power as prescribed in the mission profile.
    conds = []  # [x < mc[0]]
    for i, bp in enumerate(mc):
        conds.extend(((x > mc[i-1]) & (x < mc[i]),))
    return np.select(conds, schedule)

# initiate transient variable arrays
times = np.arange(0, mc[-1], ts)

solverTime = []
Ambient_Temp = []  # np.zeros(t_len)  # ambient temp --> per altitude model
Q_cmc_diss = []  # rate of heat dissipated by cmc
dT_CMC_plot = []  # history of CMC delta T
T_motor_exhaust = []  # motor exhaust temp
T_motor = []  # motor temperature
T_chip = []  # chip temperature

Q_m_diss = []  # rate of heat dissipated by motor

dT_Motor_Ex = []
dT_Motor = []
dT_Inlet = []
motor_flowrate = []
U0_plot = []

# states:
#0 = CMC_Temp    # CMC cooler (bulk) lumped-cap temperature
#1 = CMC_Temp_Ad  # adiabatic cmc temperature (for high power transient studies) (*not used)
#2 = Inlet_Temp      # nacelle inlet air temperature (*not used)
#3 = Motor_Ex_Temp   # JMX57 exhaust air temperature
#4 = Motor_Temp     # JMX57 Motor temp
def dxdt(temp, t):

    cruise_power = lookup(t, Cruise_pwr)
    alt = np.interp(t, time, altitude) + 700
    T_0 = np.interp(alt, alt_table, temp_table) + 8.
    #T_0 = interpolate.splrep(alt_table, temp_table, s=0)
    u_0 = lookup(t, U0)
    if (u_0 < 1.):
        u_0 = 20.
    #u_0 = 54
    mu_0 = np.interp(alt, alt_table, mu_table) #dynamic viscosity, Pa*s
    rho_0 = np.interp(alt, alt_table, rho_table)
    k_0 = np.interp(alt, alt_table, k_table)


    #module convection model - adjacent plates
    #Tfilm = 0.5*(temp[0]+temp[2]+273.)  # guesstimate of module freeconv film temperature - turn on if adjusting air properties inside cooling channels

    #evaluate cmc convection per channel model
    ################
    Pmotor = 0.5*1000.*cruise_power  # total power requested by a single cruise propulsor
    #Pmotor = 50 #kw, debug value
    Q_m = (1.0 - eff_motor)*Pmotor  # instantaneous heat dissipated by a single JMX57 stator fin row
    Q_cmc = 0.5*(1.0-eff_cmc)*Pmotor/eff_motor
    Q_motor = 25.*(temp[4]-T_0)

    #hydraulic solution  - flow through mod II nacelle using flow coeff's cf_laminar and cf_turb
    dP = 0.5*rho_0*u_0*u_0

    volratelam = dP*cf_laminar
    #volratelam = 0.041 #debug value at 1200 Pa
    volrateturb = (dP**0.5)*cf_turb
    #cmc channel convection model - high AR laminar channels
    #Tfilm = 0.5*(temp[0]+temp[2])  # guesstimate of cmc fin row intermediate temperature (for evaluating fluid props)
    mdot = rho_0*max([volratelam,volrateturb])
    #T_motor_ex = T_0 + (Q_m/(mdot*cp_air))
    #T_motor_ex = 60 #debug value from R christie study
    T_motor_ex = T_0 + (Q_motor)/(mdot*cp_air)
    u_chan_lam = volratelam/(2*nchan*A_flow_cmc)
    u_chan_turb = volrateturb/(2*nchan*A_flow_cmc)  # turbulent channel solution

    Re_lam_cmc = rho_0*u_chan_lam*dh_cmc/mu_0
    Re_turb_cmc = rho_0*u_chan_turb*dh_cmc/mu_0
    Nu_lam_cmc = 8.235*(1 - 2.042*ARchan +3.085*ARchan**2 -2.477*ARchan**3 + 1.058*ARchan**4 - 0.186*ARchan**5)  #laminar Nu for a high aspect ratio channel
    Nu_turb_cmc = ((fd_chan/8)*(Re_turb_cmc-1000))*Pr / (1+12.7*((Pr**0.66) - 1)*(fd_chan/8)**0.5)  # turbulent Nu per channel


    hconv_cmc_lam = Nu_lam_cmc*k_0/dh_cmc
    U_conv_cmc_lam = hconv_cmc_lam*A_conv_cmc

    hconv_cmc_turb = Nu_turb_cmc*k_0/dh_cmc
    U_conv_cmc_turb = hconv_cmc_turb*A_conv_cmc


    U_conv_cmc = max(U_conv_cmc_lam, U_conv_cmc_turb)  # pick bigger HTC

    #convection rates from module and cabin from current time step T's
    Q_conv_CMC = U_conv_cmc*(temp[0]-T_motor_ex)
    #################


    #temperature corrections
    dT_Inlet = 0  # Nacelle inlet temperature - T_0
    dT_CMC = (Q_cmc - Q_conv_CMC)/mcp_cmc_sink  # deg C/s,   CMC temperature change - balance between convection and heat generation
    dT_CMC_Ad = (Q_cmc)*ts/mcp_cmc_sink
    dT_CMC_Ad = 1  # not used...
    dT_Motor_Ex = 0.5*cruise_power*eff_motor/(rho_0*cp_air*volratelam)  # jmx57 heat addition to flowpath
    dT_Motor = (Q_m - Q_motor)/mcp_motor

    # save off other useful info (not integrated states)
    # note: the time breakpoints are "solverTime", not "times"
    #       to allow for solver adaptive time stepping

    solverTime.extend([t])
    motor_flowrate.extend([volratelam])
    Ambient_Temp.extend([T_0])
    dT_CMC_plot.extend([dT_CMC])
    T_motor_exhaust.extend([T_motor_ex])
    T_motor.extend([temp[4]])
    T_chip.extend([temp[0] + (T_paste * Q_cmc)])
    Q_cmc_diss.extend([Q_cmc])
    Q_m_diss.extend([Q_m])
    U0_plot.extend([u_0])

    return [dT_CMC, dT_CMC_Ad, dT_Inlet, dT_Motor_Ex, dT_Motor]

#Integrate
Temp_states = integrate.odeint(dxdt, [T_init,T_init,T_init,T_init,T_init], times, hmax=1.)


# Print Results
print('#########Peak Heat Dissipation##########')
print('Max CMC dissipated power: %f W' % np.max(Q_cmc_diss))
print('Max Motor dissipated power: %f W' % np.max(Q_m_diss))
print()
print('#########Flow Rates ##########')
print('Average motor air ingestion rate: %f m^3/s' % np.mean(motor_flowrate))
print('Peak motor air ingestion rate: %f m^3/s' % np.max(motor_flowrate))
print()
print('#########Motor Exhaust Temps##########')
print('Max JMX57 Motor Temperature: %f deg C' % np.max(T_motor))
print('Average JMX57 Exhaust Temperature: %f deg C' % np.mean(T_motor_exhaust))
print('Max JMX57 Exhaust Temperature: %f deg C' % np.max(T_motor_exhaust))
print()
print('#########CMC Sink Performance##########')
print('Average CMC Isothermal Sink Temperature: %f deg C' % np.mean(Temp_states[:, 0]))
print('Average CMC Chip Temperature: %f deg C' % np.mean(T_chip))
print('Max CMC Isothermal Sink Temperature: %f deg C' % np.max(Temp_states[:, 0]))
print('Max CMC Chip Temperature: %f deg C' % np.max(T_chip))



fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax2 = fig1.add_subplot(111)
ax3 = fig1.add_subplot(111)
#ax4 = ax2.twinx()

#plot component/flowpath temps
#ax1.plot(solverTime, Q_m_diss, 'r-', label='motor Qdiss (W)')
#ax1.plot(solverTime, T_motor_exhaust, 'r-', label='motor exhaust temp (deg C)')
#ax1.plot(solverTime, motor_flowrate, 'g-', label='motor flow rate (m^3/s)')
#ax1.plot(solverTime, U0_plot, 'g-', label='free stream velocity')
ax1.plot(solverTime, T_chip, 'r-', label='CMC chip Temp')
#ax2.plot(solverTime, Ambient_Temp, 'g-', label='Ambient Temperature')
ax2.plot(solverTime, T_motor_exhaust, 'g-', label='Motor Exhaust Temp')
#ax3.plot(solverTime, T_motor_exhaust, 'b-', label='Motor Exhaust Temp')
ax3.plot(solverTime, T_motor, 'b-', label='Motor Temp')

#ax2.plot(solverTime, Q_m_diss, 'k-', label='motor qdiss')


ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Temp (C)')
#ax1.set_ylabel('CMC sink Temperature (deg C)', color='k')
#ax4.set_ylabel('Power (W)', color='k')



legend = ax1.legend(loc='upper left', shadow=True)
legend = ax2.legend(loc='upper left', shadow=True)
legend = ax3.legend(loc='upper left', shadow=True)

#legend = ax4.legend(loc='upper center', shadow=True)

#plt.show()

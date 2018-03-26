#!/usr/bin/env python3
#Jeffrey Chin

#This script tracks the bulk temperature of the JMX57 Cruise Motor during dyno testing

import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab
from itertools import accumulate
from scipy import interpolate
import scipy.integrate as integrate

ts = 0.5  # time step, s
#event start  1    2     3    4    5    6   end
t_bp    = [0, 20, 150,  200, 450,  400,  300,  350]
pwr_bp  = [0,  0,  60,   60,   0,   60,   60,   60]
mdot_bp = [0,  0,   0,  0.4, 0.4,  0.4,  0.4,  0.4]
motor_rpm=[0,  0,   0,    0,   0,    0, 1500, 1500]

mc = list(accumulate(t_bp))

#reference air properties ground - should try to use table values from film temperature, if possible
cp_air = 1005  # J/kg-C specific heat of air (average)
Pr = 0.71
k_air = 0.026
rho_0 = 1.224  # kg/m^3

# motor properties
m_motor = 14.34  # kg, mass of motor
cp_motor = 900  # J/kg-C #specific heat of 6061
mcp_motor = m_motor*cp_motor
d_inner = 0.352  # m motor inner diam
d_outer = 0.371  # motor outer diameter
motor_area = 0.009  # m^3
eff_motor = 0.92


def lookup(x, schedule):  # choose the next closest index from the table, (rather than interpolate intermediate values) this allows for step values in power as prescribed in the mission profile.
    conds = []  # [x < mc[0]]
    for i, bp in enumerate(mc):
        conds.extend(((x > mc[i-1]) & (x < mc[i]),))
    return np.select(conds, schedule)

times = np.arange(0, mc[-1], ts)
solverTime = []
T_motor = []  # motor temperature
Q_m_diss = []  # rate of heat dissipated by motor
dT_Motor = []
motor_flowrate = []
pwrs = []
rpms = []
T0 = 22  # degree C

cf_laminar = 3.85E-04  # 3.405E-05  # s*m^4/kg,   coefficient to [inlet pressure]
cf_turb = 2.021E-03  # m^3.5/kg^0.5, coefficient to sqrt([inlet pressure])
u_0 = 20  # m/s


# states:
#0 = Motor_Temp     # JMX57 Motor temp
#1 = Motor_Ex_Temp   # JMX57 exhaust air temperature
def dxdt(temp, t):

    pwr = lookup(t, pwr_bp)
    mdot = lookup(t, mdot_bp)
    rpm = lookup(t, motor_rpm)
    Q_m = (1.0 - eff_motor)*pwr*1000 + rpm  # instantaneous heat dissipated by a single JMX57 stator fin row
    Q_motor = mdot*200.*(temp[0]-T0)

    #hydraulic solution  - flow through motor using flow coeff's cf_laminar and cf_turb
    dP = 0.5*rho_0*u_0*u_0
    volratelam = dP*cf_laminar
    volrateturb = (dP**0.5)*cf_turb
    #mdot = rho_0*max([volratelam, volrateturb])
    dT_Motor = (Q_m - Q_motor)/mcp_motor

    solverTime.extend([t])
    motor_flowrate.extend([mdot])
    T_motor.extend([temp[0]])
    Q_m_diss.extend([Q_m])
    pwrs.extend([pwr])
    rpms.extend([rpm])

    return [dT_Motor]

#Integrate
Temp_states = integrate.odeint(dxdt, [T0], times, hmax=0.5)

# Print Results
print('Total Run Time: ', mc[-1]/60, ' minutes')
print('#########Peak Heat Dissipation##########')
print('Max Motor dissipated power: %f W' % np.max(Q_m_diss))
print()
print('#########Flow Rates ##########')
# print('Average motor air ingestion rate: %f m^3/s' % np.mean(motor_flowrate))
# print('Peak motor air ingestion rate: %f m^3/s' % np.max(motor_flowrate))
print()
print('#########Motor Exhaust Temps##########')
print('Max JMX57 Motor Temperature: %f deg C' % np.max(T_motor))
# print('Average JMX57 Exhaust Temperature: %f deg C' % np.mean(T_motor_exhaust))
# print('Max JMX57 Exhaust Temperature: %f deg C' % np.max(T_motor_exhaust))


fig1 = plt.figure()
ax1 = fig1.add_subplot(411)
ax2 = fig1.add_subplot(412)
ax3 = fig1.add_subplot(413)
ax4 = fig1.add_subplot(414)
#ax4 = ax2.twinx()

#plot component/flowpath temps
ax1.plot(solverTime, pwrs, 'g-')
ax2.plot(solverTime, motor_flowrate, 'b-')
ax3.plot(solverTime, rpms, 'k-')
ax4.plot(solverTime, T_motor, 'r-')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('pwr (kW)')
ax2.set_ylabel('cooling (kg/s)')
ax3.set_ylabel('RPM')
ax4.set_ylabel('Temp (C)')
#plt.ylim([10,120])

for segments in mc:
    ax1.axvline(segments, color='k', linestyle='--',lw=0.5)
    ax2.axvline(segments, color='k', linestyle='--',lw=0.5)
    ax3.axvline(segments, color='k', linestyle='--',lw=0.5)
    ax4.axvline(segments, color='k', linestyle='--',lw=0.5)



# legend = ax1.legend(loc='upper left', shadow=True)
# legend = ax2.legend(loc='upper left', shadow=True)
# legend = ax3.legend(loc='upper left', shadow=True)

#legend = ax4.legend(loc='upper center', shadow=True)

plt.show()



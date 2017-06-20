#!/usr/bin/env python3

#tracks lumped capacitance temperatures of battery module and X57 cabin
#module-cabin convection modeled as closeley spaced vertical flat plates
#cabin-atmosphere convection assumed htc.

import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab
from itertools import accumulate
import scipy.integrate as integrate

N_module = 12.  # number of battery modules
N_cell_module = 160.   # number of cells per module
eff_cell = 0.977  # cell discharge efficiency.  Qgen=(1-eff_cell)*P

cabin_volume = 3.7  # m^3, tecnam P2006T internal volume (approximate)
cabin_mass = 100  # kg, gross estimate of participating cabin structure/skin mass

A_conv_module = .068  # m^2
h_conv_module = 25  # W/m^2-C

A_conv_cabin = 5  # m^2 - estimate of available cabin surface area for external convection
h_conv_cabin = 50  # W/m^2-C estimate of cabin external convection coefficient (computed for Lc = 0.8m, u_0=50 m/s)
cabin_height = 2.8  # m
cabin_length = 8.5  # m

#free convection parameters

s_module = 0.015  # m, module spacing (distance between vertical surfaces)
h_module = 0.3  # m module height
Lc_module = h_module  # placeholder until we get better module dims
rho_0 = 1.055  # kg/m^3
mu_air = 1.81E-05  # Pa-s, viscosity of ambient air
k_air = 0.026  # W/m-C, therm. conductivity of air

k_wall_module = 200  # W/m-C module wall conductivity (Al 6061)
k_wall_cabin = 200  # W/m-C cabin wall conductivity (Al 6061)
t_wall_module = .003  # m module wall thickness
t_wall_cabin = .003  # m cabin wall thickness
U_conv_module = A_conv_module/((1/h_conv_module)+(t_wall_module/k_wall_module))  # W/C total heat transfer coefficient from module to cabin
#U_conv_cabin = (1/N_module)*A_conv_module/((1/h_conv_cabin)+(t_wall_cabin/k_wall_module)) #W/C  total heat transfer coefficient from cabin to ambient
U_conv_module = 5.*A_conv_module  # debug value 5 W/m^2-C
U_conv_cabin = A_conv_cabin*25.  # debug value 25 W/m^2-C


cp_air = 1005  # J/kg-C specific heat of air (average)
cp_cells = 890  # J/kg-C #specific heat of 18650 cells
cp_grid = 900  # J/kg-C specific heat of aluminum grid and external plate
m_cabin_air = 1.15*cabin_volume/N_module  # m^3
m_module_grid = .016205*N_cell_module  # kg, grid and external plate area per 160 cell module.
m_cell = .045*N_cell_module  # kg mass of cells per 160 cell module

mcp_module = (m_cell*cp_grid)+(m_module_grid*cp_cells)
mcp_cabin_air = m_cabin_air*cp_air
mcp_cabin = cabin_mass*cp_grid

T_0 = 35  # C module and aircraft equilibrium temperature at start (HOT DAY)

ts = 0.5  # time step, s

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
Cruise_pwr = columns['Cruise_pwr']
U0 = columns['U0']

#mc = list(accumulate([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16]))
# time       = [       0,    mc[1],    mc[2],    mc[3],    mc[4],    mc[5],   mc[6],   mc[7],      mc[8],      mc[9],  mc[10], mc[11],  mc[12],   mc[13],   mc[14],  mc[15]]
# altitude   = [EAFB_alt, EAFB_alt, EAFB_alt, EAFB_alt, EAFB_alt, EAFB_alt, app_alt, cruise_alt, cruise_alt, app_alt,     0., app_alt, app_alt, EAFB_alt, EAFB_alt, EAFB_alt]  # m
# DEP_pwr    = [      0.,       0.,       0.,      30.,       0.,      30.,     15.,      0.,         0.,         0.,     15.,    15.,      0.,      15.,       0.,      0.]
# Cruise_pwr = [     10.,      0.1,     120.,      0.1,       0.,     120.,    120.,    120.,        90.,        60.,      0.,   120.,     90.,       0.,      15.,     10.]
# U0         = [     20.,      20.,      20.,      20.,      20.,      20.,     40.,     54.,        54.,        54.,     54.,    54.,     54.,      54.,      20.,     10.]

#http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
alt_table = [-1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000] # meters
temp_table = [49., 49., 8.5, 2., -4.49, -10.98, -17.47, -23.96, -30.45] # Celsius (21.5 @ 0 in reality, using 49 to be conservative)

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

Q_cruise = []

Q_cool = []  # module cooling rate log
Q_net = []  # module heat accumulation log

# states:
#0 = Module_Temp     # battery module (bulk) temperature
#1 = Module_Temp_Ad  # battery module temperature assuming adiabatic cabin
#2 = Cabin_Temp      # cabin bulk temperature
#3 = Cabin_Temp_Ad   # adiabatic cabin bulk temperature
def dxdt(temp, t):

    cruise_power = lookup(t, Cruise_pwr)
    dep_power = lookup(t, DEP_pwr)
    alt = np.interp(t, time, altitude)
    T_0 = np.interp(alt, alt_table, temp_table)

    Pmotor = 1000.*cruise_power  # total power requested by propulsion
    Qdiss = (1.-eff_cell)*Pmotor/N_module  # instantaneous heat dissipated by single module
    Qmisc = 1000.  # W, miscellaneous heat loads in aircraft cabin (pilot, avionics, etc)

    #(Need to update tropo altitude model!)
    T_ground = 35.  # deg C, air temperature at ground
    T_0 = T_ground - .00649*alt  # #deg C, standard tropospheric lapse rate from T_ground
    T_0 = 35.  # deg C, debug temperature
    #tracking time, sink isothermal temp, sink adiabatic temp (debug) and inlet air temp as func of altitude.

    #module convection model - adjacent plates
    Tfilm = 0.5*(temp[0]+temp[2]+273.)  # guesstimate of module freeconv film temperature
    Beta = 1./Tfilm
    Ra_module = (rho_0*9.81*Beta*cp_air*(s_module**4)*(temp[0]-temp[2])+.01)/(mu_air*k_air*h_module)
    Nu_module = (Ra_module/24.)*((1-np.exp(-35./Ra_module))**0.75)

    hconv_module = Nu_module*k_air/Lc_module  # free convection hconv
    A_conv_module = Lc_module**2  # free convection featureless surface

    #hconv_module = 50 #forced convection hconv, given sink design
    #A_conv_module = .06 #m^2, finned surface with n=10, 1cm X 30 cm fins

    U_conv_module = hconv_module*A_conv_module

    #convection rates from module and cabin from current time step T's
    Qconv_cabin = -U_conv_cabin*ts*(temp[2]-T_0)
    Qconv_module = -U_conv_module*(temp[0]-temp[2])
    Qconv_module_Ad = -U_conv_module*(temp[1]-temp[3])  # for adiabatic cabin

    #temperature corrections
    dT_cabin = (Qmisc - N_module*Qconv_module + Qconv_cabin)*ts/(mcp_cabin_air+mcp_cabin)  # cabin with module and avionics heat load
    dT_module = (Qdiss + Qconv_module)*ts/mcp_module  # module heat loss to convection
    dT_module_Ad = (Qdiss + Qconv_module_Ad)*ts/mcp_module
    dT_cabin_Ad = (Qmisc - N_module*Qconv_module_Ad)*ts/(mcp_cabin_air+mcp_cabin)

    # save off other useful info (not integrated states)
    # note: the time breakpoints are "solverTime", not "times"
    #       to allow forsolver adaptive time stepping
    solverTime.extend([t])
    Ambient_Temp.extend([T_0])
    dTdt.extend([dT_module])
    Module_Hconv.extend([hconv_module])
    Module_Ra.extend([Ra_module])
    Module_Nusselt.extend([Nu_module])
    Delta_T.extend([temp[0] - temp[2]])
    Q_gen.extend([Qdiss - Qconv_module])

    return [dT_module, dT_module_Ad, dT_cabin, dT_cabin_Ad]

#Integrate
Temp_states = integrate.odeint(dxdt, [T_0,T_0,T_0,T_0], times, hmax=1.)

# Print Results
print('Max Module Temp: %f deg C' % max(Temp_states[:,0]))
print('Battery system thermal mass: %f deg C/J' % (12.*mcp_module))
print('Cabin air thermal mass: %f deg C/J' % mcp_cabin_air)
print('Cabin participating structure thermal mass: %f deg C/J' % mcp_cabin)
print('Module overall HTC: %f W/deg C' % U_conv_module)
print('Cabin overall HTC: %f W/deg C' % U_conv_cabin)
print('Average module heat absorption: %f W' % np.mean(Q_gen))
print('Peak module heat absorption: %f W' % np.max(Q_gen))
print('Average module-cabin delta-T: %f deg C' % np.mean(Delta_T))
print('Minimum module-cabin delta_T: %f deg C' % np.min(Delta_T))
print('Maximum module-cabin delta_T: %f deg C' % np.max(Delta_T))


fig1 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig1.add_subplot(111)
ax3 = fig1.add_subplot(111)
ax4 = ax2.twinx()

ax1.plot(times, Temp_states[:,0], 'r-', label='Module Temperature')
# ax2.plot(times, Temp_states[:,2], 'g-', label='Cabin Temperature')
# ax3.plot(solverTime, Ambient_Temp, 'b-', label='Ambient Temperature')
ax4.plot(solverTime, Q_gen, 'g-', label='Module Absorbed Heat (W)')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Temperature (deg C)', color='k')
ax4.set_ylabel('Module absorbed heat rate(W)', color='k')

legend = ax1.legend(loc='upper center', shadow=True)
#legend = ax2.legend(loc='upper center', shadow=True)
legend = ax4.legend(loc='upper center', shadow=True)

plt.show()

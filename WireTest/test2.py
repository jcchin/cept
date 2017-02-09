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
current = 72.  # amps
duration = 3000.# 17.*60.  # seconds

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


ts = 0.1  # time step for Euler Integration
q_conv = 0.  # starting heat rate (q) out
T_cu = 24.  # starting copper temperature
T_ins = 24.  # starting wire temperature (Celcius)
T_open = 24.  # starting wire temperature (Celcius)
T_ambient = 24.
#--------------------------------------------------

# initiate transient variable arrays
t_len = int(duration/ts)

Time = np.zeros(t_len)
Temp = np.zeros(t_len)
Temp_open = np.zeros(t_len)
Temp_ins = np.zeros(t_len)
Q_prime = np.zeros(t_len)


# (Euler Integration of Duct Temperature State)
# all calculations are per length (ohm/m, W/m, etc..)
for i,t in enumerate(np.arange(0, duration, ts)):

    q_prime_in = current**2 * RperL

    HC = (HC_cu + HC_tpe)
    Temp_ROC = q_prime_in/HC
    T_ins = T_ins + (ts*Temp_ROC)

    h = 7.0  # (W/m^2K) doesn't vary signficantly
    q_prime_out = circ_tpe * (T_open-T_ambient) * h

    q_net = q_prime_in-q_prime_out

    Open_temp_ROC = q_net/(HC)
    T_open = T_open + (ts*Open_temp_ROC)

    # save instantaneous states to array
    Time[i] = t
    Temp_open[i] = T_open
    Temp_ins[i] = T_ins


# Print Results
plt.figure()
plt.plot(Time,Temp_ins, 'r', label = 'Insulated', lw=1.5)
plt.plot(Time,Temp_open, 'g', label = 'Open', lw=1.5)
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel('Time (s)')
plt.ylabel('Wire Temperature (C)',fontsize=18)
plt.rcParams['xtick.labelsize'] = 24
pylab.show()

output = np.column_stack((Time.flatten(),Temp.flatten(),Wire_Temp.flatten()))
np.savetxt('temp_data.csv',output,delimiter=',')


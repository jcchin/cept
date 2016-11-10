

# 1. Test fit cell cp and efficiency to match EPS data for 60 cell block
# 2. Use generated cell heating model with X-57 current schedule, scaling down
# 	 from full battery system to 60 cell block
# 3. Determine steady cooler requirement to keep batteries below critical temp: model 
# 	 first as constant Q (heat pump, simpler to model), then as Qconv ~ h*(Tbatt-Tenv).

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
Cp_cell = 1.02 # J/(gm*K)
eff_cell = 0.05
N_cell = 60
T_0 = 303 # K
T_sink = 323 #K
dT = 10 #K

I_high = 275 # amps
I_low = 150 # amps
V_batt = 10 # volts

Q_high = I_high * V_batt
Q_low = I_low * V_batt

N_scale = 32 #???

ts = 0.5

mc = list(accumulate([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16]))

t_len = int(mc[-1] * 1./ts)
Time = np.zeros(t_len)
Temp = np.zeros(t_len)
Current = np.zeros(t_len)

i = 0

for t in np.arange(0, mc[-1], ts):

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


	I_X57 = 4*(cruise_power + dep_power)/(V_batt*N_scale)

	Q = eff_cell * I_X57 * V_batt

	T_batt = T_0 + Q*ts/(N_cell*mass_cell*Cp_cell)

	Time[i] = t
	Temp[i] = T_batt
	Current[i] = I_X57

	T_0 = T_batt

	i = i + 1

    # Print Results


print('Max Temp: %f' % max(Temp))
plt.figure()

plt.plot(Time,Temp, 'r', label = 'Wire')

plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel('Time (s)')
plt.ylabel('Battery Pack Temperature (C)')
# plt.ylabel('Current rating per conductor (Amp)')
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
plt.axhline(max(Temp), color='k', linestyle='--',lw=0.5)
plt.text(3000, max(Temp)+2,['Max Temp: %f' % max(Temp)], fontsize=8)
pylab.show()
a = np.asarray([ Time, Temp, Duct_Temp ])
np.savetxt("temp_data.csv", a, delimiter=",")

output = np.column_stack((Time.flatten(),Temp.flatten(),Duct_Temp.flatten()))
np.savetxt('temp_data.csv',output,delimiter=',')






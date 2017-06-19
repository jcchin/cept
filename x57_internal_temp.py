
#tracks lumped capacitance temperatures of battery module and X57 cabin
#module-cabin convection modeled as closeley spaced vertical flat plates
#cabin-atmosphere convection assumed htc. 

import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab
from itertools import accumulate

# Mission Inputs
app_alt = 457.2  # approach altitude, in meters (1500 ft)
cruise_alt = 2438.4  # m, --> 8,000 ft
EAFB_alt = 702 #m, ground level


# mission time segments in seconds
m1 = 600  # Taxi from NASA
m2 = 120  # Take off checklist
m3 = 30  # Cruise Runup
m4 = 30  # HLP Runup
m5 = 30  # flight go/no-go
m6 = 10  # ground roll
m7 = 90  # climb to 1000'
m8 = 540  # cruise climb
m9 = 300  # cruise
m10 = 450  # descent to 1500'
m11 = 180  # final approach
m12 = 90  # go around to 1500'
m13 = 90  # approach pattern
m14 = 180  # final approach
m15 = 60  # roll-out turn off
m16 = 600  # taxi to NASA

N_module = 12 #number of battery modules
N_cell_module = 160
eff_cell = 0.977  #cell discharge efficiency.  Qgen=(1-eff_cell)*P

cabin_volume = 3.7 #m^3, tecnam P2006T internal volume (approximate)
cabin_mass = 100 #kg, gross estimate of participating cabin structure/skin mass

A_conv_module = .068 #m^2
h_conv_module = 25 #W/m^2-C

A_conv_cabin = 5 #m^2 - estimate of available cabin surface area for external convection
h_conv_cabin = 50 #W/m^2-C estimate of cabin external convection coefficient (computed for Lc = 0.8m, u_0=50 m/s)
cabin_height = 2.8 #m
cabin_length = 8.5 #m


#free convection parameters

s_module = 0.015 #m, module spacing (distance between vertical surfaces)
h_module = 0.3 #m module height
Lc_module = h_module #placeholder until we get better module dims
rho_0 = 1.055 #kg/m^3
mu_air = 1.81E-05 #Pa-s, viscosity of ambient air
k_air = 0.026 #W/m-C, therm. conductivity of air

k_wall_module = 200 #W/m-C module wall conductivity (Al 6061)
k_wall_cabin = 200 #W/m-C cabin wall conductivity (Al 6061)
t_wall_module = .003 #m module wall thickness
t_wall_cabin = .003 #m cabin wall thickness
U_conv_module = A_conv_module/((1/h_conv_module)+(t_wall_module/k_wall_module)) #W/C total heat transfer coefficient from module to cabin
#U_conv_cabin = (1/12)*A_conv_module/((1/h_conv_cabin)+(t_wall_cabin/k_wall_module)) #W/C  total heat transfer coefficient from cabin to ambient
U_conv_module = 5*A_conv_module #debug value 5 W/m^2-C
U_conv_cabin = A_conv_cabin*25 #debug value 25 W/m^2-C



cp_air = 1005 #J/kg-C specific heat of air (average)
cp_cells = 890 #J/kg-C #specific heat of 18650 cells
cp_grid = 900 #J/kg-C specific heat of aluminum grid and external plate
m_cabin_air = 1.15*cabin_volume/12 #m^3
m_module_grid = .016205*N_cell_module #kg, grid and external plate area per 160 cell module. 
m_cell = .045*N_cell_module #kg mass of cells per 160 cell module

mcp_module = (m_cell*cp_grid)+(m_module_grid*cp_cells)
mcp_cabin_air = m_cabin_air*cp_air
mcp_cabin = cabin_mass*cp_grid

T_0 = 35 # C module and aircraft equilibrium temperature at start (HOT DAY)



ts = 0.5 # time step, s

mc = list(accumulate([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16]))

t_len = int(mc[-1] * 1./ts) + 1
Time = np.zeros(t_len)

Module_Temp = np.zeros(t_len) #battery module (bulk) temperature
Module_Temp_Ad = np.zeros(t_len) #battery module temperature assuming adiabatic cabin
Cabin_Temp = np.zeros(t_len) #cabin bulk temperature
Cabin_Temp_Ad = np.zeros(t_len)  #adiabatic cabin bulk temperature
Ambient_Temp = np.zeros(t_len) #ambient temp --> per altitude model

Q_gen = np.zeros(t_len) #rate of heat absorbed by battery module
dTdt = np.zeros(t_len) #rate of battery module temperature change
Module_Hconv = np.zeros(t_len) #module convective heat transfer coefficient
Module_Ra = np.zeros(t_len)
Module_Nusselt = np.zeros(t_len) 
Delta_T = np.zeros(t_len)
 
Q_cruise = np.zeros(t_len)

Q_cool = np.zeros(t_len) #module cooling rate log
Q_net = np.zeros(t_len) #module heat accumulation log

Module_Temp[0] = T_0
Module_Temp_Ad[0] = T_0
Cabin_Temp[0] = T_0
Cabin_Temp_Ad[0] = T_0
Ambient_Temp[0] = T_0

i = 0

for t in np.arange(0, mc[-1], ts):

    if t <= mc[0]:  # Taxi from NASA
        dep_power = 0.0  # kW
        cruise_power = 10.0  # kW
        U0 = 20.0 #m/s
        alt = EAFB_alt
		

    elif t > mc[0] and t <= mc[1]:  # TO Checklist
        dep_power = 0.0  # kW
        cruise_power = 0.1  # kW
        U0 = 20.0
        alt = EAFB_alt

    elif t > mc[1] and t <= mc[2]:  # Cruise Runup
        dep_power = 0.0  # kW
        cruise_power = 120.0  # kW
        U0 = 20.0
        alt = EAFB_alt

    elif t > mc[2] and t <= mc[3]:  # HLP Runup
        dep_power = 30.0  # kW
        cruise_power = 0.1  # kW
        U0 = 20.0
        alt = EAFB_alt

    elif t > mc[3] and t <= mc[4]:  # Flight go/no-go
        dep_power = 0.0  # kW
        cruise_power = 0.0  # kW
        U0 = 20.0
        alt = EAFB_alt

    elif t > mc[4] and t <= mc[5]:  # Ground roll
        dep_power = 30.0  # kW
        cruise_power = 120.0  # kW
        U0 = 20.0
        ALT = EAFB_alt

    elif t > mc[5] and t <= mc[6]:  # Climb to 1000 feet
        dep_power = 15.0  # kW
        cruise_power = 120.0  # kW
        U0 = 40.0
        alt = app_alt

    elif t > mc[6] and t <= mc[7]:  # Cruise Climb
        dep_power = 0.0  # kW
        cruise_power = 120.0  # kW
        U0 = 54.0
        alt = cruise_alt

    elif t > mc[7] and t <= mc[8]:  # CruiseW
        dep_power = 0.0  # kW
        cruise_power = 90.0  # kW
        U0 = 54.0
        alt = cruise_alt

    elif t > mc[8] and t <= mc[9]:  # Descent to 1000 feet relative/1000 m abs
        dep_power = 0.0  # kW
        cruise_power = 60.0  # kW
        U0 = 54.0
        alt = app_alt

    elif t > mc[9] and t <= mc[10]:  # Final approach
        dep_power = 15.0  # kW
        cruise_power = 0.0  # kW
        U0 = 54.0
        alt = app_alt

		
    elif t > mc[10] and t <= mc[11]:  # Go around to 1000 feet
        dep_power = 15.0  # kW
        cruise_power = 120.0  # kW
        U0  = 54.0
        alt = app_alt

    elif t > mc[11] and t <= mc[12]:  # Approach pattern
        dep_power = 0.0  # kW
        cruise_power = 90.0  # kW
        U0 = 54.0
        alt = app_alt

    elif t > mc[12] and t <= mc[13]:  # Final Approach
        dep_power = 15.0  # kW
        cruise_power = 0.0  # kW
        U0 = 54.0
        alt = app_alt

    elif t > mc[13] and t <= mc[14]:  # Rollout and turnoff
        dep_power = 0.0  # kW
        cruise_power = 15.0  # kW
        U0 = 20.0
        alt = EAFB_alt

    #elif t>2955 and t<=3555:
    else:               # Taxi to NASA
        dep_power = 0.0  # kW
        cruise_power = 10.0  # kW
        U0 = 10.0
        alt = EAFB_alt


		
	
    
 
    
    Pmotor = 1000*cruise_power #total power requested by propulsion
    Qdiss = (1-eff_cell)*Pmotor/12 #instantaneous heat dissipated by single module
    Qmisc = 1000 #W, miscellaneous heat loads in aircraft cabin (pilot, avionics, etc)
 	
    #(Need to update tropo altitude model!)
    T_ground = 35 #deg C, air temperature at ground
    T_0 = T_ground - .00649*alt # #deg C, standard tropospheric lapse rate from T_ground
    #T_0 = 35 #deg C, debug temperature
    #tracking time, sink isothermal temp, sink adiabatic temp (debug) and inlet air temp as func of altitude.
   
    #module convection model - adjacent plates
    Tfilm = 0.5*(Module_Temp[i]+Cabin_Temp[i]+273) #guesstimate of module freeconv film temperature
    Beta = 1/Tfilm
    Ra_module = (rho_0*9.81*Beta*cp_air*(s_module**4)*(Module_Temp[i]-Cabin_Temp[i])+.01)/(mu_air*k_air*h_module)
    Nu_module = (Ra_module/24)*((1-np.exp(-35/Ra_module))**0.75)
    
    hconv_module = Nu_module*k_air/Lc_module #free convection hconv
    A_conv_module = Lc_module**2 #free convection featureless surface

    #hconv_module = 50 #forced convection hconv, given sink design
    #A_conv_module = .06 #m^2, finned surface with n=10, 1cm X 30 cm fins
	
    U_conv_module = hconv_module*A_conv_module
	
	#convection rates from module and cabin from current time step T's
    Qconv_cabin = -U_conv_cabin*ts*(Cabin_Temp[i]-T_0)
    Qconv_module = -U_conv_module*(Module_Temp[i]-Cabin_Temp[i])
    Qconv_module_Ad = -U_conv_module*(Module_Temp_Ad[i]-Cabin_Temp_Ad[i]) #for adiabatic cabin
	
    #temperature corrections
    dT_cabin = (Qmisc - 12*Qconv_module + Qconv_cabin)*ts/(mcp_cabin_air+mcp_cabin)
    dT_module = (Qdiss + Qconv_module)*ts/mcp_module
    dT_module_Ad = (Qdiss + Qconv_module_Ad)*ts/mcp_module
    dT_cabin_Ad = (Qmisc - 12*Qconv_module_Ad)*ts/(mcp_cabin_air+mcp_cabin)

    Module_Temp[i+1] = Module_Temp[i] +  dT_module#module heat loss to convection
    Module_Temp_Ad[i+1] = Module_Temp_Ad[i] + dT_module_Ad
    Cabin_Temp_Ad[i+1] = Cabin_Temp_Ad[i] + dT_cabin_Ad
    Cabin_Temp[i+1] = Cabin_Temp[i] + dT_cabin #cabin with module and avionics heat load
    Ambient_Temp[i] = T_0
	
    dTdt[i+1] = dT_module
    Module_Hconv[i] = hconv_module
    Module_Ra[i] = Ra_module
    Module_Nusselt[i] = Nu_module
    Delta_T[i] = Module_Temp[i] - Cabin_Temp[i] 
	
    Q_gen[i+1] = Qdiss - Qconv_module
    
    Time[i+1] = Time[i] + ts
	

	
	
    i = i + 1

    # Print Results


print('Max Module Temp: %f deg C' % max(Module_Temp))
print('Battery system thermal mass: %f deg C/J' % (12*mcp_module))
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

ax1.plot(Time, Module_Temp, 'r-', label='Module Temperature')
#ax2.plot(Time, Cabin_Temp, 'g-', label='Cabin Temperature')
#ax3.plot(Time, Ambient_Temp, 'b-', label='Ambient Temperature')
ax4.plot(Time, Q_gen, 'g-', label='Module Absorbed Heat (W)')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Temperature (deg C)', color='k')
ax4.set_ylabel('Module absorbed heat rate(W)', color='k')


legend = ax1.legend(loc='upper center', shadow=True)	
#legend = ax2.legend(loc='upper center', shadow=True)
legend = ax4.legend(loc='upper center', shadow=True)


plt.show()


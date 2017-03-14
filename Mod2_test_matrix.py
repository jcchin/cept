
#1.  A (seriously) reduced thermohydraulic model of the X57 cruise nacelle.  
#Thermo-hydro network consists of:  2 flow resistors, 2 heat sources, 1 dynamic pressure boundary.  
#flow coefficients picked from COMSOL studies (inverter sink) and joby SS flowpath result for as-installed motor cooler. 
#no contribution to flow resistance from aft vent (not defined yet for mod2, minor influence anyway)
#convective heat transfer coefficients are from narrow channel HS studies and R. Christie's motor cooling analysis.
 
 
#TODO:  add quasi dynamic behavior
#1. find SS heat balance into motor, inverter
#2. define heating function T_inv(t) = integrate(T_inv + ((Qdiss-Qconv(T_inv))/m*cp) dt)| 0, t_end
#3. print heating function for SS.


import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab 
from itertools import accumulate

#mod2 test params
rho_0 = 1.055 #inlet density, kg/m^3
p0_dyn = 1689 #dynamic pressure, Pa
cruise_power = 60.0 #72.1 max cruise power per prop, kW

rho_sink = 2700 #kg/m^3
cp_sink = 896 #J/kg-K, aluminum sink specific heat
cp_0 = 1005 #J/kg-K, inlet air specific heat
A0=0.013 * 2 #m^2, annular inlet area x2
A1=0.05015 * 2 #m^2, sink flow area x 2
cf_sink = .0022 #.0015 default:  m^3/Pa,  fin row flow resistance, linear.  Total volumetric flow has the form Vdot1 = (P1_res-P0_dyn)*(cf_sink + cf_motor)
cf_motor = .002 #.0012 default:  m^3/Pa, motor flow res  istance, linear
k_sink = 200 #W/m-K
L_inv_sink = 0.1 #m  #component length characteristics for transient parameters (Bi, Fo)
L_motor_sink = 0.1 #m
D_motor_sink = 0.002 #m
D_inv_sink = 0.0015 #m

mcp_motor = 4.25 * 900  #J/K
mcp_controller = 1.45*2*900 #J/K


T0 = 35 # C sink inlet air initial temp -- ground level, hot day

h_conv_motor = 32 #25 default:  motor fin row conv htc,  from R.Christie motor thermal spreadsheet
h_conv_inv = 50 #W/C fin row convective HTC
A_conv_inv=1.393 #m^2, sink exchange surface area Nfins*Lchan*Hchan*2*2
A_conv_motor=1.935 #m^2 for both motors

 
Qmotor = 1000*cruise_power*.05 #motor at 95% efficiency
Qinv = 2*1000*cruise_power*.02 #2x controller at 98% eff


dt = 1 # time step, s
t_len = 500
Time = np.zeros(t_len)
Time[1] = 0

MotorTemp = np.zeros(t_len)
ControllerTemp = np.zeros(t_len)
MotorTemp[1] = T0
ControllerTemp[1] = T0


#basic procedure:  
#1. solve for flow rate from driving dynamic pressure/combined flow coeff;  
#2. energy balance across heater component to get exit flow temp
#3. convective heat eqn to get source temperatures
#4. repeat for next downstream component

p_res = p0_dyn * (50/1500) #empirical, from TD pressure sensitivity studies
volrate = 2 * p_res *(cf_sink)  #assumes only significant flow resistors from cooling fins)
#volrate = 2*p0_dyn*(cf_sink+cf_motor)
mdot = rho_0 * volrate
T0 = 40 #deg C, hot ground level temp
T1 = T0 + (Qmotor/(cp_0*mdot)) #motor exhaust temp
T_motor = T0 + Qmotor/(h_conv_motor*A_conv_motor)  #estimated motor temp
T2 = T1 + (Qinv/(cp_0*mdot))
T_inv = T1 + Qinv/(h_conv_inv*A_conv_inv)

volrate_half = volrate/2

tau_h_motor = mcp_motor / (Qmotor - h_conv_motor*A_conv_motor)
tau_h_controller = mcp_controller / (Qinv - h_conv_inv*A_conv_inv)
tau_c_motor = mcp_motor / (h_conv_motor*A_conv_motor)
tau_c_controller = mcp_controller / (h_conv_inv*A_conv_inv)

Fo_motor = k_sink/(cp_sink*rho_sink*L_motor_sink*L_motor_sink)
Fo_controller = k_sink/(cp_sink*rho_sink*L_inv_sink*L_inv_sink)

Bi_motor = L_motor_sink*h_conv_motor/k_sink
Bi_controller = L_inv_sink*h_conv_inv/k_sink


i=1
for i in range(1,t_len-1):
	
	
	MotorTemp[i+1] = MotorTemp[i] + ((dt*Qmotor/mcp_motor) - h_conv_motor*A_conv_motor*(MotorTemp[i] - T0)/mcp_motor)
	ControllerTemp[i+1] = ControllerTemp[i] + (dt*Qinv/mcp_controller) - (dt*h_conv_inv*A_conv_inv*(ControllerTemp[i] - T1)/mcp_controller)
	
	Time[i+1] = Time[i]+dt	
		
	i+=1

print('Inlet dynamic pressure (Pa) %f:' % p0_dyn)
print('Residual pressure at controller sink %f:' % p_res)

print('SS flow through (each, m^3/s) %f:' % volrate_half)

print('SS motor temp (C): %f' % T_motor)
print('SS Controller temp (C) %f:' % T_inv)
print('Average Motor Temp (C) %f:' % np.mean(MotorTemp))
print('Motor heating time constant (s) %f:' % tau_h_motor)

print('controller heating time constant (s) %f:' % tau_h_controller)
print('motor cooling time constant (s) %f:' % tau_c_motor)
print('controller cooling time constant (s) %f:' % tau_c_controller)

print('Motor Bi: %f' % Bi_motor)


fig, ax1 = plt.subplots()
ax1.plot(Time, MotorTemp, 'r-')
ax1.plot(Time, ControllerTemp, 'b-')
ax1.set_ylabel('Motor Temp (r) and Controller Temp (b) (deg C)', color='b')
ax1.set_xlabel('time (s)')

for tl in ax1.get_yticklabels():
    tl.set_color('b')

plt.show()




input('Press enter to exit')


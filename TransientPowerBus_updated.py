import numpy as np
import math as m
import matplotlib.pyplot as plt
import pylab


#http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
alt_table = [-1000,0,1000,2000,3000,4000,5000,6000,7000]
temp_table = [49,49,8.5,2,-4.49,-10.98,-17.47,-23.96,-30.45] #21.5
gravity_table = [9.81,9.807,9.804,9.801,9.797,9.794,9.791,9.788,9.785]
pressure_table = [11.39,10.13,8.988,7.95,7.012,6.166,5.405,4.722,4.111]
kin_viscosity_table = [1.35189E-05,1.46041E-05,1.58094E-05,1.714E-05,1.86297E-05,2.02709E-05,2.21076E-05,2.4163E-05,2.64576E-05]

# http://www.engineeringtoolbox.com/air-properties-d_156.html
temperature_table = [-100,-50,0,20,40,60,80,100,120,140,160,180,200]
k_table = [0.016,0.0204,0.0243,0.0257,0.0271,0.0285,0.0299,0.0314,0.0328,0.0343,0.0358,0.0372,0.0386]
Pr_table = [0.74,0.725,0.715,0.713,0.711,0.709,0.708,0.703,0.7,0.695,0.69,0.69,0.685]

#http://www.engineeringtoolbox.com/dry-air-properties-d_973.html
temperature_alpha = [-73,-23,27,77,127]
alpha_table = [0.00001017,0.00001567,0.00002207,0.00002918,0.00003694]
cruise_alt = 10000 #ft

time = [0, 825, 885, 1605, 1905, 2505, 2625, 2685, 2775, 2895, 3555]
altitude = [0, 0, 304.8, cruise_alt * .3048, cruise_alt*.3048, 304.8, 0, 304.8, 304.8, 0, 0] #m

#carbon fiber specs
t_cf = 0.0006096 # m
k_cf = 21 # W/mK

RperL = .003276
bus_voltage = 416 # V
motor_efficiency = 0.95
inverter_efficiency = 0.95
# number_conductors = 4
# bundle_derating = 0.5125
number_conductors = 8
bundle_derating = 0.375
alt_derating = 0.925
d = 0.01905 # m
AperStrand = 16*np.pi*(2.59E-3)**2/4
Cp = 385
rho = 8890
HC = AperStrand*Cp*rho

T_wire = 49

ts = 0.5

k_air = 0.0285
t_air = 0.003937

Time = [0] * 7110
Temp= [0] * 7110
Temp_cruise = [0] * 7110
Q_prime = [0] * 7110
Cruise_current= [0] * 7110
Duct_Temp=[0] * 7110

i = 0
q_conv = 0

T_duct = 49

for t in np.arange(0, 3555, ts):

    if t<=600: # Taxi from NASA
        dep_power = 0.0 #kW
        cruise_power = 2.5 #kW
        #alt = 0

    elif t>600 and t<=720: # TO Checklist
        dep_power = 0.0 #kW
        cruise_power = 0.0 #kW
       # alt = 0

    elif t>720 and t<=780: #Engine Runup
        dep_power = 30.0 #kW
        cruise_power = 30.0 #kW
        #alt = 0

    elif t>780 and t<=810: # Flight go/no-go
        dep_power = 0.0 #kW
        cruise_power = 0.0 #kW
        #alt = 0

    elif t>810 and t<=825: # Ground roll
        dep_power = 30.0 #kW
        cruise_power = 30.0 #kW
       # alt = 0

    elif t>825 and t<=885: # Climb to 1000 feet
        dep_power = 22.5 #kW
        cruise_power = 30.0 #kW
        #alt = 0 # *************************

    elif t>885 and t<=1605: # Cruise Climb
        dep_power = 0.0 #kW
        cruise_power = 30.0 #kW
        #alt = 0 # *************************

    elif t>1605 and t<=1905: # Cruise
        dep_power = 0.0 #kW
        cruise_power = 22.5 #kW
       # alt = 0 # *************************

    elif t>1905 and t<=2505: # Descent to 1000 feet
        dep_power = 0.0 #kW
        cruise_power = 15.0 #kW
        #alt = 0 # *************************

    elif t>2505 and t<=2625: # Final approach
        dep_power = 22.5 #kW
        cruise_power = 0.0 #kW
        #alt = 0 # *************************

    elif t>2625 and t<=2685: # Go around to 1000 feet
        dep_power = 22.5 #kW
        cruise_power = 30.0 #kW
        #alt = 0 # *************************

    elif t>2685 and t<=2775: # Approach pattern
        dep_power = 0.0 #kW
        cruise_power = 30.0 #kW
        #alt = 0 # *************************

    elif t>2775 and t<=2895: # Final Approach
        dep_power = 22.5 #kW
        cruise_power = 0.0 #kW
        #alt = 0 # *************************

    elif t>2895 and t<=2955: # Rollout and turnoff
        dep_power = 0.0 #kW
        cruise_power = 3.8 #kW
        #alt = 0

    #elif t>2955 and t<=3555: # Taxi to NASA
    else:
        dep_power = 0.0 #kW
        cruise_power = 2.5 #kW
        #alt = 0 # *************************

    # else:
    #     print('error')
    #     quit()
    alt = np.interp(t, time, altitude)
    T_ambient = np.interp(alt, alt_table, temp_table)
    kinematic_viscosity = np.interp(alt, alt_table, kin_viscosity_table)
    k_air = np.interp(T_ambient, temperature_table, k_table)
    Pr = np.interp(T_ambient, temperature_table, Pr_table)

    # CRUISE BUS
    cruise_bus_power = cruise_power/(motor_efficiency*inverter_efficiency) #kW
    cruise_bus_current = 1000*cruise_bus_power/bus_voltage # ****
    cruise_current_per_conductor = cruise_bus_current/number_conductors
    cruise_current_rating_per_conductor = cruise_current_per_conductor/(bundle_derating*alt_derating)

    q_prime_cruise = cruise_current_rating_per_conductor**2 * RperL
    # Temp_ROC_cruise = q_prime_cruise/HC
    # T_wire_cruise = T_wire_cruise + (ts*Temp_ROC_cruise)



    # DEP BUS
    dep_bus_power = dep_power/(motor_efficiency*inverter_efficiency) #kW
    dep_bus_current = 1000*dep_bus_power/bus_voltage # ****
    dep_current_per_conductor = dep_bus_current/number_conductors
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
    
    h = 7.0 #W/m^2K

    r_conv = 1/(h)

    q_prime_out = np.pi * d *(T_duct-T_ambient)/r_conv
    
    q_net = q_prime_in-q_prime_out

    Duct_temp_ROC = q_net/(HC)
    T_duct = T_duct + (ts*Duct_temp_ROC)

    



    Time[i] = t
    Temp[i] = T_wire
    Duct_Temp[i] = T_duct
    # Temp_cruise[i] = T_wire_cruise
    # Temp_dep[i] = T_wire_dep
    # Q_prime[i] = total_q_prime
    # Cruise_current[i] =cruise_current_rating_per_conductor

    i = i+1

print(max(Duct_Temp))
plt.figure()
plt.plot(Time,Temp, 'r', label = 'Wire')
plt.plot(Time,Duct_Temp, 'b', label = 'Carbon Fiber Duct')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.xlabel('Time (s)')
plt.ylabel('Wire Temperature (C)')
pylab.show()




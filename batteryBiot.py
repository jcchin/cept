

#assume
# aluminum walls (steel?)
# material of plates on battery pallet?

from math import pi
# Generic 18650 SDI cell- steel case

m_cell = 45. #grams
d_cell = 18. #mm
z_cell = 64.8 #mm

vol_cell = 0.25*d_cell**2*z_cell  # mm^3 --> cm^3

rho_cell = m_cell/vol_cell # g/cm^3

k_alum = 200. #W/m*K
k_cell = 0.1*k_alum

cp_cell = 1.02 #J/g*K

Q_cell = 0.1 #W

k_air = 0.0285 #W/m*K
kin_air = 1.89e-5 #m^2/s

cp_air = 1005 #J/g*K

Pr = 0.71

rho_air = 1.067 # kg/m^3

T_max = 60.+273.15  # K

T_amb = 35. + 273.15  # K

T_film = 0.5*(T_max+T_amb)

Beta = 1./T_film

alpha_air = k_air/rho_air*cp_air

# Lumped Capacitance methods valid for Bi

g = 9.81 #m/s^2
A_cell = 0.25*pi*d_cell**2

#For h.cell from endcap convection (insulated cell wall) - 2 times vertical flat plate

# Ra for uniform surface heat flux
Ra_d_cell = (g*Beta*Q_cell/A_cell*d_cell**4)/(k_air*alpha_air*kin_air)


Nu_d_cell = 2*(0.825 + ((0.387*Ra_d_cell**(1./6.))/(1+(0.492/Pr)**(9./16.))**(8./27.)))**2.




h_cell = Nu_d_cell*k_cell/d_cell  #W/m^2*K

hA_cell = h_cell*A_cell

Bi = h_cell*d_cell / k_alum #1.787


print("Bi: ", Bi)


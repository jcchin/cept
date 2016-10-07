#!/usr/bin/env python3

from numpy import tanh, arange, meshgrid, empty, exp, zeros
from numpy import sqrt
import numpy as np

# http://www.hpl.hp.com/techreports/Compaq-DEC/WRL-86-4.pdf

def calc(V_in, fin_h2):
    InvPwr = 45000.0  # desc='total power through the inverter = units='W')
    Inv_eff = 0.98  # desc='inverter efficiency')
    T_air = 47.  # desc='Cooling Air Temp = units='degC')
    T_max = 80.  # desc='Maximum Electronics Temp = units='degC')
    sink_w = .192  # desc='Heat Sink Base Width = units='m')
    sink_l = .160  # desc='Heat Sink Base Length = units='m')
    sink_h = .002  # desc='Heat Sink Base Height = units='m')
    avail_h = 0.1  # desc='maximum available height for heat sink = units='m')
    gamm_eff = 1.8  # desc='boundary layer thickness parameter, dimensionless')
    V_in = 20.  # desc='initial velocity = units='m/s')
    fin_h2 = 0.031  # desc='user defined fin height = units='m' )
    grease_thickness = 0.0001  # desc='thickness of thermal grease = units='m')
    k_grease = 0.79  # desc='Thermal conductivity of grease = units='W/(m*K)')
    case_thickness = 0.001  # desc='thickness of thermal grease = units='m')
    k_case = 150.  # desc='Thermal conductivity of grease = units='W/(m*K)')
    rho = 1.158128  # desc='density of air = units='kg/m**3')
    mu = 0.000018  # desc='dynamic viscosity',units='Pa*s')
    nu = 0.0000168  # desc='kinematic viscosity',units='m**2/s')
    Cp = 1004.088235  # desc='specific heat at altitude = units='J/(kg*K)')
    k_air = 0.026726  # desc='thermal conductivity of air = units='W/(m*K)')
    k_fin = 200.0  # desc='thermal conductivity of aluminum = units='W/(m*K)' )
    flag = 2  # desc='test flag, to set values')

    Dh = 1.0# desc='hydraulic diameter = units='m')
    Lstar = 1.0  # desc='characteristic length = units='m')

    h = 20.  # desc='fin heat transfer coefficient = units='W/(m**2*K)')
    Re = 10.  # desc='Reynolds  #')
    Pr = 10.  # desc='Prandtl  #')
    Nu = 10.  # desc='Nusselt  #')
    R_max = 0.05  # desc='Max Thermal Resistance = units='degC/W')
    R_tot = 0.05  # desc='Total Thermal Resistance = units='degC/W')
    fin_eff = 1.0  # desc='Fin Efficiency (based on Aspect ratio)')
    R_hs = 0.05  # desc='Heat Sink Resistance = units='degC/W')
    n_fins = 137.  # desc='Number of fins that can fit on the sink')
    min_area = 1.4  # desc='Minimum cooling area needed = units='m**2')
    fin_h = .032  # desc='Fin height = units='m')
             # Pressure drop calcs

    lambda_ = 2.  # desc='dimensionless channel height')
    omega = 0.0554  # desc='normalized thermal Resistance')
    R_lossless = 0.845  # desc='thermal resistance of optimal lossless heat sink = units='degC/W')
    R_global = 1.28  # desc='global upper bound on thermal resistance of an optimal sink = units='degC/W')
    zeta_global = 1.512  # desc='effectiveness ratio of global upper bound, dimensionless')
    R_thin = 1.15  # desc='thin fin upper bound on thermal resistance of an optimal sink = units='degC/W')
    zeta_thin = 1.361  # desc='effectiveness ratio of thin fin upper bound, dimensionless')
    alpha = 37.  # desc='maximum increase in effective heat transfer area, dimensionless')
    fin_gap = 0.001  # desc='spacing between fins = units='m')
    fin_w = 0.0009  # desc='Heat sink fin width = units='m')
    R_case = 0.0001  # desc='inverter case thermal resistance = units='degC/W')
    R_grease = 0.004  # desc='grease thermal resistance = units='degC/W')


    pwr = InvPwr*(1.-Inv_eff)
    Abase = sink_w*sink_l
    gam_e = gamm_eff
    R_max = (T_max - T_air)/ pwr
    #min_area = pwr/(h_0*(T_max-T_air))

    #choose fin height Arthur Method:
    #H = fin_h = 0.5*min_area/(n_fins*sink_l)
    if flag ==1:  # force to test case fin height value
        H = 0.0141
    if flag ==2:  # let user specify fin height
        H = fin_h2

    Pr = Pr = (mu*Cp)/k_air
    if flag:
        Pr = Pr = 0.708  #force to test case value

    Nu = 2.656*gam_e*Pr**(1./3.)  #non-ducted
    ducted = False
    if ducted:
        Nu = 8.235  # http://www.dtic.mil/dtic/tr/fulltext/u2/a344846.pdf
    alpha = sqrt(k_fin/(k_air*Nu))

    V = V_in
    b = fin_gap = 2.*gam_e*sqrt((nu*sink_l)/V)
    lam = lambda_ = H/(fin_gap * alpha)
    omega = H / (k_fin * Abase)


    Dh = 2.*b
    l = sink_l
    Re = (rho*V*b*b)/(mu*l)
    Lstar = l/(Dh*Re*Pr)

    R_lossless = omega*lam**-2.
    zeta_global = 2.*lam + 1.
    R_global = zeta_global * R_lossless

    zeta_thin = sqrt(lam)*(lam+1.)/tanh(sqrt(lam))
    R_thin = zeta_thin * R_lossless

    fin_w = H/alpha  # actual ideal will be a bit smaller, accurate with small lambda
    n_fins = sink_w/(fin_w+fin_gap)

    R_case = grease_thickness/(k_grease*Abase)

    R_grease = case_thickness/(k_case*Abase)

    R_tot = R_case + R_grease + R_thin

    return R_max,R_tot,fin_gap,fin_w,R_case,R_grease

if __name__ == '__main__':

    def printstuff():
        print('=============')
        print('V_in : %f' %V_in)
        print('Nu : %f' %Nu)
        print('Pr : %f' %Pr)
        print('Re : %f' %Re)
        print('L* : %f' %Lstar)
        print('alpha : %f' %alpha)
        print('fin height: %f' %fin_h)
        print('lambda : %f' %lambda_)
        print('omega : %f' %omega)
        print('R_lossless : %f' %R_lossless)
        print('R_global: %f' % R_global)
        print('zeta_thin : %f' %zeta_thin)
        print('# of fins : %f' %n_fins)
        print('-------------')
        print('fin thickness: %f' % fin_w)
        print('fin gap: %f' % fin_gap)
        print('R_thin: %f' % R_thin)

    R_max,R_tot,fin_gap,fin_w,R_case,R_grease = calc(20.,0.031)
    print('R_max: %f' % (R_max-R_case-R_grease))
    #print('Arthur Height')
    #printstuff()

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
     # Some sample data
    x_side = arange(4., 24., .25)  # velocity
    y_side = arange(0.01, 0.06, 0.000625)
    X, Y = meshgrid(x_side,y_side)
    Z = zeros(np.shape(X))
    Za = zeros(np.shape(X))
    Zb = zeros(np.shape(X))

    flag = 2
    for i,vel in enumerate(x_side):
        for j,ht in enumerate(y_side):
            # V_in = vel
            # fin_h2 = ht
            R_max,R_tot,fin_gap,fin_w,R_case,R_grease = calc(vel,ht)
            Z[j,i] = R_tot
            Za[j,i] = fin_gap
            Zb[j,i] = fin_w
        print(fin_gap)

     # Assign colors based off some user-defined condition
    COLORS = empty(X.shape, dtype=str)
    COLORS[:,:] = 'r'
    COLORS[Z<R_max] = 'y'
    COLORS[Z<(R_max-R_case-R_grease)] = 'g'

     # 3D surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=COLORS, rstride=1, cstride=1,
            linewidth=0)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Fin Height (m)')
    ax.set_zlabel('Thermal Resistance')
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Za, facecolors=COLORS, rstride=1, cstride=1,
            linewidth=0)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Fin Height (m)')
    ax.set_zlabel('fin gap')
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Zb, facecolors=COLORS, rstride=1, cstride=1,
            linewidth=0)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Fin Height (m)')
    ax.set_zlabel('Fin thickness')
    plt.show()



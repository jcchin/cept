function [ dT ] = dxdt( t,temp )


    global solverTime
    global Ambient_Temp % np.zeros(t_len)  % ambient temp --> per altitude model
    global Q_gen  % rate of heat absorbed by battery module
    global Delta_T

    global DEP_pwr
    global altitude
    global temp_table
    global alt_table
    global time

    dep_power = interp1(time, [0 DEP_pwr],t(1),'next');
    alt = interp1(time, [0 altitude],t(1));
    T_0 = interp1(alt_table, temp_table, alt);

    A_conv_motor = .0474;  % m^2
    h_conv_motor = 30;  % W/m^2-C
    r_motor = 0.25;  % K/W
    len_motor = 0.065;
    eff_motor = 0.95;

    A_conv_inv = 0.05;  % m^2 - estimate of available cabin surface area for external convection
    h_conv_inv = 50;  % W/m^2-C estimate of cabin external convection coefficient (computed for Lc = 0.8m, u_0=50 m/s)
    r_sink = 0.25;  % K/W
    len_inv = 0.08;
    eff_inv = 0.97;


    %free convection parameters
    rho_0 = 1.055;  % kg/m^3
    mu_air = 1.81E-05;  % Pa-s, viscosity of ambient air
    k_air = 0.026;  % W/m-C, therm. conductivity of air

    k_wall_nacelle = 200;  % W/m-C module wall conductivity (Al 6061)
    t_wall_nacelle = .003;  % m module wall thickness
    U_conv_motor = A_conv_motor/((1/h_conv_motor)+(t_wall_nacelle/k_wall_nacelle));  % W/C total heat transfer coefficient from module to cabin
    U_conv_motor = 30.;  % override value 5 W/m^2-C


    cp_air = 1005;  % J/kg-C specific heat of air (average)
    cp_al = 900;  % J/kg-C specific heat of aluminum grid and external plate
    m_nacelle_air = 1.;  % m^3
    m_inv = 1.0;  % kg, grid and external plate area per 160 cell module.
    m_motor = 2.0;  % kg mass of cells per 160 cell module

    mcp_nacelle = m_nacelle_air*cp_air;
    mcp_motor = m_inv*cp_al;
    mcp_inv = m_motor*cp_al;

    Pmotor = 1000./12.*dep_power;  % W, total power requested by propulsion
    Qmotor = (1.-eff_motor)*Pmotor;  % W, instantaneous heat dissipated by the motor
    Qinv = (1.-eff_motor)*(Pmotor+Qmotor);  % W, inverter heat diss

    % convection model - adjacent plates
    %Tfilm = 0.5*(temp[0]+temp[2]+273.)  % guesstimate of freeconv film temperature
    %Beta = 1./Tfilm
    %Ra_module = (rho_0*9.81*Beta*cp_air*(s_module**4)*(temp[0]-temp[2])+.01)/(mu_air*k_air*h_module)
    %Nu_module = (Ra_module/24.)*((1-np.exp(-35./Ra_module))**0.75)

    %hconv_module = Nu_module*k_air/Lc_module  % free convection hconv
    %A_conv_module = Lc_module**2  % free convection featureless surface

    % hconv_module = 30. %forced convection hconv, given sink design
    % A_conv_module = .06 %m^2, finned surface with n=10, 1cm X 30 cm fins

    U_conv_motor = h_conv_motor*A_conv_motor;
    U_conv_inv = h_conv_inv*A_conv_inv;

    %convection rates from module and cabin from current time step T's
    Qconv_motor = U_conv_motor*((1.-r_motor)*temp(1)-T_0);
    Qconv_inv = U_conv_inv*((1.-r_motor)*temp(2)-T_0);

    %temperature corrections
    dT_motor = (Qmotor - Qconv_motor)/mcp_motor;  % cabin with module and avionics heat load
    dT_inv = (Qinv - Qconv_inv)/mcp_inv;  % module heat loss to convection

    dT = [dT_motor; dT_inv];
    % save off other useful info (not integrated states)
    % note: the time breakpoints are "solverTime", not "times"
    %       to allow forsolver adaptive time stepping
    solverTime = [solverTime t];
    Ambient_Temp = [Ambient_Temp T_0];

    if (Pmotor < 10000)
        Delta_T = [Delta_T dT_motor];
    Q_gen = [Q_gen Qmotor];

end


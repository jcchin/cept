global altitude
global DEP_pwr
global U0
global Ambient_Temp
global Q_gen
global solverTime

filename = '/Users/jeffchin/Documents/NASA/2016/SCEPTOR/Python/mission.csv';
delimiter = ',';
startRow = 2;
formatSpec = '%f%f%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);
fclose(fileID);
segmentDuration = dataArray{:, 1};
altitude = dataArray{:, 2}';
DEP_pwr = dataArray{:, 3}';
Cruise_pwr = dataArray{:, 4};
U0 = dataArray{:, 5}';
clearvars filename delimiter startRow formatSpec fileID dataArray ans;

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
m_inv = 0.5;  % kg, grid and external plate area per 160 cell module.
m_motor = 1.0;  % kg mass of cells per 160 cell module

mcp_nacelle = m_nacelle_air*cp_air;
mcp_motor = m_inv*cp_al;
mcp_inv = m_motor*cp_al;

T_0 = 35.;  % C module and aircraft equilibrium temperature at start (HOT DAY)

ts = 0.5;  % time step, s

mc = cumsum(segmentDuration)'; %uncomment if using 'segment duration'
global time 
time = [0 mc];

%http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
global alt_table
alt_table = [-1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]; % meters
global temp_table 
temp_table = [35., 35., 35., 8.5, 2., -4.49, -10.98, -17.47, -23.96];%, -30.45] % Celsius (21.5 @ 0 in reality, using 49 to be conservative)

% initiate transient variable arrays
times = 0:ts:mc(end);



% states:
%0 = Motor_Temp     % motor (bulk) temperature
%1 = Inv_Temp  % inverter (bulk) temperature
%2 = Nacelle_Temp % nacelle skin (bulk) temperature


%Integrate
[t, Temp_states] = ode23(@dxdt, times, [T_0;T_0]);

hold on
plot(times,Temp_states)
plot(solverTime, Ambient_Temp)
% % Print Results
% print('Max Comp Temp: %f deg C') % max(Temp_states[:,0]))
% 
% 
% fig1 = plt.figure()
% 
% ax1 = fig1.add_subplot(111)
% ax2 = fig1.add_subplot(111)
% ax3 = fig1.add_subplot(111)
% ax4 = ax2.twinx()
% ax5 = fig1.add_subplot(111)
% 
% ax1.plot(times, Temp_states[:,0], 'r-', label='Motor Temperature')
% % ax2.plot(times, Temp_states[:,2], 'g-', label='Cabin Temperature')
% ax3.plot(times, Temp_states[:,1], 'k-', label='Inverter Temperature')
% ax4.plot(solverTime, Q_gen, 'g-', label='Motor Heat (W)')
% ax5.plot(solverTime, Ambient_Temp, 'b-', label='Ambient Temp (C)')
% 
% ax1.set_xlabel('Time (s)')
% ax1.set_ylabel('Temperature (deg C)', color='k')
% ax4.set_ylabel('Module absorbed heat rate(W)', color='k')
% 
% legend = ax1.legend(loc='upper center', shadow=True)
% %legend = ax2.legend(loc='upper center', shadow=True)
% %legend = ax4.legend(loc='upper center', shadow=True)
% 
% plt.show()

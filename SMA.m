global altitude
global DEP_pwr
global U0
global Ambient_Temp
global Q_gen
global Delta_T
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
hold off

t_limit = (100.-40.)/max(Delta_T)
disp(sprintf('Max Comp Temp: %f deg C', max(Temp_states(:,1))));
disp(sprintf('Temp Over Time: %f seconds', t_limit));

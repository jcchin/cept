from openmdao.api import Component, ScipyGMRES, Newton, IndepVarComp
from openmdao.units.units import convert_units as cu
from math import pi

class PowerBus(Component):
    def __init__(self):
        super(PowerBus, self).__init__()

        self.deriv_options['type'] = "fd"

        self.add_param('busVoltage', 416., desc='lowest battery operating voltage', units='V')
        self.add_param('to_pwr', 45000., desc='take-off power per bus, two buses per wing', units='W')
        self.add_param('peak_pwr', 75000., desc='peak power', units='W')
        self.add_param('climb_pwr', 30000., desc='climb power per bus.. (total of 60kW motor)', units='W')
        self.add_param('cruise_pwr', 20000., desc='cruise/climb power bus', units='W')
        self.add_param('efficiency', 0.9, desc='product of motor and inverter efficiency')

        self.add_param('to_period', 120., desc='take-off period, e.g. 45KW for 2 min', units='s')
        self.add_param('peak_period', 30., desc='peak power period, e.g. 75KW for 30s', units='s')
        self.add_param('climb_period', 600., desc='climb period, e.g. 30KW for 10 min', units='s')
        self.add_param('cruise_period', 600., desc='cruise period, e.g. 20KW for 10 min', units='s')

        self.add_param('Tambient', 71., desc='ambient temperature (160F), includes solar radiation', units='degC')
        self.add_param('Trating', 150., desc='wire temperature rating', units='degC')

        # Ref. EXRAD XLE High Voltage Cable
        # http://www.champcable.com/product_pdfs/12_SAE_150_FX_High_Voltage.pdf
        # No bundle de-rating (only applies to bundles of indivudally
        # insulated wires bundled together)
        # Altitude de-rating Not Applicable below 60,000ft

        cable_data = [\
        {'Wire':1, 'strands':779, 'AWG':30, 'AperStand':0.0509, 'Resistance':0.4056, 'A40':293, 'A71':270},
        {'Wire':2, 'strands':665, 'AWG':30, 'AperStand':0.0509, 'Resistance':0.5114, 'A40':255, 'A71':230},
        {'Wire':4, 'strands':133, 'AWG':25, 'AperStand':0.1624, 'Resistance':0.8132, 'A40':190, 'A71':170}]

        self.add_param('cable_data', cable_data)

        # Equivalent Bundled Wire Table
        # http://www.rapidtables.com/calc/wire/wire-gauge-chart.htm
        # http://www.faa.gov/documentLibrary/media/Advisory_Circular/Chapter_11.pdf
        # Figure 11-4b page 33

        # MatWeb ETP copper
        self.add_param('cpcu', 0.385, desc='Cp of copper', units='J/gC')
        self.add_param('rhocu', 8.89, desc='density of copper', units='g/cc')
        self.add_param('Resistivity', 1.72e-8, desc='electrical resistivity', units='Ohms/m')

        #todo lookup function to calc wire area given stand sizes

        self.add_param('QperL', 33., desc='Heat per unit length', units='W/m')
        self.add_param('wireD', 0.0052, desc='Wire Diameter', units='m')

        self.add_param('n_hl_motors', 6., desc='Number of High Lift Motors')
        self.add_param('hlpower', 10000., desc='distributed motor power during take-off', units='W')

        self.add_output('deltaT', 79., desc='temperature difference between rated temperature and actual free air temperature', units='degC')
        self.add_output('to_current', 0., units='A')
        self.add_output('peak_current',0., units='A')
        self.add_output('climb_current', 0., units='A')
        self.add_output('cruise_current', 0., units='A')

        self.add_output('A_dL', 0.0163, desc='Area/Length (circumference)')
        self.add_output('deltaT2', 127., desc='wire temp rise above the environment', units='degC')
        self.add_output('wireTemp', 167., desc='wire Temperature', units='degC')

        self.add_output('cable_out',[0.])
        self.add_output('hl_current', 160., desc='current for all motors on one bus', units='A')

    def solve_nonlinear(self, p, u, r):
        u['to_current'] = p['to_pwr']/p['efficiency']/p['busVoltage']
        u['peak_current'] = p['peak_pwr']/p['efficiency']/p['busVoltage']
        u['climb_current'] = p['climb_pwr']/p['efficiency']/p['busVoltage']
        u['cruise_current'] = p['cruise_pwr']/p['efficiency']/p['busVoltage']
        u['deltaT'] = p['Trating']-p['Tambient']

        # In this case the SAE graph shows with a temperature difference
        # of 110C (150C-40C, i.e. rated temperature - environment temperature)
        # the allowable current is 200A. Which mean with 200A going through it,
        # the wire should be at its rated temperature when in a 40C environment.
        # The spreadsheet calculated 167C which is rather close considering
        # the simplified equation used.

        pc = u['peak_current']
        cd = p['cable_data']

        for i,row in enumerate(cd):
            row['area'] = cd[i]['strands']*cd[i]['AperStand']
            row['heat'] = pc**2*(cd[i]['Resistance']/1000.)
            row['heat_cap'] = row['area'] * p['cpcu'] * p['rhocu']
            row['roc'] = row['heat']/row['heat_cap']

        u['A_dL'] = p['wireD']*pi
        turbulent = 0
        if turbulent:
            u['deltaT2'] = (p['QperL']*(1./u['A_dL'])*1./1.24)**(1/1.33)
        else:
            u['deltaT2'] = (p['QperL']*(1./u['A_dL'])*(p['wireD']**0.24)/1.32)**(1/1.25)

        u['wireTemp'] = u['deltaT2']+p['Tambient']

        u['hl_current'] = p['n_hl_motors']*p['hlpower']/p['efficiency']/p['busVoltage']

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    import matplotlib.pyplot as plt

    p = Problem()
    root = p.root = Group()

    root.add('comp', PowerBus())

    p.setup()

    #Check Case: Steady State #4AWG 200A 40C
    p['comp.Tambient'] = 40.
    p.run()
    wT = p['comp.wireTemp']
    print('Wire Temp(C): ', wT , 'Expected: 150degC', 'error %(w.r.t degK):', 100.*(wT - 150.)/(wT+273.), '%')

    #Steady State #4AWG 200A 71C
    p['comp.Tambient'] = 71.
    p.run()
    print('Wire Temp(C): ', wT , 'Expected: 150degC', 'error %(w.r.t degK):', 100.*(wT - 150.)/(wT+273.), '%')

    #Steady State #4AWG 120A 71C
    p['comp.Tambient'] = 71.
    p.run()
    print('Wire Temp(C): ', wT , 'Expected: 150degC', 'error %(w.r.t degK):', 100.*(wT - 150.)/(wT+273.), '%')
    wT = p['comp.wireTemp']
    co = p['comp.cable_data']
    print('Time to reach 150degC after 200A is applied:', (150-128)/co[2]['roc'], 'seconds')

    # High Lift Distributed Small Motors

    # Plot Transient
    time = [0.]                                 # 0
    time.append(time[0]+p['comp.to_period'])    # 1
    time.append(time[1]+1.)                     # 2
    time.append(time[1]+p['comp.peak_period'])  # 3
    time.append(time[3]+1.)                     # 4
    time.append(time[3]+p['comp.climb_period']) # 5
    time.append(time[5]+1.)                     # 6
    time.append(time[5]+p['comp.cruise_period'])# 7

    toc, pc, cc, crc = p['comp.to_current'], p['comp.peak_current'], p['comp.climb_current'], p['comp.cruise_current']
    current = [toc, toc, pc, pc, cc, cc, crc, crc]

    plot = 0
    if plot:
        plt.plot(time,current)
        plt.xlabel('Elapsed Time (sec)')
        plt.ylabel('Bus Current (A)')
        plt.title('Bus Current vs Time')
        plt.show()
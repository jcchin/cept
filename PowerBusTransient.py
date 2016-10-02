from openmdao.api import Component
from itertools import accumulate
from math import pi


class PowerBus(Component):
    def __init__(self):
        super(PowerBus, self).__init__()

        self.deriv_options['type'] = "fd"

        self.add_param('busVoltage', 416., desc='lowest battery operating voltage', units='V')
        self.add_param('to_pwr', 45000. / 10., desc='take-off power per bus, two buses per wing', units='W')
        self.add_param('peak_pwr', 75000. / 10., desc='peak power', units='W')
        self.add_param('climb_pwr', 30000. / 10., desc='climb power per bus.. (total of 60kW motor)', units='W')
        self.add_param('cruise_pwr', 20000. / 10., desc='cruise/climb power bus', units='W')
        self.add_param('efficiency', 0.9, desc='product of motor and inverter efficiency')

        self.add_param('to_period', 120., desc='take-off period, e.g. 45KW for 2 min', units='s')
        self.add_param('peak_period', 30., desc='peak power period, e.g. 75KW for 30s', units='s')
        self.add_param('climb_period', 600., desc='climb period, e.g. 30KW for 10 min', units='s')
        self.add_param('cruise_period', 600., desc='cruise period, e.g. 20KW for 10 min', units='s')

        self.add_param('Tambient', 49., desc='ambient temperature (120F), includes solar radiation', units='degC')
        self.add_param('Trating', 150., desc='wire temperature rating', units='degC')

        # Ref. EXRAD XLE High Voltage Cable
        # http://www.champcable.com/product_pdfs/12_SAE_150_FX_High_Voltage.pdf
        # No bundle de-rating (only applies to bundles of indivudally
        # insulated wires bundled together)
        # Altitude de-rating Not Applicable below 60,000ft

        # AperStrand (mm^2), Resistance (ohms/km), A40 = current rating at 40degC
        cable_data = [
            {'Wire': 1, 'strands': 779, 'AWG': 30, 'AperStand': 0.0509, 'Resistance': 0.4056, 'A40': 293, 'A71': 270},
            {'Wire': 2, 'strands': 665, 'AWG': 30, 'AperStand': 0.0509, 'Resistance': 0.5114, 'A40': 255, 'A71': 230},
            {'Wire': 4, 'strands': 133, 'AWG': 25, 'AperStand': 0.1624, 'Resistance': 0.8132, 'A40': 190, 'A71': 170},
            {'Wire': 10, 'strands': 1, 'AWG': 10, 'AperStand': 5.26, 'Resistance': 3.276, 'A40': 28.8, 'A71': 28.8}
        ]

        self.add_param('cable_data', cable_data)

        # Equivalent Bundled Wire Table
        # http://www.rapidtables.com/calc/wire/wire-gauge-chart.htm
        # http://www.faa.gov/documentLibrary/media/Advisory_Circular/Chapter_11.pdf
        # Figure 11-4b page 33

        # MatWeb ETP copper
        self.add_param('cpcu', 0.385, desc='Cp of copper', units='J/g*degC')
        self.add_param('rhocu', 8.89, desc='density of copper', units='g/cm^3')
        self.add_param('Resistivity', 1.72e-8, desc='electrical resistivity', units='Ohms/m')

        self.add_param('cpfg', 0.843, desc='Cp of FiberGlass', units='J/g*C')
        self.add_param('rhofg', 0.048, desc='density of fiberglass', units='g/cm^3')
        self.add_param('k_fg', 0.037492, desc='thermal conductivity of fiberglass', units='W/m*degC')

        self.add_param('fg_t', 0.1, desc='fiberglass duct thickness', units='cm')

        # todo lookup function to calc wire area given stand sizes

        self.add_param('wireD', 0.0082, desc='Wire Diameter', units='m')

        self.add_param('n_hl_motors', 6., desc='Number of High Lift Motors')
        self.add_param('hlpower', 10000., desc='distributed motor power during take-off', units='W')

        self.add_output('deltaT', 79.,
                        desc='temperature difference between rated temperature and actual free air temperature',
                        units='degC')
        self.add_output('to_current', 0., units='A')
        self.add_output('peak_current', 0., units='A')
        self.add_output('climb_current', 0., units='A')
        self.add_output('cruise_current', 0., units='A')

        self.add_output('fg_A', 0.1, units='mm*2')
        self.add_output('QperL', 9., desc='Heat per unit length', units='W/m')
        self.add_output('A_dL', 0.0163, desc='Area/Length (circumference)')
        self.add_output('deltaT2', 127., desc='wire temp rise above the environment', units='degC')
        self.add_output('wireTemp', 167., desc='wire Temperature', units='degC')

        self.add_output('cable_out', [0.])
        self.add_output('hl_current', 160., desc='current for all motors on one bus', units='A')

    def solve_nonlinear(self, p, u, r):
        u['to_current'] = p['to_pwr'] / p['efficiency'] / p['busVoltage']
        u['peak_current'] = p['peak_pwr'] / p['efficiency'] / p['busVoltage']
        u['climb_current'] = p['climb_pwr'] / p['efficiency'] / p['busVoltage']
        u['cruise_current'] = p['cruise_pwr'] / p['efficiency'] / p['busVoltage']
        u['deltaT'] = p['Trating'] - p['Tambient']

        print(u['peak_current'])
        # Plot Transient
        time = list(accumulate([
            0., p['to_period'],
            1., p['peak_period'] - 1,
            1., p['climb_period'] - 1,
            1., p['cruise_period'] - 1,
        ]))  # accumulate integrates the durations into a cumulative sum

        toc, pc, cc, crc = u['to_current'], u['peak_current'], u['climb_current'], u['cruise_current']
        current = [toc, toc, pc, pc, cc, cc, crc, crc]

        # In this case the SAE graph shows with a temperature difference
        # of 110C (150C-40C, i.e. rated temperature - environment temperature)
        # the allowable current is 200A. Which mean with 200A going through it,
        # the wire should be at its rated temperature when in a 40C environment.
        # The spreadsheet calculated 167C which is rather close considering
        # the simplified equation used.

        pc = u['peak_current']
        cd = p['cable_data']

        u['fg_A'] = p['fg_t'] * (4.445 + 4.445 + 1.9 + 1.9)  # 1.75"x0.75" w 0.375" radius removed

        fg_heat_cap = u['fg_A'] * p['cpfg'] * p['rhofg']

        for i, row in enumerate(cd):
            row['area'] = cd[i]['strands'] * cd[i]['AperStand']
            row['heat'] = pc ** 2 * (cd[i]['Resistance'] / 1000.)
            row['heat_cap'] = row['area'] * p['cpcu'] * p['rhocu']
            row['roc'] = row['heat'] / row['heat_cap']

        u['A_dL'] = p['wireD'] * pi
        turbulent = 0

        u['QperL'] = cd[3]['heat']
        # table 7-2 pg 253 (Christie photocopy)
        if turbulent:
            u['deltaT2'] = (u['QperL'] * (1. / u['A_dL']) * 1. / 1.24) ** (1 / 1.33)
        else:
            u['deltaT2'] = (u['QperL'] * (1. / u['A_dL']) * (p['wireD'] ** 0.24) / 1.32) ** (1 / 1.25)

        u['wireTemp'] = u['deltaT2'] + p['Tambient']

        u['hl_current'] = p['n_hl_motors'] * p['hlpower'] / p['efficiency'] / p['busVoltage']


if __name__ == '__main__':
    from openmdao.api import Problem, Group
    import matplotlib.pyplot as plt

    p = Problem()
    root = p.root = Group()

    root.add('comp', PowerBus())

    p.setup(check=False)

    # Check Case: Steady State #4AWG 200A 40C
    # p['comp.Tambient'] = 40.
    # p.run()
    # wT = p['comp.wireTemp']
    # print('Wire Temp(C): ', wT , 'Expected: 150degC', 'error %(w.r.t degK):', 100.*(wT - 150.)/(wT+273.), '%')

    # Steady State #4AWG 200A 71C
    # p['comp.Tambient'] = 71.
    # p.run()
    # wT = p['comp.wireTemp']
    # print('Wire Temp(C): ', wT , 'Expected: 150degC', 'error %(w.r.t degK):', 100.*(wT - 150.)/(wT+273.), '%')

    # Steady State #4AWG 120A 49C
    p['comp.Tambient'] = 49.
    p.run()
    wT = p['comp.wireTemp']
    print('Wire Temp(C): ', wT)
    wT = p['comp.wireTemp']
    co = p['comp.cable_data']
    print('Time to reach 73degC (165F) after 21A is applied:', (73 - 56.4) / co[3]['roc'], 'seconds')

    # High Lift Distributed Small Motors

    plot = 0
    if plot:
        plt.plot(time, current)
        plt.xlabel('Elapsed Time (sec)')
        plt.ylabel('Bus Current (A)')
        plt.title('Bus Current vs Time')
        plt.show()

from openmdao.api import Component, ScipyGMRES, Newton, IndepVarComp
from openmdao.units.units import convert_units as cu

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
        # No bundle de-rating (only applies to bundles of indivudally
        # insulated wires bundled together)
        # Altitude de-rating Not Applicable below 60,000ft

        cable_data = [{'Wire':1, 'strands':779, 'AWG':30, 'AperStand':0.0509, 'Resistance':0.0456, 'A40':293, 'A71':270},
        {'Wire':2, 'strands':665, 'AWG':30, 'AperStand':0.0509, 'Resistance':0.5114, 'A40':255, 'A71':230},
        {'Wire':4, 'strands':133, 'AWG':25, 'AperStand':0.1624, 'Resistance':0.8132, 'A40':190, 'A71':170}]

        # self.add_param('cable1', 293., desc='Ampacity at 40degC of copper gauge#1')
        # self.add_param('cable2', 255., desc='Ampacity at 40degC of copper gauge#2')
        # self.add_param('cable4', 190., desc='Ampacity at 40degC of copper gauge#4')
        # self.add_param('cable1l', 270., desc='Ampacity at 71degC of copper gauge#1')
        # self.add_param('cable2l', 230., desc='Ampacity at 71degC of copper gauge#2')
        # self.add_param('cable4l', 170., desc='Ampacity at 71degC of copper gauge#4')

        # Equivalent Bundled Wire Table
        # http://www.rapidtables.com/calc/wire/wire-gauge-chart.htm

        #todo lookup function to calc wire area given stand sizes

        self.add_param('QperL', 33., desc='Heat per unit length', units='W/m')
        self.add_param('wireD', 0.0052, desc='Wire Diameter', units='m')




        self.add_output('deltaT', 79., desc='temperature difference between rated temperature and actual free air temperature', units='degC')
        self.add_output('to_current', 0., units='A')
        self.add_output('peak_current',0., units='A')
        self.add_output('climb_current', 0., units='A')
        self.add_output('cruise_current', 0., units='A')

        self.add_output('A_dL', 0.0163, desc='Area/Length (circumference)')
        self.add_output('deltaT2', 127., desc='wire temp rise above the environment', units='degC')


        def solve_nonlinear(self, p, u, r):
            u['to_current'] = p['to_pwr']/p['efficiency']/p['busVoltage']
            u['peak_current'] = p['peak_pwr']/p['efficiency']/p['busVoltage']
            u['climb_current'] = p['climb_pwr']/p['efficiency']/p['busVoltage']
            u['cruise_current'] = p['cruise_pwr']/p['efficiency']/p['busVoltage']
            u['deltaT'] = p['Trating']-p['Tambient']


            '''In this case the SAE graph shows with a temperature difference
            of 110C (150C-40C, i.e. rated temperature - environment temperature)
            the allowable current is 200A. Which mean with 200A going through it,
            the wire should be at its rated temperature when in a 40C environment.
            The spreadsheet calculated 167C which is rather close considering
            the simplified equation used. '''

            u['A_dL'] = p['wireD']*math.pi()
            turbulent = 0
            if turbulent:
                u['deltaT2'] = (p['QperL']*(1./u['A_dL'])*1./1.24)**(1/1.33)
            else:
                u['deltaT2'] = (p['QperL']*(1./u['A_dL'])*(p['wireD']**0.24)/1.32)**(1/1.25)


if __name__ == '__main__':
    from openmdao.core.problem import Problem
    from openmdao.core.group import Group

    p = Problem()
    root = p.root = Group()

    root.add('comp', PowerBus())

    #Check Case: Steady State #4 200A 40C

    #Steady State #4 200A 71C

    #Steady State #4 120A 71C
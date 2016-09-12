"""
    HEIST Inverter Heat Sink
        Determines the steady state temperature of the HEIST nacelle.
        Calculates Q released/absorbed due to:
        Ambient Natural Convection, Solar Flux In, Radiation Out

    Compatible with OpenMDAO v1.7.1
"""
from math import log, pi, sqrt, e

from openmdao.api import Problem, Group, Component, IndepVarComp
from openmdao.solvers.newton import Newton
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.api import NLGaussSeidel
from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel
from openmdao.solvers.ln_direct import DirectSolver
from openmdao.units.units import convert_units as cu

from pycycle import species_data
from pycycle.species_data import janaf
from pycycle.components import FlowStart
from pycycle.constants import AIR_FUEL_MIX, AIR_MIX
from pycycle.flowstation import FlowIn, PassThrough

class TempBalance(Component):

    def __init__(self):
        super(TempBalance, self).__init__()

        self.add_param('q_total_out', 28., units='W', desc='Total Heat Released via Radiation and Natural Convection') #
        self.add_param('q_total_in', 280., units='W', desc='Total Heat Absorbed/Added via Pods and Solar Absorption') #
        self.add_state('temp_boundary', val=322.0)

        self.deriv_options['type'] = 'fd'

    def solve_nonlinear(self, p, u, r):

        r['temp_boundary'] = (p['q_total_out'] - p['q_total_in'])/p['q_total_out']


class NacelleWallTemp(Component):
    """ Calculates Q released/absorbed by the nacelle """
    def __init__(self, thermo_data=species_data.janaf, elements=AIR_MIX):
        super(NacelleWallTemp, self).__init__()
        self.deriv_options['type'] = 'fd'

        #--Inputs--
        #Parameters/Design Variables
        self.add_param('radius_outer_tube', 0.08, units='m', desc='tube outer radius')
        self.add_param('length_tube', 0.3, units='m', desc='Length of entire nacelle')
        self.add_param('temp_boundary', 322.0, units='K', desc='Average Temperature of the tube wall') #
        self.add_param('temp_outside_ambient', 305.6, units='K', desc='Average Temperature of the outside air') #
        self.add_param('heat_inverter', 220, units='W', desc='Heating Due to a Single Pods') #

        #constants
        self.add_param('solar_insolation', 100., units='W/m**2', desc='solar irradiation at sea level on a clear day') #
        self.add_param('nn_incidence_factor', 0.7, desc='Non-normal incidence factor') #
        self.add_param('surface_reflectance', 0.5, desc='Solar Reflectance Index') #
        self.add_param('emissivity_tube', 0.5, units='W', desc='Emmissivity of the Tube') #
        self.add_param('sb_constant', 0.00000005670373, units='W/((m**2)*(K**4))', desc='Stefan-Boltzmann Constant') #
        self.add_param('Nu_multiplier', 1, desc="fudge factor on nusslet number to account for small breeze on tube")

        #--Outputs--
        self.add_output('diameter_outer_tube', shape=1)
        self.add_output('area_viewing', shape=1)
        self.add_output('q_per_area_solar', 350., units='W/m**2', desc='Solar Heat Rate Absorbed per Area') #
        self.add_output('q_total_solar', 3., units='W', desc='Solar Heat Absorbed by Tube') #
        self.add_output('area_rad', 1.1, units='m**2', desc='Tube Radiating Area')  #
        #Required for Natural Convection Calcs
        self.add_output('GrDelTL3', 1946216.7, units='1/((ft**3)*F)', desc='Heat Radiated to the outside') #
        self.add_output('Pr', 0.707, desc='Prandtl') #
        self.add_output('Gr', 12730351223., desc='Grashof #') #
        self.add_output('Ra', 8996312085., desc='Rayleigh #') #
        self.add_output('Nu', 232.4543713, desc='Nusselt #') #
        self.add_output('k', 0.02655, units='W/(m*K)', desc='Thermal conductivity of air') #
        self.add_output('h', 0.845464094, units='W/((m**2)*K)', desc='Heat Radiated to the outside') #
        self.add_output('area_convection', 1.0, units='m**2', desc='Convection Area') #
        #Natural Convection
        self.add_output('q_per_area_nat_conv', 7.9, units='W/(m**2)', desc='Heat Radiated per Area to the outside') #
        self.add_output('total_q_nat_conv', 50., units='W', desc='Total Heat Radiated to the outside via Natural Convection') #
        #Radiated Out
        self.add_output('q_rad_per_area', 31.6, units='W/(m**2)', desc='Heat Radiated to the outside') #
        self.add_output('q_rad_tot', 10.5, units='W', desc='Heat Radiated to the outside') #
        #Radiated In
        self.add_output('viewing_angle', 1., units='m**2', desc='Effective Area hit by Sun') #
        #Total Heating
        self.add_output('q_total_out', 28., units='W', desc='Total Heat Released via Radiation and Natural Convection') #
        self.add_output('q_total_in', 280., units='W', desc='Total Heat Absorbed/Added via Pods and Solar Absorption') #

    def solve_nonlinear(self, p, u, r):
        """Calculate Various Paramters"""

        u['diameter_outer_tube'] = 2*p['radius_outer_tube']

        #Determine thermal resistance of outside via Natural Convection or forced convection
        if(p['temp_outside_ambient'] < 400):
            u['GrDelTL3'] = 41780000000000000000*((p['temp_outside_ambient'])**(-4.639)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)
        else:
            u['GrDelTL3'] = 4985000000000000000*((p['temp_outside_ambient'])**(-4.284)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)

        #Prandtl Number
        #Pr = viscous diffusion rate/ thermal diffusion rate = Cp * dyanamic viscosity / thermal conductivity
        #Pr << 1 means thermal diffusivity dominates
        #Pr >> 1 means momentum diffusivity dominates
        if (p['temp_outside_ambient'] < 400):
            u['Pr'] = 1.23*(p['temp_outside_ambient']**(-0.09685)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)
        else:
            u['Pr'] = 0.59*(p['temp_outside_ambient']**(0.0239))
        #Grashof Number
        #Relationship between buoyancy and viscosity
        #Laminar = Gr < 10^8
        #Turbulent = Gr > 10^9
        u['Gr'] = u['GrDelTL3']*abs(p['temp_boundary']-p['temp_outside_ambient'])*(u['diameter_outer_tube']**3) #JSG: Added abs incase subtraction goes negative
        #Rayleigh Number
        #Buoyancy driven flow (natural convection)
        u['Ra'] = u['Pr'] * u['Gr']
        #Nusselt Number
        #Nu = convecive heat transfer / conductive heat transfer
        if (u['Ra']<=10**12): #valid in specific flow regime
            u['Nu'] = p['Nu_multiplier']*((0.6 + 0.387*u['Ra']**(1./6.)/(1 + (0.559/u['Pr'])**(9./16.))**(8./27.))**2) #3rd Ed. of Introduction to Heat Transfer by Incropera and DeWitt, equations (9.33) and (9.34) on page 465
        if(p['temp_outside_ambient'] < 400):
            u['k'] = 0.0001423*(p['temp_outside_ambient']**(0.9138)) #SI units (https://mdao.grc.nasa.gov/publications/Berton-Thesis.pdf pg51)
        else:
            u['k'] = 0.0002494*(p['temp_outside_ambient']**(0.8152))
        #h = k*Nu/Characteristic Length
        u['h'] = (u['k'] * u['Nu'])/ u['diameter_outer_tube']
        #Convection Area = Surface Area
        u['area_convection'] = pi * p['length_tube'] * u['diameter_outer_tube']
        #Determine heat radiated per square meter (Q)
        u['q_per_area_nat_conv'] = u['h']*(p['temp_boundary']-p['temp_outside_ambient'])
        #Determine total heat radiated over entire tube (Qtotal)
        u['total_q_nat_conv'] = u['q_per_area_nat_conv'] * u['area_convection']
        #Determine heat incoming via Sun radiation (Incidence Flux)
        #Sun hits an effective rectangular cross section
        u['area_viewing'] = p['length_tube'] * u['diameter_outer_tube']
        u['q_per_area_solar'] = (1-p['surface_reflectance'])* p['nn_incidence_factor'] * p['solar_insolation']
        u['q_total_solar'] = u['q_per_area_solar'] * u['area_viewing']
        #Determine heat released via radiation
        #Radiative area = surface area
        u['area_rad'] = u['area_convection']
        #P/A = SB*emmisitivity*(T^4 - To^4)
        u['q_rad_per_area'] = p['sb_constant']*p['emissivity_tube']*((p['temp_boundary']**4) - (p['temp_outside_ambient']**4))
        #P = A * (P/A)
        u['q_rad_tot'] = u['area_rad'] * u['q_rad_per_area']
        #------------
        #Sum Up
        u['q_total_out'] = u['q_rad_tot'] + u['total_q_nat_conv']
        u['q_total_in'] = u['q_total_solar'] + p['heat_inverter']

        print(u['q_total_out'], 'tot out')
        print(u['q_total_in'], 'tot in')


class Assembly(Group):
    """A top level assembly to connect the two components"""

    def __init__(self, thermo_data=species_data.janaf, elements=AIR_MIX):
        super(Assembly, self).__init__()

        self.add('tm', NacelleWallTemp(), promotes=['radius_outer_tube'])
        self.add('tmp_balance', TempBalance())

        self.connect('tm.q_total_out','tmp_balance.q_total_out')
        self.connect('tm.q_total_in','tmp_balance.q_total_in')
        self.connect('tmp_balance.temp_boundary','tm.temp_boundary')

        self.nl_solver = Newton()
        self.nl_solver.options['atol'] = 1e-8
        self.nl_solver.options['iprint'] = 1
        self.nl_solver.options['rtol'] = 1e-8
        # self.nl_solver.options['maxiter'] = 50
        self.nl_solver.options['maxiter'] = 10

        self.ln_solver = ScipyGMRES()
        self.ln_solver.options['atol'] = 1e-6
        self.ln_solver.options['maxiter'] = 100
        self.ln_solver.options['restart'] = 100

#run stand-alone component
if __name__ == "__main__":

    root = Group()
    root.add('fs', Assembly())

    prob = Problem(root)

    dvars = (
        ('radius', 0.08), #desc='Tube out diameter' #7.3ft
        ('length_tube', 0.4),  #desc='Length of entire nacelle
        ('temp_boundary',340), #desc='Average Temperature of the nacelle') #
        ('temp_outside_ambient',305.6) #desc='Average Temperature of the outside air
        )

    prob.root.add('vars', IndepVarComp(dvars))

    prob.setup()

    # from openmdao.api import view_model
    # view_model(prob)
    # exit()

    prob.root.connect('vars.radius','fs.radius_outer_tube')
    prob.root.connect('vars.length_tube','fs.length_tube')
    prob.root.connect('vars.temp_boundary','fs.temp_boundary')
    prob.root.connect('vars.temp_outside_ambient','fs.temp_outside_ambient')

    prob.run()

    prob.root.list_states()

    #print "temp_boundary: ", prob['root.tm.tmp_balance']

    # print "-----Completed Tube Heat Flux Model Calculations---"
    # print ""
    # print "CompressQ-{} SolarQ-{} RadQ-{} ConvecQ-{}".format(test.tm.total_heat_rate_pods, test.tm.q_total_solar, test.tm.q_rad_tot, test.tm.total_q_nat_conv )
    # print "Equilibrium Wall Temperature: {} K or {} F".format(tesparams['temp_boundary'], cu(tesparams['temp_boundary'],'degK','degF'))
    # print "Ambient Temperature:          {} K or {} F".format(test.tm.temp_outside_ambient, cu(test.tm.temp_outside_ambient,'degK','degF'))
    # print "Q Out = {} W  ==>  Q In = {} W ==> Error: {}%".format(test.tm.q_total_out,test.tm.q_total_in,((test.tm.q_total_out-test.tm.q_total_in)/test.tm.q_total_out)*100)
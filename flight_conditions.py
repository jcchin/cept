#!/usr/bin/env python3
from openmdao.api import Component, Group, ScipyGMRES, Newton, IndepVarComp
from openmdao.units.units import convert_units as cu
from math import pi

from numpy import interp

from pycycle.species_data import janaf
from pycycle.components.flight_conditions import FlightConditions
from pycycle.constants import AIR_MIX
from pycycle.components.atmModels.US1976 import USatm1976

class FC_Calcs(Component):
    def __init__(self):
        super(FC_Calcs, self).__init__()

        self.add_param('alt', 2300., desc='Altitude ASL (assumes standard barometric pressure at sea level', units='ft')
        self.add_param('SLTemp', 100., desc='Temperature at sea level. Ref. FAR Part 23.1043. Cooling tests', units='degF')
        self.add_param('Speed', 105., desc='Vehicle Speed in knots', units='kn')
        self.add_param('Cdia', 0.3, desc='Characteristic Diameter of motor nacelle', units='m')
        self.add_param('Clen', 1.425, desc='Characteristic Length, typically chord', units='m')

        self.add_param('MotPwr', 60000., desc='Motor Power', units='W')
        self.add_param('MotEff', 0.95, desc='Motor efficiency')
        self.add_param('MotSpeed', 2250., desc='Motor Rotations per Minute', units='rpm')

        self.add_output('FLTemp', 92., desc='Temperature at flight level, using a lapse rate of 3.6ft/1kft, per FAR Part 23.1043', units='degF')
        self.add_output('Vsonic', val=1.0, desc="computed speed of sound", units="m/s")
        self.add_output('MN', val=1.0, desc="computed Mach Number")
        self.add_output('AirCond', val=0.0265, desc='Air Thermal Conductivity', units='W/m/K')
        self.add_output('Pr', val=0.7188, desc='Prandtl Number', units='W/m/K')

        self.add_output('MotHeat', val=3000, desc='Motor Waste Heat', units='W')

    def solve_nonlinear(self, p, u, r):
        R_gas_const = 286 # m**2/s**2/K
        gamma = 1.4 # air

        # http://www.engineeringtoolbox.com/dry-air-properties-d_973.html
        tb_t  = 175. ,200. , 225., 250., 275., 300., 325., 350., 375., 400., 450., 500., 550., 600., 650., 700. # temp breakpoints (K)
        tb_tc = 1.593,1.809,2.020,2.227,2.428,2.624,2.816,4.004,3.186,3.365,3.710,4.041,4.357,4.466,4.954,5.236 # Thermal Conductivity (10e-2 W/m/K)
        tb_pr = 0.744,0.736,0.728,0.720,0.713,0.707,0.701,0.697,0.692,0.688,0.684,0.680,0.680,0.680,0.682,0.684 # Prandtl Number

        u['Vsonic'] = self.a = (gamma*R_gas_const*cu(p['SLTemp'],'degF','degK'))**0.5
        u['MN'] = cu(p['Speed'],'kn','m/s')/u['Vsonic']
        t = u['FLTemp'] = p['SLTemp']-3.6*(p['alt']/1000.) # degF
        t2 = cu(t,'degF','degK')                           # degK
        #print(t,t2)

        u['AirCond'] = interp(cu(t,'degF','degK'),tb_t,tb_tc)/100.
        u['Pr'] = interp(cu(t,'degF','degK'),tb_t,tb_pr)

        u['MotHeat'] = p['MotPwr']*(1.-p['MotEff'])


class FC_Group(Group):
    def __init__(self):
        super(FC_Group, self).__init__()

        self.add('amb', FlightConditions()) # atmModel=USatm1976()))
        self.add('calcs', FC_Calcs())

        self.connect('calcs.MN', 'amb.MN_target')
        self.connect('calcs.alt', 'amb.alt')

if __name__ == "__main__":

    from openmdao.api import IndepVarComp, Problem

    p = Problem()
    p.root = Group()
    p.root.add('fc', FC_Group())

    #Define Parameters
    params = (
        ('alt', 2300.0, {'units' : 'ft'}),
        ('Speed', 105., {'units' : 'kn'})
    )

    p.root.add('des_vars', IndepVarComp(params))
    p.root.connect('des_vars.alt','fc.calcs.alt')

    p.setup()

    p.run()

    #print following properties
    print ('Pt : %f Pa' % cu(p['fc.amb.Fl_O:tot:P'],'psi','Pa'))
    print ('rho : %f kg/m**3' % cu(p['fc.amb.Fl_O:tot:rho'],'lbm/ft**3','kg/m**3'))
     # --- Sutherlands Law ---
    mustar = 0.00001716*(cu(p['fc.amb.Fl_O:stat:T'], 'degR', 'degK')/273.15)**1.5*(273.15+110.4)/(cu(p['fc.amb.Fl_O:stat:T'], 'degR', 'degK')+110.4)
    print ('Air Visc : %f Pa*s' % cu(mustar,'kg/m/s','Pa*s'))
    print ('Air Cond : %f' % p['fc.calcs.AirCond'])
    print ('Pr : %f' % p['fc.calcs.Pr'])
    print ('Cp : %f' % cu(p['fc.amb.Fl_O:stat:Cp'],'Btu/(lbm*degR)','J/(kg*K)'))

    p['des_vars.alt'] = 0
    p['des_vars.Speed'] = 0

    p.run()
    print ('Ps : %f Pa SLS' % cu(p['fc.amb.Fl_O:stat:P'],'psi','Pa'))

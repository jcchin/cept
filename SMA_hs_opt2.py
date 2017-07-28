#!/usr/bin/env python3
from openmdao.api import ExplicitComponent, IndepVarComp, ScipyOptimizer, Group, Problem, Component
from openmdao.api import ImplicitComponent, Group, IndepVarComp, DirectSolver, BoundsEnforceLS, ArmijoGoldsteinLS, NewtonSolver, DenseJacobian, NonlinearBlockGS, PetscKSP
from openmdao.utils.units import convert_units as cu
from numpy import tanh, arange, meshgrid, empty, exp, zeros
from numpy import sqrt
import numpy as np
from pycycle.balance import Balance

from scipy.optimize import minimize
# from openmdao.devtools.partition_tree_n2 import view_tree

# http://www.hpl.hp.com/techreports/Compaq-DEC/WRL-86-4.pdf

# k_alum = 230
# k_copper = 400

class HeatSink(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', type_=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_input('MotorPwr', shape=(nn,), val=10E3, desc='total Motor Power', units='W')
        self.add_input('Motor_eff', shape=(nn,), val=0.96, desc='Motor efficiency')
        self.add_input('V_in', shape=(nn,), val = 20, desc = 'flow rate in m/s', units='m/s')
        self.add_input('k_hs', shape=(nn,), val = 230., desc = 'Thermal Conductivity of heat sink', units='W/(m*K)')
        self.add_input('T_max', shape=(nn,), val=100., desc='Maximum Electronics Temp', units='degC')
        self.add_input('T_amb', shape=(nn,), val=20., desc='Ambient Cooling Air Temp', units='degC')
        self.add_input('mu', shape=(nn,), val=0.000018, desc='dynamic viscosity',units='Pa*s')
        self.add_input('Cp', shape=(nn,), val=1004.088235, desc='specific heat at altitude', units='J/(kg*K)')
        self.add_input('k_air', shape=(nn,), val=0.026726, desc='thermal conductivity of air', units='W/(m*K)')
        self.add_input('R_inner', shape=(nn,), val=27.5E-3, desc='inner radius of the heat sink', units='m')
        self.add_input('R_outer', shape=(nn,), val=50E-3, desc='outer radius of the heat sink', units='m')
        self.add_input('gamm_eff', shape=(nn,), val=1.8, desc='boundary layer thickness inputeter, dimensionless')
        self.add_input('nu', shape=(nn,), val=0.0000168, desc='kinematic viscosity',units='m**2/s')
        self.add_input('sink_l', shape=(nn,), val=.025, desc='Heat Sink Base Length', units='m')
        self.add_input('rho', shape=(nn,), val=1.158128, desc='density of air', units='kg/m**3')
        self.add_input('k_metal', shape=(nn,), val=230, desc='thermal conductivity of metal bw coils and hs', units='W/(m*K)')
        self.add_input('l_metal', shape=(nn,), val=0.007, desc='length of metal bw coils and hs', units='m')
        self.add_input('A_metal', shape=(nn,), val=0.007, desc='Area of metal bw coils and hs', units='m')
        self.add_input('grease_thickness', shape=(nn,), val=0.0001, desc='thickness of thermal grease', units='m')
        self.add_input('k_grease', shape=(nn,), val=1.5, desc='Thermal conductivity of grease', units='W/(m*K)')
        self.add_input('case_thickness', shape=(nn,), val=0.0001, desc='thickness of thermal grease', units='m')
        self.add_input('k_case', shape=(nn,), val=200., desc='Thermal conductivity of case', units='W/(m*K)')
        self.add_input('fin_gap', shape=(nn,), val=0.002, desc='spacing between fins', units='m')
        self.add_input('fin_w', shape=(nn,), val=0.002, desc='fin thickness', units='m')

        self.add_output('Nu', val=10., desc='Nusselt #')
        self.add_output('Re', val=10., desc='Reynolds #')
        self.add_output('Pr', val=10., desc='Prandtl #')
        self.add_output('fin_h', val=.032, desc='Fin height', units='m')
        self.add_output('lambda', val=2., desc='dimensionless channel height')
        self.add_output('V_fin',val=0.05, desc='velocity through the fins', units='m/s')
        self.add_output('alpha', val=37., desc='maximum increase in effective heat transfer area, dimensionless')
        self.add_output('R_max',val=0.05, desc='Max Thermal Resistance', units='degK/W')
        self.add_output('R_tot1', val = 0.01, desc='Total Thermal Resistance', units='degK/W')
        self.add_output('R_tot', val = 0.01, desc='Total Thermal Resistance', units='degK/W')
        self.add_output('V_dot', val = 0.01, desc='Volumetric Flow Rate through sink', units='m**3/s')
        self.add_output('n_fins', val=20, desc='number of fins')
        self.add_output('dP', val=1.0, desc='pressure drop based on friction', units='Pa')
        self.add_output('h', val=20., desc='fin heat transfer coefficient', units='W/(m**2*K)')
        self.add_output('omega', val=0.0554, desc='normalized thermal Resistance')
        self.add_output('R_lossless', val=0.845, desc='thermal resistance of optimal lossless heat sink', units='degK/W')
        self.add_output('R_global', val=1.28, desc='global upper bound on thermal resistance of an optimal sink', units='degK/W')
        self.add_output('zeta_global', val=1.512, desc='effectiveness ratio of global upper bound, dimensionless')
        self.add_output('R_thin', val=1.15, desc='thin fin upper bound on thermal resistance of an optimal sink', units='degK/W')
        self.add_output('zeta_thin', val=1.361, desc='effectiveness ratio of thin fin upper bound, dimensionless')
        self.add_output('R_case', val=0.0001, desc='inverter case thermal resistance', units='degK/W')
        self.add_output('R_grease', val=0.004, desc='grease thermal resistance', units='degK/W')

    def compute(self, inputs, outputs):

        b = inputs['fin_gap']
        l = inputs['sink_l']

        # Geometry
        H = outputs['fin_h'] = inputs['R_outer'] - inputs['R_inner']
        sink_w = 2 * np.pi*inputs['R_outer']
        outputs['n_fins'] = sink_w/(inputs['fin_w']+inputs['fin_gap'])
        #V = outputs['V_in'] = inputs['cfm']/(inputs['sink_w']*H-(H*outputs['n_fins']*inputs['fin_w']))
        outputs['V_fin'] = inputs['V_in']*(sink_w/(outputs['n_fins']*b))

        Q = inputs['MotorPwr']*(1-inputs['Motor_eff'])
        outputs['R_max'] = (inputs['T_max'] - inputs['T_amb'])/Q

        # flow regime
        Pr = outputs['Pr'] = (inputs['mu']*inputs['Cp'])/inputs['k_air']
        AR = inputs['fin_gap']/H
        Nu = 2.656*inputs['gamm_eff']*Pr**(1./3.) #non-ducted
        Nu = 8.325 * (1. - 2.042 * AR + 3.085 * AR**2 - 2.477 * AR**3)
        outputs['Nu'] = Nu
        Dh = 2* b

        # heat transfer, method 1
        outputs['h'] =(Nu * inputs['k_hs'])/Dh
        #print(outputs['h'],Nu,Dh)
        A = 2*H*outputs['n_fins']*l
        Abase = l*sink_w
        outputs['R_tot1'] = 1/(outputs['h']*A)


        # calculate thermal resistance, method 2
        outputs['alpha'] = sqrt(inputs['k_hs']/(inputs['k_air']*outputs['Nu']))
        lam = outputs['lambda'] = H/(inputs['fin_gap']*outputs['alpha'])
        outputs['omega'] = H / (inputs['k_hs'] * Abase)
        outputs['R_lossless'] = outputs['omega']*lam**-2.
        outputs['zeta_global'] = 2.*lam + 1.
        outputs['R_global'] = outputs['zeta_global'] * outputs['R_lossless']
        outputs['zeta_thin'] = sqrt(lam)*(lam+1.)/tanh(sqrt(lam))
        outputs['R_thin'] = outputs['zeta_thin'] * outputs['R_lossless']
        outputs['R_case'] = inputs['case_thickness']/(inputs['k_case']*Abase)
        outputs['R_grease'] = inputs['grease_thickness']/(inputs['k_grease']*Abase)
        if lam >=1:
            outputs['R_tot'] = outputs['R_case'] + outputs['R_grease'] + outputs['R_global']
        else:
            outputs['R_tot'] = outputs['R_case'] + outputs['R_grease'] + outputs['R_thin']


        s = 1.- (outputs['n_fins']*inputs['fin_w'])/sink_w
        Ke = (1.-s**2.)**2.
        Kc = 0.42 * (1-s**2)
        Re = outputs['Re'] = (inputs['rho']*inputs['V_in']*b*b)/(inputs['mu']*l)
        Lstar2 = (l/Dh)/Re
        lamb = b/H
        f = (24.-32.527*lamb + 46.721*lamb**2. - 40.829*lamb**3. + 22.954*lamb**4 - 6.089*lamb**5 ) / Re
        f_app = sqrt(((3.44/sqrt(Lstar2))**2. + (f*Re)**2.))/Re
        rho = inputs['rho']
        outputs['dP'] = ((Kc+4.*f_app*(l/Dh)+Ke)*rho*(inputs['V_in']**2.)/2.)


# class HeatSinkOpt(Group):

#   def initialize(self):
#         self.metadata.declare('num_nodes', type_=int)

#   def setup(self):
#         nn = self.metadata['num_nodes']
#         self.add_subsystem('hs', HeatSink(num_nodes=1))


#    # newton = self.nonlinear_solver = NewtonSolver()
#         newton.options['atol'] = 1e-6
#         newton.options['rtol'] = 1e-6
#         newton.options['iprint'] = 2
#         newton.options['maxiter'] = 50
#         newton.options['solve_subsystems'] = True
#         newton.options['max_sub_solves'] = 100
#         newton.linesearch = BoundsEnforceLS()
#         # newton.linesearch = ArmijoGoldsteinLS()
#         # newton.linesearch.options['c'] = .0001
#         newton.linesearch.options['maxiter'] = 1
#         newton.linesearch.options['bound_enforcement'] = 'scalar'
#         newton.linesearch.options['iprint'] = -1


        # self.linear_solver = DirectSolver()





if __name__ == '__main__':

    def runO(x):
        p = Problem()
        p.model.add_subsystem('hs', HeatSink(num_nodes=1))

        p.setup(check=False)

        p['hs.fin_gap'] = x[0]
        p['hs.fin_w'] = x[1]

        p.run_model()



        def printstuff():
            print('=============')
            print('V_in (m/s) : %f' %p['hs.V_in'])
            print('V_fin (m/s) : %f' %p['hs.V_fin'])
            print('Nu : %f' %p['hs.Nu'])
            print('Pr : %f' %p['hs.Pr'])
            print('Re : %f' %p['hs.Re'])
            print('fin height: %f' %p['hs.fin_h'])
            print('# of fins : %f' %p['hs.n_fins'])
            print('-------------')
            print('fin thickness (m): %f' % p['hs.fin_w'])
            print('fin gap (m): %f' % p['hs.fin_gap'])
            print('Volumetric Flow Rate (m^3/s): %f' % p['hs.V_dot'])
            print('Maximum thermal resistance (K/W): %f' %p['hs.R_max'])
            print('Actual total thermal resistance (K/W): %f' % p['hs.R_tot'])
            #print('Actual total thermal resistance1 (K/W): %f' % p['hs.R_tot1'])
            #print('h %f' % p['hs.h'])
            print('Pressure Drop (Pa): %f' % p['hs.dP'])
            print()
        printstuff()

        return np.array([p['hs.R_tot']*10000 + p['hs.dP']*1.4])


    bnds = ((0.001, None), (0.001, None)) # limit fin_gap and fin_thick

    res = minimize(runO, (0.002, 0.002), bounds=bnds)



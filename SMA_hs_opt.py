#!/usr/bin/env python3
from openmdao.api import ExplicitComponent, IndepVarComp, ScipyOptimizer, Group, Problem, Component
from openmdao.api import ImplicitComponent, Group, IndepVarComp, DirectSolver, BoundsEnforceLS, ArmijoGoldsteinLS, NewtonSolver, DenseJacobian, NonlinearBlockGS, PetscKSP
from openmdao.utils.units import convert_units as cu
from numpy import tanh, arange, meshgrid, empty, exp, zeros
from numpy import sqrt
import numpy as np
from pycycle.balance import Balance
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


        self.add_output(name='R_max',val=0.05, desc='Max Thermal Resistance', units='degK/W')
        self.add_output(name='R_tot', val = 0.01, desc='Total Thermal Resistance', units='degK/W')
        self.add_output(name='V_dot', val = 0.01, desc='Volumetric Flow Rate through sink', units='m**3/s')
        self.add_output(name='fin_gap', val=0.001, desc='spacing between fins', units='m')
        self.add_output(name='fin_w', val=0.001, desc='fin thickness', units='m')
        self.add_output(name='n_fins', val=20, desc='number of fins')
        self.add_output(name='dP', val=1.0, desc='pressure drop based on friction', units='Pa')
        self.add_output(name='h', val=20., desc='fin heat transfer coefficient', units='W/(m**2*K)')

    def compute(self, inputs, outputs):

        Q = inputs['MotorPwr']*(1-inputs['Motor_eff'])
        outputs['R_max'] = (inputs['T_max'] - inputs['T_amb'])/Q

        Pr = (inputs['mu']*inputs['Cp'])/inputs['k_air']
        Nu = 2.656*inputs['gamm_eff']*Pr**(1./3.) #non-ducted
        H = inputs['R_outer'] - inputs['R_inner']
        sink_w = 2 * np.pi*inputs['R_outer']
        Abase = sink_w * inputs['sink_l']


        # this section changes depending on whether you're putting a user defined heatsink in or optimizing based on velocity
        V = inputs['V_in']
        b = outputs['fin_gap'] = 0.002 # user input
        # b = outputs['fin_gap'] = 2.*inputs['gamm_eff']*sqrt((inputs['nu']*inputs['sink_l'])/V) # Optimized fin gap
        outputs['fin_w'] = .002 # user input
        # outputs['fin_w'] = H/alpha # optimize, actual ideal will be a bit smaller, accurate with small lambda
        N_fin = outputs['n_fins'] = sink_w/(outputs['fin_w']+outputs['fin_gap'])

        alpha = sqrt(inputs['k_hs']/(inputs['k_air']*Nu))
        lam = H/(outputs['fin_gap']*alpha)
        omega = H / (inputs['k_hs'] * Abase)
        Dh = 2.*b
        mu = inputs['mu']
        l = inputs['sink_l']
        Re = (inputs['rho']*V*b*b)/(mu*l)
        Lstar = l/(Dh*Re*Pr)

        # R_lossless = omega*lam**-2.
        # zeta_global = 2.*lam + 1.
        # R_global = zeta_global * R_lossless

        # zeta_thin = sqrt(lam)*(lam+1.)/tanh(sqrt(lam))
        # R_thin = zeta_thin * R_lossless

        R_metal = inputs['l_metal']/(inputs['k_metal']*.01*sink_w) #same as heat sink base

        # outer skin calcs
        Nu_out = 0.664*Re**(0.5)*Pr**(1/3)
        h_conv_motor = 30#(Nu_out*inputs['k_air'])/.04
        Iron_cond = 79.5 # W/m*K
        Iron_t = .0035 # m
        Al_cond = 230  # W/m*K      # aluminum conductivity
        Al_t = .004 #m
        L_motor = 0.04 #m
        C_motor = 2*np.pi*.134 #
        R_skin = ((Iron_t/Iron_cond)+(Al_t/Al_cond)+(h_conv_motor)**-1)/(L_motor) # K/W (outer skin)


        s = 1.- (outputs['n_fins']*outputs['fin_w'])/sink_w
        Ke = (1.-s**2.)**2.
        Kc = 0.42 * (1-s**2)
        Lstar2 = (l/Dh)/Re
        lamb = b/H
        f = (24.-32.527*lamb + 46.721*lamb**2. - 40.829*lamb**3. + 22.954*lamb**4 - 6.089*lamb**5 ) / Re
        f_app = sqrt(((3.44/sqrt(Lstar2))**2. + (f*Re)**2.))/Re
        rho = inputs['rho']
        outputs['dP'] = ((Kc+4.*f_app*(l/Dh)+Ke)*rho*(V**2.)/2.)

        # outputs['h'] = Nu*inputs['k_air']/Dh

        # b = 0.01
        t_fin = outputs['fin_w']


        Re = (inputs['rho']*V*b*b)/(mu*l)
        Nu = ((((Re*Pr)/2)**-3)+((0.664*sqrt(Re)*Pr**0.33*sqrt(1+(3.65/sqrt(Re)))**3)**-1))**-.33
        AR = b/H
        Nu = 8.235*(1-2.042*AR+3.085*AR**2-2.447*AR**3)
        w_gap = b
        # t_fin = b

        h = Nu*inputs['k_air']/b
        x = (2*h)/(inputs['k_hs']*t_fin)
        m = sqrt(x)
        eff_fin = tanh(m*H)/(m*H)
        R_hs = (h*(Abase+(N_fin*eff_fin*.025*.0225)))**-1
        # outputs['R_tot'] = (((R_hs + R_metal)**-1)+((R_skin)**-1))**-1
        outputs['R_tot'] = R_skin #(no heat sink)

        print(R_hs)
        # h = (Nu * inputs['k_air'])/l
        print(h)
        outputs['V_dot'] = inputs['V_in'] * outputs['fin_gap'] * H * outputs['n_fins']
        T_final = (outputs['R_tot']*Q)+inputs['T_amb']
        print(T_final)

# class HeatSinkOpt(Group):

#   def initialize(self):
#         self.metadata.declare('num_nodes', type_=int)

#   def setup(self):
#         nn = self.metadata['num_nodes']
#    self.add_subsystem('hs', HeatSink(num_nodes=1))
#    self.add_subsystem('balance', Balance(units="m/s", input_units="degK/W"))

#    self.connect('balance.indep', 'hs.V_in')
#    self.connect('hs.R_max', 'balance.lhs')
#    self.connect('hs.R_tot', 'balance.rhs')

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

    p = Problem()
    p.model.add_subsystem('hs', HeatSink(num_nodes=1))

    p.setup(check=False)
    def printstuff():
        print('=============')
        print('V_in (m/s) : %f' %p['hs.V_in'])
        # print('Nu : %f' %p['hs.Nu'])
        # print('Pr : %f' %p['hs.Pr'])
        # print('Re : %f' %p['hs.Re'])
        # print('L* : %f' %p['hs.Lstar'])
        # print('alpha : %f' %p['hs.alpha'])
        # print('fin height: %f' %p['hs.fin_h'])
        # print('lambda : %f' %p['hs.lambda'])
        # print('omega : %f' %p['hs.omega'])
        # print('R_max : %f' %p['hs.R_max'])
        # print('R_global: %f' % p['hs.R_global'])
        # print('zeta_thin : %f' %p['hs.zeta_thin'])
        print('# of fins : %f' %p['hs.n_fins'])
        # print('-------------')
        print('fin thickness (m): %f' % p['hs.fin_w'])
        print('fin gap (m): %f' % p['hs.fin_gap'])
        print('Volumetric Flow Rate (m^3/s): %f' % p['hs.V_dot'])
        print('Maximum thermal resistance (K/W): %f' %p['hs.R_max'])
        print('Actual total thermal resistance (K/W): %f' % p['hs.R_tot'])
        print('Pressure Drop (Pa): %f' % p['hs.dP'])
        # print('h %f' % p['hs.h'])
        # print('R_max %f' % p['hs.R_max'])
    p.run_model()

    printstuff()










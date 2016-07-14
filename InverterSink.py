from openmdao.api import Component, ScipyGMRES, NLGaussSeidel, IndepVarComp, ScipyOptimizer
from openmdao.units.units import convert_units as cu
from numpy import tanh, arange
from numpy import sqrt
from openmdao.devtools.partition_tree_n2 import view_tree

# http://www.electronics-cooling.com/2003/02/estimating-parallel-plate-fin-heat-sink-thermal-resistance/
# http://www.electronics-cooling.com/2003/05/estimating-parallel-plate-fin-heat-sink-pressure-drop/
# Simons, R.E., "Estimating Parallel Plate-Fin Heat Sink Thermal Resistance," ElectronicsCooling, Vol. 9, No. 1, pp. 8-9, 2003.
#Culham, J.R., and Muzychka, Y.S. "Optimization of Plate Fin Heat Sinks Using Entropy Generation Minimization," IEEE Trans. Components and Packaging Technologies, Vol. 24, No. 2, pp. 159-165, 2001.
#Simons, R.E., and Schmidt, R.R., "A Simple Method to Estimate Heat Sink Air Flow Bypass," ElectronicsCooling, Vol. 3, No. 2, pp. 36-37, 1997.

class Balance(Component):
    def __init__(self):
        super(Balance, self).__init__()
        self.deriv_options['type'] = "fd"
        self.add_state('V_0', 10.,   units="m/s")
        self.add_state('h_0', 20.)

        self.add_param('V_avg', 20.)
        self.add_param('h', 20.)

        #self.add_state('fin_w')

    def solve_nonlinear(self, p,u,r):

        u['V_0'] = p['V_avg']
        u['h_0'] = p['h']

    def apply_nonlinear(self, p,u,r):
        r['V_0'] = (p['V_avg'] - u['V_0'])
        r['h_0'] = (p['h'] - u['h_0'])

class InverterSink(Component):
    def __init__(self):
        super(InverterSink, self).__init__()
        self.deriv_options['type'] = "fd"
        self.add_param('InvPwr', 45000.0, desc='total power through the inverter', units='W')
        self.add_param('Inv_eff', 0.98, desc='inverter efficiency')
        self.add_param('T_air', 47., desc='Cooling Air Temp', units='degC')
        self.add_param('T_max', 80., desc='Maximum Electronics Temp', units='degC')
        self.add_param('sink_w', .192, desc='Heat Sink Base Width', units='m')
        self.add_param('sink_l', .160, desc='Heat Sink Base Length', units='m')
        self.add_param('sink_h', .002, desc='Heat Sink Base Height', units='m')
        self.add_param('fin_w', 0.0009, desc='Heat sink fin width', units='m')
        self.add_param('V_0', 20., desc='initial average velocity guess', units='m/s')
        self.add_param('h_0', 20., desc='initial average velocity guess', units='W/(m**2*K)')
        self.add_param('rho', 1.158128, desc='density of air', units='kg/m**3')
        self.add_param('fin_gap', 0.001, desc='Min spacing between fins', units='m')
        self.add_param('mu', 0.000018, desc='dynamic viscosity',units='Pa*s')
        self.add_param('Cp', 1004.088235, desc='specific heat at altitude', units='J/(kg*K)')
        self.add_param('k_air', 0.026726, desc='thermal conductivity of air', units='W/(m*K')
        self.add_param('k_fin', 200.0, desc='thermal conductivity of aluminum', units='W/(m*K)' )
        self.add_output('Dh', 1.0, desc='hydraulic diameter', units='m')
        self.add_output('Lstar', 1.0, desc='characteristic length', units='m')
        self.add_output('sigma', 1.0, desc='ratio of flow channel areas to approaching flow area')
        self.add_output('Kc', 1.0, desc='pressure loss coefficient due to sudden flow compression')
        self.add_output('Ke', 1.0, desc='pressure loss coefficient due to sudden flow expansion')
        self.add_output('f', 1.0, desc='friction factor for fully developed laminar flow, based on fin aspect ratio')
        self.add_output('f_app', 1.0, desc='apparent friction factor for hydrodynamically developing \
                                laminar flow, based on f')
        # material props
        self.add_param('case_t', 150., desc='thermal conductivity of the casing, Al-6061-T6', units='W/(m*K)' )
        self.add_param('fin_t',220., desc='thermal conductivity of the fins, Al-1100', units='W/(m*K)' )
        self.add_output('h', 20., desc='fin heat transfer coefficient', units='W/(m**2*K)')
        self.add_output('Re', 10., desc='Reynolds #')
        self.add_output('Pr', 10., desc='Prandtl #')
        self.add_output('Nu', 10., desc='Nusselt #')
        self.add_output('Rmax', 0.05, desc='Max Thermal Resistance', units='degC/W')
        self.add_output('R_tot', 0.05, desc='Total Thermal Resistance', units='degC/W')
        self.add_output('fin_eff', 1.0, desc='Fin Efficiency (based on Aspect ratio)')
        self.add_output('R_hs', 0.05, desc='Heat Sink Resistance', units='degC/W')
        self.add_output('n_fins', 137., desc='Number of fins that can fit on the sink')
        self.add_output('min_area', 1.4, desc='Minimum cooling area needed', units='m**2')
        self.add_output('fin_h', .032, desc='Fin height', units='m')
        # Pressure drop calcs
        self.add_output('Q', 10., desc='volumetric flow rate', units='m**3/s')
        self.add_output('dP1', 1.0, desc='pressure drop based on friction', units='Pa')
        #self.add_output('dP2', 0.0, desc='pressure drop based on velocity', units='Pa')
        self.add_output('V_out', 5., desc='exit velocity', units='m/s')
        self.add_output('lambda', 2., desc='aspect ratio')
        self.add_output('V_avg', 20., desc='computed average velocity guess', units='m/s')

    def solve_nonlinear(self, p, u, r):

        pwr = p['InvPwr']*(1.-p['Inv_eff'])
        Abase = p['sink_w']*p['sink_l']
        u['n_fins'] = p['sink_w']/(p['fin_w']+p['fin_gap'])
        u['min_area'] = pwr/(p['h_0']*(p['T_max']-p['T_air']))
        u['fin_h'] = 0.5*u['min_area']/(u['n_fins']*p['sink_l'])
        # print(p['fin_w'],p['fin_gap'])
        # print(u['n_fins'])
        h = u['fin_h']
        H = h + p['sink_h']
        b = p['fin_gap']
        E_Abase = (u['n_fins']-1.)*b*p['sink_l']
        Afin = 2*u['fin_h']*p['sink_l']
        u['Rmax'] = p['T_max'] - p['T_air'] / pwr
        V = p['V_0']
        mu = p['mu']
        l = p['sink_l']
        Pr = u['Pr'] = (mu*p['Cp'])/p['k_air']
        # Valid range for Re is 0.26 <Reb < 175
        Re = u['Re'] = (p['rho']*V*b*b)/(mu*l)
        # Teertstra, P., Yovanovich, M.M., and Culham, J.R., "Analytical Forced Convection Modeling of Plate Fin Heat Sinks," Proceedings of 15th IEEE Semi-Therm Symposium, pp. 34-41, 1999.
        Nu = u['Nu'] = ((1/(((Re*Pr)/2)**3))+(1/(0.644*sqrt(Re)*Pr**0.33*sqrt(1+(3.65/(sqrt(Re)))))**3))**-0.33
        u['h'] = Nu*p['k_air']/b # heat transfer, not to be confused with fin height
        m = sqrt(2.*h/p['k_fin']*p['fin_t'])
        u['fin_eff'] = tanh(m*h)/(m*h)
        u['R_hs'] = 1./(u['h']*(E_Abase+u['n_fins']*u['fin_eff']*Afin))
        u['R_tot'] = u['R_hs'] + ((H-h)/(p['k_fin']*Abase))
        #u['G'] = p['V_in']*u['n_fins']*b*h
        s = u['sigma'] = 1- (u['n_fins']*p['fin_w'])/p['sink_w']
        u['Ke'] = (1-s**2)**2
        u['Kc'] = 0.42 * (1-s**2)
        Dh = u['Dh'] = 2.*b
        u['Lstar'] = (l/Dh)/Re
        lamb = u['lambda'] = b/h
        u['f'] = (24.-32.527*lamb + 46.721*lamb**2 - 40.829*lamb**3 + 22.954*lamb**4 - 6.089*lamb**5 ) / Re
        u['f_app'] = sqrt(((3.44/sqrt(u['Lstar']))**2 + (u['f']*Re)**2))/Re
        rho = p['rho']
        u['dP1'] = (u['Kc']+4*u['f_app']*(l/Dh)+u['Ke'])*rho*(V**2)/2.
        #volumetric flow rate through a single gap
        u['Q'] = (h*u['dP1']*p['fin_gap']**3)/(12*mu)
        #assumed same velocity through all
        u['V_avg'] = u['Q']/(p['fin_gap']*h)

        print "foobar", u['V_avg'], p['V_0']

        #u['dP2'] = (rho/2.)*(u['V_out']**2.-p['V_in']**2.)


if __name__ == '__main__':
    from openmdao.core.problem import Problem
    from openmdao.core.group import Group
    from openmdao.api import ScipyGMRES

    p = Problem()
    root = p.root = Group()
    root.add('balance', Balance())
    root.add('inverter', InverterSink())

    root.connect('inverter.V_avg','balance.V_avg')
    root.connect('balance.V_0', 'inverter.V_0')

    root.connect('inverter.h','balance.h')
    root.connect('balance.h_0', 'inverter.h_0')

    # driver = p.driver = ScipyOptimizer()
    # driver.options['optimizer'] = 'SLSQP'
    params = (('fin_gap', 0.001, {'units': 'm'}),
            ('fin_w', 0.0009, {'units': 'm'}))
    root.add('input_vars', IndepVarComp(params))
    root.connect('input_vars.fin_gap', 'inverter.fin_gap')
    root.connect('input_vars.fin_w', 'inverter.fin_w')
    #driver.add_desvar('input_vars.fin_gap', lower=.7, scaler=1.0)
    #driver.add_desvar('input_vars.fin_w', lower=.7, scaler=1.0)
    #driver.add_objective('inverter.R_hs')
    #top.driver.add_constraint('con1.c1', lower=0.0, scaler=1000.0)
    #top.driver.add_constraint('con2.c2', lower=0.0)
    root.ln_solver = ScipyGMRES()
    root.nl_solver = NLGaussSeidel()
    root.nl_solver.options['maxiter'] = 5
    # root.nl_solver.options['iprint'] = 1
    #p.print_all_convergence()

    p.root.set_order(['input_vars', 'inverter', 'balance'])
    p.setup()
    #root.list_connections()
    # view_tree(p)
    # exit()
    p.run()
    print('# of fins : %f' %p['inverter.n_fins'])
    print(p['inverter.h'], p['inverter.min_area'], p['inverter.Re'])
    print('R_hs : %f' %p['inverter.R_hs'])
    print('V_0, V_avg : %f, %f' %(p['inverter.V_0'],p['inverter.V_avg']))
    # print('Min Area : %f m^2' %p['inverter.min_area'])
    # print('Heat Sink Conductivity (h) : %f W/(m**2*K)' %p['inverter.fin_h'])
    print('Reynolds # : %f , which must be between 0.26 <Reb < 175' % p['inverter.Re'])
    print('--------------')
    for w in arange(0.0007,.002,0.01):
        p['input_vars.fin_gap'] = w



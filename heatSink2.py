from openmdao.api import Component, ScipyGMRES, NLGaussSeidel, IndepVarComp, ScipyOptimizer
from openmdao.units.units import convert_units as cu
from numpy import tanh, arange
from numpy import sqrt
from openmdao.devtools.partition_tree_n2 import view_tree

# http://www.hpl.hp.com/techreports/Compaq-DEC/WRL-86-4.pdf

class HeatSink(Component):
    def __init__(self):
        super(HeatSink, self).__init__()
        self.deriv_options['type'] = "fd"

        self.add_param('InvPwr', 45000.0, desc='total power through the inverter', units='W')
        self.add_param('Inv_eff', 0.98, desc='inverter efficiency')
        self.add_param('T_air', 47., desc='Cooling Air Temp', units='degC')
        self.add_param('T_max', 80., desc='Maximum Electronics Temp', units='degC')
        self.add_param('sink_w', .192, desc='Heat Sink Base Width', units='m')
        self.add_param('sink_l', .160, desc='Heat Sink Base Length', units='m')
        self.add_param('sink_h', .002, desc='Heat Sink Base Height', units='m')
        self.add_param('avail_h', 0.1, desc='maximum available height for heat sink', units='m')
        self.add_param('gamm_eff', 1.8, desc='boundary layer thickness parameter, dimensionless')
        self.add_param('V_in', 20., desc='initial velocity', units='m/s')
        self.add_param('h_0', 20., desc='initial average velocity guess', units='W/(m**2*K)')
        self.add_param('rho', 1.158128, desc='density of air', units='kg/m**3')
        self.add_param('mu', 0.000018, desc='dynamic viscosity',units='Pa*s')
        self.add_param('nu', 0.0000168, desc='kinematic viscosity',units='m**2/s')
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
        self.add_output('lambda', 2., desc='dimensionless channel height')
        self.add_output('V_avg', 20., desc='computed average velocity guess', units='m/s')
        self.add_output('omega', 0.0554, desc='normalized thermal Resistance')
        self.add_output('R_lossless', 0.845, desc='thermal resistance of optimal lossless heat sink', units='degC/W')
        self.add_output('R_global', 1.28, desc='global upper bound on thermal resistance of an optimal sink', units='degC/W')
        self.add_output('zeta_global', 1.512, desc='effectiveness ratio of global upper bound, dimensionless')
        self.add_output('R_thin', 1.15, desc='thin fin upper bound on thermal resistance of an optimal sink', units='degC/W')
        self.add_output('zeta_thin', 1.361, desc='effectiveness ratio of thin fin upper bound, dimensionless')
        self.add_output('alpha', 37., desc='maximum increase in effective heat transfer area, dimensionless')
        self.add_output('fin_gap', 0.001, desc='spacing between fins', units='m')
        self.add_output('fin_w', 0.0009, desc='Heat sink fin width', units='m')

    def solve_nonlinear(self, p, u, r):

        pwr = p['InvPwr']*(1.-p['Inv_eff'])
        Abase = p['sink_w']*p['sink_l']
        gam_e = p['gamm_eff']
        #u['min_area'] = pwr/(p['h_0']*(p['T_max']-p['T_air']))
        H = u['fin_h'] = 0.5*u['min_area']/(u['n_fins']*p['sink_l'])

        Pr = u['Pr'] = (p['mu']*p['Cp'])/p['k_air']
        u['Nu'] = 2.656*gam_e*Pr**(1./3.) #non-ducted
        u['alpha'] = sqrt(p['k_fin']/(p['k_air']*u['Nu']))

        u['fin_gap'] = 2.*gam_e*sqrt((p['nu']*p['sink_l'])/p['V_in'])
        lam = u['lambda'] = H/(u['fin_gap']*u['alpha'])
        u['omega'] = H / p['k_fin'] * Abase

        u['R_lossless'] = u['omega']*lam**-2.
        u['zeta_global'] = 2.*lam + 1.
        u['R_global'] = u['zeta_global'] * u['R_lossless']

        u['zeta_thin'] = sqrt(lam)*(lam+1.)/tanh(sqrt(lam))
        u['R_thin'] = u['zeta_global'] * u['R_lossless']

        u['fin_w'] = H/u['alpha'] # actual ideal will be a bit smaller, accurate with small lambda
        u['n_fins'] = p['sink_w']/(u['fin_w']+p['fin_gap'])

if __name__ == '__main__':
    from openmdao.core.problem import Problem
    from openmdao.core.group import Group
    from openmdao.api import ScipyGMRES

    p = Problem()
    root = p.root = Group()
    root.add('hs', HeatSink())

    params = (('sink_w', 0.001, {'units': 'm'}),
            ('sink_l', 0.0009, {'units': 'm'}))
    root.add('input_vars', IndepVarComp(params))
    root.connect('input_vars.sink_w', 'hs.sink_w')
    root.connect('input_vars.sink_l', 'hs.sink_l')
    p.setup()
    #root.list_connections()
    # view_tree(p)
    # exit()
    p.run()
    print('# of fins : %f' %p['inverter.n_fins'])
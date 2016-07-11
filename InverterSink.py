from openmdao.api import Component, ScipyGMRES, Newton, IndepVarComp
from openmdao.units.units import convert_units as cu
from numpy import tanh
from numpy import sqrt

# http://www.electronics-cooling.com/2003/02/estimating-parallel-plate-fin-heat-sink-thermal-resistance/
# http://www.electronics-cooling.com/2003/05/estimating-parallel-plate-fin-heat-sink-pressure-drop/

class InverterSink(Component):
    def __init__(self):
        super(InverterSink, self).__init__()
        self.deriv_options['type'] = "cs"

        self.add_param('InvPwr', 45000.0, desc='total power through the inverter', units='W')
        self.add_param('Inv_eff', 0.98, desc='inverter efficiency')
        self.add_param('T_air', 47., desc='Cooling Air Temp', units='degC')
        self.add_param('T_max', 80., desc='Maximum Electronics Temp', units='degC')
        self.add_param('sink_w', 192., desc='Heat Sink Base Width', units='mm')
        self.add_param('sink_l', 160., desc='Heat Sink Base Length', units='mm')
        self.add_param('fin_w', 0.7, desc='Heat sink fin width', units='K')
        self.add_param('fin_gap', 0.7, desc='Min spacing between fins', units='mm')

        self.add_param('V_in', 20., desc='incoming airflow speed', units='m/s')
        # material props
        self.add_param('h', 20., desc='fin heat transfer coefficient', units='W/(m**2*K)')
        self.add_param('case_t', 150., desc='thermal conductivity of the casing, Al-6061-T6', units='W/(m*K)' )
        self.add_param('fin_t',220., desc='thermal conductivity of the fins, Al-1100', units='W/(m*K)' )


        self.add_output('Rmax', 0.05, desc='Max Thermal Resistance', units='degC/W')
        self.add_output('R_hs', 0.05, desc='Heat Sink Resistance', units='degC/W')
        self.add_output('n_fins', 137., desc='Number of fins that can fit on the sink')
        self.add_output('min_area', 1.4, desc='Minimum cooling area needed', units='m**2')
        self.add_output('fin_h', 32., desc='Fin height', units='mm')

    def solve_nonlinear(self, p, u, r):
        pwr = p['InvPwr']*(1.-p['Inv_eff'])
        Abase = p['sink_w']*p['sink_l']
        h = p['fin_h']

        u['n_fins'] = p['sink_w']/(p['fin_w']+p['fin_gap'])
        u['min_area'] = pwr/(p['h']*(p['T_max']-p['T_air']))
        u['fin_h'] = 0.5*cu(u['min_area'],'m**2','mm**2')/(u['n_fins']*p['sink_l'])

        E_Abase = (u['n_fins']-1.)*b*p['sink_l']
        Afin = 2*u['fin_h']*p['sink_l']
        u['Rmax'] = p['T_max'] - p['T_air'] / pwr

        b = p['fin_gap']
        V = p['V']

        Pr = (mu*cp)/k
        Re = (rho*V*b*b)/(mu*p['sink_l'])
        # Teertstra, P., Yovanovich, M.M., and Culham, J.R., “Analytical Forced Convection Modeling of Plate Fin Heat Sinks,” Proceedings of 15th IEEE Semi-Therm Symposium, pp. 34-41, 1999.
        Nu = (1/(((Re*Pr)/2)**3))+(1/(0.644*Re**0.33*(1+(3.65/(sqrt(Re))))**3))**-0.33

        u['h'] = Nu*k/b # heat transfer, not to be confused with fin height

        m = sqrt(2.*h/k_fin*p['fin_t'])
        u['fin_eff'] = tanh(m*h)/(m*h)
        u['R_hs'] = 1./u['h']*(E_Abase+u['n_fins']*u['fin_eff']*Afin)
        u['R_tot'] = u['R_hs'] + (H-h)/(k_base*p['sink_w']*Abase)

    def apply_nonlinear(self, p, u, r):
        pass


if __name__ == '__main__':
    from openmdao.core.problem import Problem
    from openmdao.core.group import Group

    p = Problem()
    root = p.root = Group()
    root.add('comp', InverterSink())

    p.setup()
    p.run()

    print(p['comp.n_fins'])
    print(p['comp.min_area'])
    print(p['comp.fin_h'])
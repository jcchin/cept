from openmdao.api import Component, ScipyGMRES, Newton, IndepVarComp

class WingTemp(Component):
    def __init__(self):
        super(WingTemp, self).__init__()
        self.deriv_options['type'] = "cs"


        self.add_param('sol_heat_flux', 1145.0, desc='solar heat flux', units='W/m**2')
        self.add_param('AMO', 1366.0, desc='', units='W/m**2')
        self.add_param('SBconstantq', 5.67e-08, desc='Stefan Boltzmann Constant', units='W/m**2*K**4')
        self.add_param('absorb', 0.4, desc='Solar Absorbtivity')
        self.add_param('emiss', 0.9, desc='Emissivity')
        self.add_param('skyTemp', 239.0, desc='Sky Temperature', units='K')
        self.add_param('airTemp', 318.0, desc='Air Temperature', units='K')
        self.add_param('area', 1.0, desc='Area', units='m**2')

        self.add_state('plateTemp', 500.00, desc='Plate Temperature', units='K')
        self.add_output('sol_heat', 458.0, desc='Solar Heat', units='W')
        self.add_output('conv_heat', 26.0, desc='Convection Heat', units='W')
        self.add_output('rad_heat', 432.0, desc='Radiation Heat', units='W')

    def solve_nonlinear(self, p, u, r):

        s, c, r = self.apply_nonlinear(p,u,r)
        u['sol_heat'] = s
        u['conv_heat'] = c
        u['rad_heat'] = r


    def apply_nonlinear(self, p, u, r):
        sol_heat = p['sol_heat_flux']*p['area']*p['absorb']
        r['sol_heat'] = u['sol_heat'] - sol_heat

        conv_heat = 1.32*((u['plateTemp']-p['airTemp'])**0.25)*(u['plateTemp']-p['airTemp'])
        r['conv_heat'] = u['conv_heat'] - conv_heat

        rad_heat = p['emiss']*p['SBconstantq']*p['area']*(u['plateTemp']**4 - p['skyTemp']**4)
        r['rad_heat'] = u['rad_heat'] - rad_heat

        r['plateTemp'] = sol_heat - (conv_heat + rad_heat)

        return sol_heat, conv_heat, rad_heat

if __name__ == '__main__':
    from openmdao.core.problem import Problem
    from openmdao.core.group import Group

    p = Problem()
    root = p.root = Group()
    root.add('desVar', IndepVarComp('sol_heat_flux', 1145.), promotes=['sol_heat_flux'])
    root.add('comp', WingTemp(), promotes=['sol_heat_flux'])
    #root.add('bal', TempBalance())
    #root.connect('comp.tot_heat_resid','bal.tot_heat_resid')
    #root.connect('bal.plateTemp','comp.plateTemp')
    root.ln_solver = ScipyGMRES()
    root.nl_solver = Newton()
    #p.root.set_order(['bal', 'comp'])
    p.root.nl_solver.options['iprint'] = 1
    p.root.nl_solver.options['maxiter'] = 10
    p.setup()
    #p.root.list_connections()

    # average effective blackbody temperature of the sky,
    # use between -34C to 10C (Section 4.5.2)
    p['comp.skyTemp'] = 273-34

    p.run()
    print('Results:  Sky Temp |    Plate Temp  ')
    print('            -34C   |  %3.1f degC  %3.1f degF ' % (p['comp.plateTemp'], p['comp.plateTemp']))

    p['comp.skyTemp'] = 273+10
    p.run()
    print('             10C   |  %3.1f degC  %3.1f degF ' % (p['comp.plateTemp'], p['comp.plateTemp']))

# NASA TM 2008 215633 Table 4-8
#              |  Design High Solar Radiation |
#   Time of Day|   BTU/ft^2/hr  |    W/m^2    |
#         0500 |             0  |           0 |
#         1100 |           363  |        1145 |
#         1400 |           363  |        1145 |
#         2000 |             0  |           0 |

#              |  Design Low Solar Radiation |
#   Time of Day|   BTU/ft^2/hr  |    W/m^2    |
#         0655 |             0  |           0 |
#         1100 |            70  |         221 |
#         1400 |            80  |         252 |
#         2000 |             0  |           0 |

# NASA TM 2008 215633 Table 4-10
#   Surface, air and sky radiation temperature extremes  |   Sky Radiation    |
#               |       Surface Air Temp Extremes        |  Extreme   | Equiv |
#               |      |       Max      |       Min      |  Min Equiv |  Rad  |
#      Area     | unit |  100%  |  95%  |  100%  |  95%  |    Temp    | W/m^2 |
#Hunstville, AL |      |        |       |        |       |            |       |
#               |  C   |  40.0  |  36.7 |  -23.9 | -12.6 |     -30    |  198  |
#               |  F   |  104   |  98   |  -11   |   9   |     -22    |       |
#    KSC, FL    |      |        |       |        |       |            |       |
#               |  C   |  37.2  |  35.0 |  -7.2  |   0.6 |     -15    |  252  |
#               |  F   |  99    |  95   |   19   |   33  |       5    |       |
#    VAFB, CA   |      |        |       |        |       |            |       |
#               |  C   |  37.8  |  29.4 |   -3.9 |   1.1 |     -15    |  252  |
#               |  F   |  100   |  85   |   25   |   34  |       5    |       |
#    EAFB, CA   |      |        |       |        |       |            |       |
#               |  C   |  45.0  |  41.7 |  -15.6 |  -7.8 |     -30    |  198  |
#               |  F   |  113   |  107  |    4   |  18   |     -22    |       |
#Hickam Field,HW|      |        |       |        |       |            |       |
#               |  C   |  33.9  |  32.8 |   11.1 |  15.6 |     -15    |  252  |
#               |  F   |   93   |  91   |   52   |   60  |      5     |       |
#Anderson AFB,Guam     |        |       |        |       |            |       |
#               |  C   |  34.4  |  31.1 |  18.9  |  22.2 |     -15    |  252  |
#               |  F   |   94   |  88   |   66   |   72  |       5    |       |
#Santa Susana, CA      |        |       |        |       |            |       |
#               |  C   |  46.7  |  36.1 |  -7.8  |  1.7  |     -15    |  252  |
#               |  F   |  116   |  97   |   18   |   35  |       5    |       |
#Thickiol Wasatch Div, UT       |       |        |       |            |       |
#               |  C   |  41.7  |  35.6 |  -33.9 | -16.1 |     -30    |  198  |
#               |  F   |  107   |   96  |  -29.0 |   3   |     -22    |       |
#New Orleans, LA|      |        |       |        |       |            |       |
#               |  C   |  38.9  |  35.0 |  -10.0 |  -3.3 |    -17.8   |  241  |
#               |  F   |  102   |  95   |  -14   |   26  |       0    |       |
#Stennis Space Cntr, MS|        |       |        |       |            |       |
#               |  C   |  39.4  |  35.6 |  -14.4 | -2.2  |    -17.8   |  241  |
#               |  F   |  103   |  96   |    6   |  28   |       0    |       |
#Continental Transportation     |       |        |       |            |       |
#               |  C   |  47.2  |   -   |  -34.4 |   -   |     -30    |  198  |
#               |  F   |  117   |   -   |  -30   |   -   |     -22    |       |
#Ship Transportation   |        |       |        |       |            |       |
#               |  C   |  37.8  |   -   |  -12.2 |   -   |     -15    |  252  |
#               |  F   |  100   |   -   |  -10   |   -   |       5    |       |
#   JSC, TX     |      |        |       |        |       |            |       |
#               |  C   |  42.2  |  36.7 |  -13.3 | -2.2  |    -17.8   |  241  |
#               |  F   |  108   |  98   |    8   |  28   |      0     |       |
#GSFC, Wallops, VA     |        |       |        |       |            |       |
#               |  C   |  38.3  |  33.3 |  -20   |  -5.6 |    -17.8   |  241  |
#               |  F   |  101   |  92   |  -4    |   22  |      0     |       |
#   WSMR, NM    |      |        |       |        |       |            |       |
#               |  C   |  44.4  |  38.9 |  -25.6 | -10.0 |     -30    |  198  |
#               |  F   |  112   |  102  |  -14   |   14  |     -22    |       |
# see source for details

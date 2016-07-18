from openmdao.api import Component, ScipyGMRES, NLGaussSeidel, IndepVarComp, ScipyOptimizer
from openmdao.units.units import convert_units as cu
from numpy import tanh, arange, meshgrid, empty, exp, zeros
from numpy import sqrt
import numpy as np
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
        self.add_param('fin_h2', 0.031, desc='user defined fin height', units='m' )
        self.add_param('grease_thickness', 0.0001, desc='thickness of thermal grease', units='m')
        self.add_param('k_grease', 0.79, desc='Thermal conductivity of grease', units='W/(m*K)')
        self.add_param('case_thickness', 0.001, desc='thickness of thermal grease', units='m')
        self.add_param('k_case', 150., desc='Thermal conductivity of grease', units='W/(m*K)')
        #self.add_param('h_0', 20., desc='initial average velocity guess', units='W/(m**2*K)')
        self.add_param('rho', 1.158128, desc='density of air', units='kg/m**3')
        self.add_param('mu', 0.000018, desc='dynamic viscosity',units='Pa*s')
        self.add_param('nu', 0.0000168, desc='kinematic viscosity',units='m**2/s')
        self.add_param('Cp', 1004.088235, desc='specific heat at altitude', units='J/(kg*K)')
        self.add_param('k_air', 0.026726, desc='thermal conductivity of air', units='W/(m*K)')
        self.add_param('k_fin', 200.0, desc='thermal conductivity of aluminum', units='W/(m*K)' )
        self.add_param('flag', 0.0, desc='test flag, to set values')
        self.add_output('Dh', 1.0, desc='hydraulic diameter', units='m')
        self.add_output('Lstar', 1.0, desc='characteristic length', units='m')
        #self.add_output('sigma', 1.0, desc='ratio of flow channel areas to approaching flow area')
        #self.add_output('Kc', 1.0, desc='pressure loss coefficient due to sudden flow compression')
        #self.add_output('Ke', 1.0, desc='pressure loss coefficient due to sudden flow expansion')
        #self.add_output('f', 1.0, desc='friction factor for fully developed laminar flow, based on fin aspect ratio')
        #self.add_output('f_app', 1.0, desc='apparent friction factor for hydrodynamically developing \
        #                        laminar flow, based on f')
        # material props
        #self.add_param('case_t', 150., desc='thermal conductivity of the casing, Al-6061-T6', units='W/(m*K)' )
        #self.add_param('fin_t',220., desc='thermal conductivity of the fins, Al-1100', units='W/(m*K)' )
        self.add_output('h', 20., desc='fin heat transfer coefficient', units='W/(m**2*K)')
        self.add_output('Re', 10., desc='Reynolds #')
        self.add_output('Pr', 10., desc='Prandtl #')
        self.add_output('Nu', 10., desc='Nusselt #')
        self.add_output('R_max', 0.05, desc='Max Thermal Resistance', units='degC/W')
        self.add_output('R_tot', 0.05, desc='Total Thermal Resistance', units='degC/W')
        self.add_output('fin_eff', 1.0, desc='Fin Efficiency (based on Aspect ratio)')
        self.add_output('R_hs', 0.05, desc='Heat Sink Resistance', units='degC/W')
        self.add_output('n_fins', 137., desc='Number of fins that can fit on the sink')
        self.add_output('min_area', 1.4, desc='Minimum cooling area needed', units='m**2')
        self.add_output('fin_h', .032, desc='Fin height', units='m')
        # Pressure drop calcs
        #self.add_output('Q', 10., desc='volumetric flow rate', units='m**3/s')
        #self.add_output('dP1', 1.0, desc='pressure drop based on friction', units='Pa')
        #self.add_output('dP2', 0.0, desc='pressure drop based on velocity', units='Pa')
        #self.add_output('V_out', 5., desc='exit velocity', units='m/s')
        self.add_output('lambda', 2., desc='dimensionless channel height')
        #self.add_output('V_avg', 20., desc='computed average velocity guess', units='m/s')
        self.add_output('omega', 0.0554, desc='normalized thermal Resistance')
        self.add_output('R_lossless', 0.845, desc='thermal resistance of optimal lossless heat sink', units='degC/W')
        self.add_output('R_global', 1.28, desc='global upper bound on thermal resistance of an optimal sink', units='degC/W')
        self.add_output('zeta_global', 1.512, desc='effectiveness ratio of global upper bound, dimensionless')
        self.add_output('R_thin', 1.15, desc='thin fin upper bound on thermal resistance of an optimal sink', units='degC/W')
        self.add_output('zeta_thin', 1.361, desc='effectiveness ratio of thin fin upper bound, dimensionless')
        self.add_output('alpha', 37., desc='maximum increase in effective heat transfer area, dimensionless')
        self.add_output('fin_gap', 0.001, desc='spacing between fins', units='m')
        self.add_output('fin_w', 0.0009, desc='Heat sink fin width', units='m')
        self.add_output('R_case', 0.0001, desc='inverter case thermal resistance', units='degC/W')
        self.add_output('R_grease', 0.004, desc='grease thermal resistance', units='degC/W')


    def solve_nonlinear(self, p, u, r):

        flag = p['flag']
        pwr = p['InvPwr']*(1.-p['Inv_eff'])
        Abase = p['sink_w']*p['sink_l']
        gam_e = p['gamm_eff']
        u['R_max'] = (p['T_max'] - p['T_air'])/ pwr
        #u['min_area'] = pwr/(p['h_0']*(p['T_max']-p['T_air']))

        #choose fin height Arthur Method:
        H = u['fin_h'] = 0.5*u['min_area']/(u['n_fins']*p['sink_l'])
        if flag ==1: # force to test case fin height value
            H = 0.0141
        if flag ==2: # let user specify fin height
            H = p['fin_h2']

        Pr = u['Pr'] = (p['mu']*p['Cp'])/p['k_air']
        if flag:
            Pr = u['Pr'] = 0.708 #force to test case value

        u['Nu'] = 2.656*gam_e*Pr**(1./3.) #non-ducted
        ducted = False
        if ducted:
            u['Nu'] = 8.235 # http://www.dtic.mil/dtic/tr/fulltext/u2/a344846.pdf
        u['alpha'] = sqrt(p['k_fin']/(p['k_air']*u['Nu']))

        V = p['V_in']
        b = u['fin_gap'] = 2.*gam_e*sqrt((p['nu']*p['sink_l'])/V)
        lam = u['lambda'] = H/(u['fin_gap']*u['alpha'])
        u['omega'] = H / (p['k_fin'] * Abase)


        Dh = u['Dh'] = 2.*b
        mu = p['mu']
        l = p['sink_l']
        Re = u['Re'] = (p['rho']*V*b*b)/(mu*l)
        u['Lstar'] = l/(Dh*Re*Pr)

        u['R_lossless'] = u['omega']*lam**-2.
        u['zeta_global'] = 2.*lam + 1.
        u['R_global'] = u['zeta_global'] * u['R_lossless']

        u['zeta_thin'] = sqrt(lam)*(lam+1.)/tanh(sqrt(lam))
        u['R_thin'] = u['zeta_thin'] * u['R_lossless']

        u['fin_w'] = H/u['alpha'] # actual ideal will be a bit smaller, accurate with small lambda
        u['n_fins'] = p['sink_w']/(u['fin_w']+u['fin_gap'])

        u['R_case'] = p['grease_thickness']/(p['k_grease']*Abase)

        u['R_grease'] = p['case_thickness']/(p['k_case']*Abase)

        u['R_tot'] = u['R_case'] + u['R_grease'] + u['R_thin']

if __name__ == '__main__':
    from openmdao.core.problem import Problem
    from openmdao.core.group import Group
    from openmdao.api import ScipyGMRES

    p = Problem()
    root = p.root = Group()
    root.add('hs', HeatSink())

    # params = (('sink_w', 0.001, {'units': 'm'}),
    #         ('sink_l', 0.0009, {'units': 'm'}))
    # root.add('input_vars', IndepVarComp(params))
    # root.connect('input_vars.sink_w', 'hs.sink_w')
    # root.connect('input_vars.sink_l', 'hs.sink_l')
    p.setup(check=False)
    #root.list_connections()
    # view_tree(p)

    # Test Settings
    # p['hs.flag'] = 1
    # p['hs.sink_l'] = 0.0404 # m
    # p['hs.sink_w'] = 0.0404 # m
    # p['hs.V_in'] = 4.06 # m/s
    # p['hs.gamm_eff'] = 1.8
    # p['hs.k_air'] = 0.026 # W/m-c
    # p['hs.k_fin'] = 156. # W/m-c
    # p['hs.nu'] = 0.0000168 # m2/s

    # Test Results
    # Nu : 4.261001
    # Pr : 0.708000
    # alpha : 37.524922
    # fin height: 0.126473
    # lambda : 0.255279
    # omega : 0.055377
    # R_lossless : 0.849773
    # R_global: 1.283630
    # zeta_thin : 1.360319
    # # of fins : 21.865349
    # -------------
    # fin thickness: 0.000376
    # fin gap: 0.001472
    # R_thin: 1.155962

    def printstuff():
        print('=============')
        print('V_in : %f' %p['hs.V_in'])
        print('Nu : %f' %p['hs.Nu'])
        print('Pr : %f' %p['hs.Pr'])
        print('Re : %f' %p['hs.Re'])
        print('L* : %f' %p['hs.Lstar'])
        print('alpha : %f' %p['hs.alpha'])
        print('fin height: %f' %p['hs.fin_h'])
        print('lambda : %f' %p['hs.lambda'])
        print('omega : %f' %p['hs.omega'])
        print('R_lossless : %f' %p['hs.R_lossless'])
        print('R_global: %f' % p['hs.R_global'])
        print('zeta_thin : %f' %p['hs.zeta_thin'])
        print('# of fins : %f' %p['hs.n_fins'])
        print('-------------')
        print('fin thickness: %f' % p['hs.fin_w'])
        print('fin gap: %f' % p['hs.fin_gap'])
        print('R_thin: %f' % p['hs.R_thin'])

    p.run()
    print('R_max: %f' % (p['hs.R_max']-p['hs.R_case']-p['hs.R_grease']))
    print('Arthur Height')
    printstuff()

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    # Some sample data
    x_side = arange(4., 24., .25) # velocity
    y_side = arange(0.01, 0.06, 0.000625)
    X, Y = meshgrid(x_side,y_side)
    Z = zeros(np.shape(X))
    Za = zeros(np.shape(X))
    Zb = zeros(np.shape(X))

    p['hs.flag'] = 2
    for i,vel in enumerate(x_side):
        for j,ht in enumerate(y_side):
            p['hs.V_in'] = vel
            p['hs.fin_h2'] = ht
            p.run()
            Z[j,i] = p['hs.R_tot']
            Za[j,i] = p['hs.fin_gap']
            Zb[j,i] = p['hs.fin_w']
        print(p['hs.fin_gap'])

    # Assign colors based off some user-defined condition
    COLORS = empty(X.shape, dtype=str)
    COLORS[:,:] = 'r'
    COLORS[Z<p['hs.R_max']] = 'y'
    COLORS[Z<(p['hs.R_max']-p['hs.R_case']-p['hs.R_grease'])] = 'g'

    # 3D surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=COLORS, rstride=1, cstride=1,
            linewidth=0)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Fin Height (m)')
    ax.set_zlabel('Thermal Resistance')
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Za, facecolors=COLORS, rstride=1, cstride=1,
            linewidth=0)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Fin Height (m)')
    ax.set_zlabel('fin gap')
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Zb, facecolors=COLORS, rstride=1, cstride=1,
            linewidth=0)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Fin Height (m)')
    ax.set_zlabel('Fin thickness')
    plt.show()

# plotly imports

# import plotly.plotly as py
# from plotly import tools
# from plotly.graph_objs import Surface


# #send data to Plotly
# scene = dict(
#     xaxis=dict(
#         title='Velocity',
#         gridcolor='rgb(255, 255, 255)',
#         zerolinecolor='rgb(255, 255, 255)',
#         showbackground=True,
#         backgroundcolor='rgb(230, 230,230)'
#     ),
#     yaxis=dict(
#         title='Fin Height',
#         gridcolor='rgb(255, 255, 255)',
#         zerolinecolor='rgb(255, 255, 255)',
#         showbackground=True,
#         backgroundcolor='rgb(230, 230,230)'
#     ),
#     zaxis=dict(
#         gridcolor='rgb(255, 255, 255)',
#         zerolinecolor='rgb(255, 255, 255)',
#         showbackground=True,
#         backgroundcolor='rgb(230, 230,230)'
#     )
# )

# pfig = tools.make_subplots(rows=1, cols=3,
#                           specs=[[{'is_3d': True},{'is_3d': True},{'is_3d': True}]])
# pfig.append_trace(
#     dict(x=X, y= Y, z=Z, type='surface', scene='scene1'), 1, 1)
#     #dict(x=cmapdata.Wc_data[0,:,:], y= cmapdata.PR_data[0,:,:], z=cmapdata.eff_data[0,:,:], opacity=0.9, type='surface', scene='scene1')]

# pfig.append_trace(
#     dict(x=X, y= Y, z=Za, type='surface',scene='scene2'), 1, 2)
#     #dict(x=tmapdata.PR_vals[0,:,:], y= tmapdata.Wp_data[0,:,:], z=tmapdata.eff_data[0,:,:], opacity=0.9, type='surface',scene='scene2')]

# pfig.append_trace(
#     dict(x=X, y= Y, z=Zb, type='surface',scene='scene3'), 1, 3)
#     #dict(x=tmapdata.PR_vals[0,:,:], y= tmapdata.Wp_data[0,:,:], z=tmapdata.eff_data[0,:,:], opacity=0.9, type='surface',scene='scene2')]

# pfig['layout']['scene1'].update(scene)
# pfig['layout']['scene2'].update(scene)
# pfig['layout']['scene3'].update(scene)

# py.iplot(pfig, filename="test_comp")



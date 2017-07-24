#!/usr/bin/env python3
from openmdao.api import ExplicitComponent, IndepVarComp, ScipyOptimizer, Group, Problem, Component
from openmdao.utils.units import convert_units as cu
from numpy import tanh, arange, meshgrid, empty, exp, zeros
from numpy import sqrt
import numpy as np
# from openmdao.devtools.partition_tree_n2 import view_tree

# http://www.hpl.hp.com/techreports/Compaq-DEC/WRL-86-4.pdf

class HeatSink(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', type_=int)

    def setup(self):
        nn = self.metadata['num_nodes']
        # self.deriv_options['type'] = "fd"

        self.add_input('InvPwr', shape=(nn,), val=10000, desc='total power through the inverter', units='W')
        self.add_input('Inv_eff', shape=(nn,), val=0.96, desc='inverter efficiency')
        self.add_input('T_air', shape=(nn,), val=47., desc='Cooling Air Temp', units='degC')
        self.add_input('T_max', shape=(nn,), val=100., desc='Maximum Electronics Temp', units='degC')
        self.add_input('sink_w', shape=(nn,), val=.15, desc='Heat Sink Base Width', units='m')
        self.add_input('sink_l', shape=(nn,), val=.02, desc='Heat Sink Base Length', units='m')
        self.add_input('sink_h', shape=(nn,), val=.002, desc='Heat Sink Base Height', units='m')
        self.add_input('avail_h', shape=(nn,), val=0.04, desc='maximum available height for heat sink', units='m')
        self.add_input('gamm_eff', shape=(nn,), val=1.8, desc='boundary layer thickness inputeter, dimensionless')
        self.add_input('fin_h2', shape=(nn,), val=0.035, desc='user defined fin height', units='m' )
        self.add_input('grease_thickness', shape=(nn,), val=0.0001, desc='thickness of thermal grease', units='m')
        self.add_input('k_grease', shape=(nn,), val=1.5, desc='Thermal conductivity of grease', units='W/(m*K)')
        self.add_input('case_thickness', shape=(nn,), val=0.0001, desc='thickness of thermal grease', units='m')
        self.add_input('k_case', shape=(nn,), val=200., desc='Thermal conductivity of case', units='W/(m*K)')
        self.add_input('cfm', shape=(nn,), val=0.0015, desc='flow rate in cubic feet per minute', units='m**3/s')

        #self.add_input('h_0', shape=(nn,), val=20., desc='initial average velocity guess', units='W/(m**2*K)')
        self.add_input('rho', shape=(nn,), val=1.158128, desc='density of air', units='kg/m**3')
        self.add_input('mu', shape=(nn,), val=0.000018, desc='dynamic viscosity',units='Pa*s')
        self.add_input('nu', shape=(nn,), val=0.0000168, desc='kinematic viscosity',units='m**2/s')
        self.add_input('Cp', shape=(nn,), val=1004.088235, desc='specific heat at altitude', units='J/(kg*K)')
        self.add_input('k_air', shape=(nn,), val=0.026726, desc='thermal conductivity of air', units='W/(m*K)')
        self.add_input('k_fin', shape=(nn,), val=230.0, desc='thermal conductivity of aluminum', units='W/(m*K)' ) # 230 in COMSOL
        self.add_input('flag', shape=(nn,), val=0.0, desc='test flag, to set values')
        self.add_output('Dh', shape=(nn,), val=1.0, desc='hydraulic diameter', units='m')
        self.add_output('Lstar', shape=(nn,), val=1.0, desc='characteristic length', units='m')
        self.add_output('sigma', shape=(nn,), val=1.0, desc='ratio of flow channel areas to approaching flow area')
        self.add_output('Kc', shape=(nn,), val=1.0, desc='pressure loss coefficient due to sudden flow compression')
        self.add_output('Ke', shape=(nn,), val=1.0, desc='pressure loss coefficient due to sudden flow expansion')
        self.add_output('f', shape=(nn,), val=1.0, desc='friction factor for fully developed laminar flow, based on fin aspect ratio')
        self.add_output('f_app', shape=(nn,), val=1.0, desc='apparent friction factor for hydrodynamically developing \
                                laminar flow, based on f')
        # material props
        #self.add_param('case_t', 150., desc='thermal conductivity of the casing, Al-6061-T6', units='W/(m*K)' )
        #self.add_param('fin_t',220., desc='thermal conductivity of the fins, Al-1100', units='W/(m*K)' )
        self.add_output(name='V_in', val=7., desc='initial velocity', units='m/s')
        self.add_output(name='h', val=20., desc='fin heat transfer coefficient', units='W/(m**2*K)')
        self.add_output(name='Re', val=10., desc='Reynolds #')
        self.add_output(name='Pr', val=10., desc='Prandtl #')
        self.add_output(name='Nu', val=10., desc='Nusselt #')
        self.add_output(name='R_max', val=0.05, desc='Max Thermal Resistance', units='degK/W')
        self.add_output(name='R_tot', val=0.05, desc='Total Thermal Resistance', units='degK/W')
        self.add_output(name='fin_eff', val=1.0, desc='Fin Efficiency (based on Aspect ratio)')
        self.add_output(name='R_hs', val=0.05, desc='Heat Sink Resistance', units='degK/W')
        self.add_output(name='n_fins', val=137., desc='Number of fins that can fit on the sink')
        self.add_output(name='min_area', val=1.4, desc='Minimum cooling area needed', units='m**2')
        self.add_output(name='fin_h', val=.032, desc='Fin height', units='m')
        # Pressure drop calcs
        #self.add_output(name='Q', val=10., desc='volumetric flow rate', units='m**3/s')
        self.add_output(name='dP', val=1.0, desc='pressure drop based on friction', units='Pa')
        #self.add_output(name='dP2', val=0.0, desc='pressure drop based on velocity', units='Pa')
        #self.add_output(name='V_out', val=5., desc='exit velocity', units='m/s')
        self.add_output(name='lambda', val=2., desc='dimensionless channel height')
        #self.add_output(name='V_avg', val=20., desc='computed average velocity guess', units='m/s')
        self.add_output(name='omega', val=0.0554, desc='normalized thermal Resistance')
        self.add_output(name='R_lossless', val=0.845, desc='thermal resistance of optimal lossless heat sink', units='degK/W')
        self.add_output(name='R_global', val=1.28, desc='global upper bound on thermal resistance of an optimal sink', units='degK/W')
        self.add_output(name='zeta_global', val=1.512, desc='effectiveness ratio of global upper bound, dimensionless')
        self.add_output(name='R_thin', val=1.15, desc='thin fin upper bound on thermal resistance of an optimal sink', units='degK/W')
        self.add_output(name='zeta_thin', val=1.361, desc='effectiveness ratio of thin fin upper bound, dimensionless')
        self.add_output(name='alpha', val=37., desc='maximum increase in effective heat transfer area, dimensionless')
        self.add_output(name='fin_gap', val=0.001, desc='spacing between fins', units='m')
        self.add_output(name='fin_w', val=0.0009, desc='Heat sink fin width', units='m')
        self.add_output(name='R_case', val=0.0001, desc='inverter case thermal resistance', units='degK/W')
        self.add_output(name='R_grease', val=0.004, desc='grease thermal resistance', units='degK/W')

    def compute(self, inputs, outputs):

        flag = inputs['flag']
        pwr = inputs['InvPwr']*(1.-inputs['Inv_eff'])
        Abase = inputs['sink_w']*inputs['sink_l']
        gam_e = inputs['gamm_eff']
        outputs['R_max'] = (inputs['T_max'] - inputs['T_air'])/ pwr
        #outputs['min_area'] = pwr/(inputs['h_0']*(inputs['T_max']-inputs['T_air']))

        #choose fin height Arthur Method:
        H = outputs['fin_h'] = 0.0225 #0.5*outputs['min_area']/(outputs['n_fins']*inputs['sink_l'])
        # if flag ==1: # force to test case fin height value
        #     H = 0.0141
        # if flag ==2: # let user specify fin height
        #     H = inputs['fin_h2']

        Pr = outputs['Pr'] = (inputs['mu']*inputs['Cp'])/inputs['k_air']
        if flag:
            Pr = outputs['Pr'] = 0.708 #force to test case value

        outputs['Nu'] = 2.656*gam_e*Pr**(1./3.) #non-ducted

        ducted = True
        if ducted:
            outputs['Nu'] = 8.235 # http://www.dtic.mil/dtic/tr/fulltext/u2/a344846.pdf
        outputs['alpha'] = sqrt(inputs['k_fin']/(inputs['k_air']*outputs['Nu']))

        V = outputs['V_in']  = inputs['cfm']/(inputs['sink_w']*H-(H*outputs['n_fins']*outputs['fin_w']))
        print('V',V)

        b = outputs['fin_gap'] = 2.*gam_e*sqrt((inputs['nu']*inputs['sink_l'])/V)
        lam = outputs['lambda'] = H/(outputs['fin_gap']*outputs['alpha'])
        outputs['omega'] = H / (inputs['k_fin'] * Abase)


        Dh = outputs['Dh'] = 2.*b
        mu = inputs['mu']
        l = inputs['sink_l']
        Re = outputs['Re'] = (inputs['rho']*V*b*b)/(mu*l)
        outputs['Lstar'] = l/(Dh*Re*Pr)

        outputs['R_lossless'] = outputs['omega']*lam**-2.
        outputs['zeta_global'] = 2.*lam + 1.
        outputs['R_global'] = outputs['zeta_global'] * outputs['R_lossless']

        outputs['zeta_thin'] = sqrt(lam)*(lam+1.)/tanh(sqrt(lam))
        outputs['R_thin'] = outputs['zeta_thin'] * outputs['R_lossless']

        outputs['fin_w'] = H/outputs['alpha'] # actual ideal will be a bit smaller, accurate with small lambda
        outputs['n_fins'] = inputs['sink_w']/(outputs['fin_w']+outputs['fin_gap'])

        outputs['R_case'] = inputs['case_thickness']/(inputs['k_case']*Abase)

        outputs['R_grease'] = inputs['grease_thickness']/(inputs['k_grease']*Abase)

        if lam >=1:
            outputs['R_tot'] = outputs['R_case'] + outputs['R_grease'] + outputs['R_global']

        else:
            outputs['R_tot'] = outputs['R_case'] + outputs['R_grease'] + outputs['R_thin']

        #pressure drop calcs
        s = outputs['sigma'] = 1.- (outputs['n_fins']*outputs['fin_w'])/inputs['sink_w']
        outputs['Ke'] = (1.-s**2.)**2.
        outputs['Kc'] = 0.42 * (1-s**2)
        Lstar2 = (l/Dh)/Re
        lamb = outputs['lambda'] = b/H
        outputs['f'] = (24.-32.527*lamb + 46.721*lamb**2. - 40.829*lamb**3. + 22.954*lamb**4 - 6.089*lamb**5 ) / Re
        outputs['f_app'] = sqrt(((3.44/sqrt(Lstar2))**2. + (outputs['f']*Re)**2.))/Re
        rho = inputs['rho']
        outputs['dP'] = ((outputs['Kc']+4.*outputs['f_app']*(l/Dh)+outputs['Ke'])*rho*(V**2.)/2.)

        outputs['h'] = outputs['Nu']*inputs['k_air']/Dh
        #print(u['h'],(1./(Abase*u['R_thin'])))


if __name__ == '__main__':
    # from openmdao.core.problem import Problem
    # from openmdao.core.group import Group
    # from openmdao.api import ScipyGMRES

    p = Problem()
    p.model.add_subsystem('hs', HeatSink(num_nodes=1))

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
        print('R_tot: %f' % p['hs.R_tot'])
        print('Pressure Drop %f' % p['hs.dP'])
        print('h %f' % p['hs.h'])
        print('R_max %f' % p['hs.R_max'])

    p.run()
    print('R_max: %f' % (p['hs.R_max']-p['hs.R_case']-p['hs.R_grease']))
    print('Arthur Height')
    printstuff()
    #quit()

    # from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    # # Some sample data
    # #x_side = arange(4., 24., .25) # velocity
    x_side = arange(0.0015, 0.008, 0.00025) # velocity
    y_side = arange(0.015, 0.06, 0.000625)
    X, Y = meshgrid(x_side,y_side)
    Z = zeros(np.shape(X))
    Za = zeros(np.shape(X))
    Zb = zeros(np.shape(X))

    # p['hs.flag'] = 2
    # for i,cfm in enumerate(x_side):
    #     for j,ht in enumerate(y_side):
    #         p['hs.cfm'] = cfm
    #         p['hs.fin_h2'] = ht
    #         p.run()
    #         p.run()
    #         p.run()
    #         Z[j,i] = p['hs.R_tot']
    #         Za[j,i] = p['hs.fin_gap']
    #         Zb[j,i] = p['hs.fin_w']

    #         if (cfm == 0.004 and ht == 0.035000000000000024):
    #             print('CFM, H-->', cfm, ht, ' R_tot, Fin_gap, Fin_W, N_fins')
    #             print(cfm,ht,p['hs.R_tot'],p['hs.fin_gap'],p['hs.fin_w'],p['hs.n_fins'])
    #             print
    # #quit()
    # # Assign colors based off some user-defined condition
    # COLORS = empty(X.shape, dtype=str)
    # COLORS[:,:] = 'r'
    # COLORS[Z<p['hs.R_max']] = 'y'
    # COLORS[Z<(p['hs.R_max']-p['hs.R_case']-p['hs.R_grease'])] = 'g'

    # # 3D surface plot
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, facecolors=COLORS, rstride=1, cstride=1,
    #         linewidth=0)
    # ax.set_xlabel('CFM (m^3/s)')
    # ax.set_ylabel('Fin Height (m)')
    # ax.set_zlabel('Thermal Resistance')
    # plt.show()

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Za, facecolors=COLORS, rstride=1, cstride=1,
    #         linewidth=0)
    # ax.set_xlabel('CFM (m^3/s)')
    # ax.set_ylabel('Fin Height (m)')
    # ax.set_zlabel('fin gap')
    # plt.show()

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Zb, facecolors=COLORS, rstride=1, cstride=1,
    #         linewidth=0)
    # ax.set_xlabel('CFM (m^3/s)')
    # ax.set_ylabel('Fin Height (m)')
    # ax.set_zlabel('Fin thickness')
    # plt.show()

# plotly imports

import plotly.plotly as py
from plotly import tools
from plotly.graph_objs import Surface


#send data to Plotly
scene = dict(
    xaxis=dict(
        title='Velocity',
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)'
    ),
    yaxis=dict(
        title='Fin Height',
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)'
    ),
    zaxis=dict(
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)'
    )
)

pfig = tools.make_subplots(rows=1, cols=3,
                          specs=[[{'is_3d': True},{'is_3d': True},{'is_3d': True}]])
pfig.append_trace(
    dict(x=X, y= Y, z=Z, type='surface', scene='scene1'), 1, 1)
    # dict(x=cmapdata.Wc_data[0,:,:], y= cmapdata.PR_data[0,:,:], z=cmapdata.eff_data[0,:,:], opacity=0.9, type='surface', scene='scene1'))

pfig.append_trace(
    dict(x=X, y= Y, z=Za, type='surface',scene='scene2'), 1, 2)
    # dict(x=tmapdata.PR_vals[0,:,:], y= tmapdata.Wp_data[0,:,:], z=tmapdata.eff_data[0,:,:], opacity=0.9, type='surface',scene='scene2'))

pfig.append_trace(
    dict(x=X, y= Y, z=Zb, type='surface',scene='scene3'), 1, 3)
    # dict(x=tmapdata.PR_vals[0,:,:], y= tmapdata.Wp_data[0,:,:], z=tmapdata.eff_data[0,:,:], opacity=0.9, type='surface',scene='scene2'))

pfig['layout']['scene1'].update(scene)
pfig['layout']['scene2'].update(scene)
pfig['layout']['scene3'].update(scene)

# py.iplot(pfig, filename="test_comp")



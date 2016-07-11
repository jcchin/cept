

    self.add_param('R3nacOD', 35., units='me-3',  desc='Nacelle OD')
    self.add_param('R3magOD', 34., units='me-3',  desc='Magnets OD')
    self.add_param('R3rotorID',   334.4 , units='me-3',  desc='Rotor ID')
    self.add_param('R3statorOD',  330.4 , units='me-3',  desc='Stator OD')
    self.add_param('R3radgap',    2. , units='me-3',  desc='Radial gap between stator and rotor')
    self.add_param('R3statorL',   13., units='me-3',  desc='Stator axial length')
    self.add_param('R3coilgaphgt',    13.5  , units='me-3',  desc='Radial height of coil gap')
    self.add_param('R3numcoils',  48., desc='Number of coils & slots between coils')
    self.add_param('R3maggap',    1. , units='me-3',  desc='Tangential gap between coils')
    self.add_param('R3coillength',    146.8 , units='me-3',  desc='Coil axial length')
    self.add_param('R3numfins',  305., desc= 'Number of fins based on angular spacing')
    self.add_param('R3fingap',    1.83  , units='me-3',  desc='Tangential gap between fins, average of tip and base 1.8 + 1.86')
    self.add_param('R3tipgapOD',  258.2 , units='me-3',  desc='Tip gap OD')
    self.add_param('R3tipgapID',  256.2 , units='me-3',  desc='Tip gap ID')
    self.add_param('R3fintipgap', 1. , units='me-3',  desc='Tip radial gap')
    self.add_param('R3finhgt',    16.2  , units='me-3',  desc='Fin radial height')
    self.add_param('R3fingapOD',  290.6 , units='me-3',  desc='Fin gap OD')
    self.add_param('R3fingapID',  258.2 , units='me-3',  desc='Fin gap ID (same as Tip gap ID)')
    self.add_param('Check',   16.2  , units='me-3',  desc='Fin gap height (should be same as Fin Radial Height)')
    self.add_param('R3finthk',   1., units='me-3',  desc='Fin thickness (average of tip and base)')
    # Flow area
    self.add_param('R3rotstatgapA',   0.0021  ,units='m**2', desc=' Axial flow area between rotor and stator')
    self.add_param('R3fingapA',   0.0090  ,units='m**2', desc=' Axial flow area of 305 fins')
    self.add_param('R3coilgapA',  0.0006  ,units='m**2', desc=' Axial flow area between 48 coils')
    self.add_param('R3fintipgapA',    0.0008  ,units='m**2', desc=' Axial flow area at fin tip gap')
    self.add_param('R3totalFA',   0.0126  ,units='m**2', desc=' Total axial flow area of moto & fins')
    self.add_param('R3wvel',  235.6   units='rad/s',   desc='rotational velocity')
    # Properties
    self.add_param('R3dvis',  1.94e-05    ,units='Pa*s',    desc='dynamic viscosity')
    self.add_param('R3rhoent',    1.033   ,units='kg/(m**3)',  desc='Density')
    self.add_param('R3Kair',  0.027   ,units='W/(m*K)' ,  desc=' conductivity')
    self.add_param('R3cp',    1007.    ,units='J/(kg*K)' ,  desc='specific heat')

    #Thermal resistance of fins
    self.add_param('Size', 5, desc='fin per length')
    self.add_param('NumFins', 305./5., desc='number of fins')
    self.add_param('Base width',  182.6  , units='me-3', desc='')
    self.add_param('Fin thickness',   1.0, units='me-3', desc='')
    self.add_param('Length ', 130, units='me-3', desc='')
    self.add_param('Base thickness',  2  , units='me-3', desc='')
    self.add_param('Fin height',  16.2   , units='me-3', desc='')
    # Material - Aluminum 7075T6
    self.add_param('Velocity',    24.7  , units='m/s', desc='')

    self.add_param('Therm Res',   0.11, units='degC/W', desc='thermal resistance per 1/5 sector')

    self.add_output('Tot res', 0.022 units='degC/W', desc='')


    self.add_param('R3inletID',   292., units='me-3',  desc='Inlet ID')
    self.add_output('R3hubOD', 262, units='me-3',  desc='Hub OD')
    self.add_output('R3areainlet', 0.0132, units='m**2',  desc='Inlet area')
    self.add_output('R3inletV',    22.4, units='m/s',  desc='Velocity at inlet contraction')
    self.add_output('R3inletdP',   259., units='Pa',  desc='Pressure drop at inlet add this to States compressor inlet')
from dolfin import *
import pylab
from pylab import deg2rad,plot,show,linspace,ones,zeros,copy,array,arange,randn
from ice_model_functions import *
##########################################################
#################  SET PETSC OPTIONS  ####################
##########################################################

PETScOptions.set("ksp_type","preonly")
PETScOptions.set("pc_type","lu")
PETScOptions.set("pc_factor_mat_solver_package","mumps")
PETScOptions.set("mat_mumps_icntl_14","1000")
#PETScOptions.set("snes_ls_alpha","1e-5")
#PETScOptions.set("ksp_final_residual","0")

##########################################################
#################  SET FENICS OPTIONS  ###################
##########################################################

parameters['form_compiler']['quadrature_degree'] = 2
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['representation'] = 'quadrature'
parameters['form_compiler']['precision'] = 30

ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

set_log_active(True)
set_log_level(PROGRESS)


##########################################################
####################  CONSTANTS  #########################
##########################################################

# TIME
minute = 60.0
hour = 60*minute
day = 24*hour
year = 365*day

# CONSTANTS
g = 9.81

# RHEOLOGICAL CONSTANTS
rho = 910.
n = 3.0

Bc = 3.61e-13*year
Bw = 1.73e3*year
Qc = 6e4
Qw = 13.9e4
Rc = 8.314
gamma = 8.7e-4

eps_reg = Constant(1e-10)

# THERMAL CONTANTS
k = 2.1*year
Cp = 2009.
kappa = k/(rho*Cp)
q_geo = 0.042*year

# MASS
thklim = 10.0

# SLIDING
beta_min = 1e3
beta_max = 1e6
r = 3.0

###################################################
########### GEOMETRY AND INPUT DATA  ##############
###################################################

##### MESH (HORIZONTAL) #####
L = 800000.
cell_size = 25000.
#mesh = CircleMesh(Point(0.,0.),L,cell_size)
mesh = Mesh('circle_mesh_25000.xml')
#mesh = RectangleMesh(-L,-1,L,1,128,1)

##### GRID (VERTICAL) #####
N_layers = 8
deltax = 1./(N_layers-1.)
sigmas = linspace(0,1,N_layers,endpoint=True)

###### TIMES ######
dt = 10.0
t_start = 0
t_end = 120000
times = arange(t_start,t_end,dt)

##### GEOMETRY #####
class Surface(Expression):
  def eval(self,values,x):
    values[0] = thklim + randn()

class Bed(Expression):
  def eval(self,values,x):
    values[0] = 0

##### BOUNDARY DATA #####
class Beta2(Expression):
  def eval(self,values,x):
    values[0] = 1e10

class Adot(Expression):
  Rel = 450000
  s = 1e-5
  def eval(self,values,x):
    #values[0] = 0.3
    values[0] = min(0.5,self.s*(self.Rel-sqrt(x[0]**2 + x[1]**2)))

class SurfaceTemperature(Expression):
  Tmin = 238.15
  St = 1.67e-5
  def eval(self,values,x):
    values[0] = self.Tmin + self.St*sqrt(x[0]**2 + x[1]**2)

##################################################
#############  MODEL SPECIFICATION  ##############
##################################################

# FUNCTION SPACES
Q = FunctionSpace(mesh,"CG",1) # SCALAR
Q2 = MixedFunctionSpace([Q,]*2)
V = MixedFunctionSpace([Q]*4) # VELOCITY
Z = MixedFunctionSpace([Q]*N_layers) # TEMPERATURE

# GEOMETRY AND DATA INTERPOLANTS
B = interpolate(Bed(),Q)
beta2 = interpolate(Beta2(),Q)
adot = interpolate(Adot(),Q)
T_s = interpolate(SurfaceTemperature(),Q)

# FUNCTIONS 
U = Function(V)
Phi = TestFunction(V)
dU = TrialFunction(V)

H = Function(Q)
xsi = TestFunction(Q)
dH = TrialFunction(Q)

T_ = Function(Z)
Psi = TestFunction(Z)
dT = TrialFunction(Z)

ubar,vbar,udef,vdef = split(U)
phibar,psibar,phidef,psidef = split(Phi)

H0 = Function(Q)
T0_ = Function(Z)

# THETA METHOD
theta = Constant(1.5)
Hmid = theta*H + (1-theta)*H0
S = B + Hmid

# METRICS FOR COORDINATE TRANSFORM
def dsdx(s):
    return 1./Hmid*(S.dx(0) - s*Hmid.dx(0))

def dsdy(s):
    return 1./Hmid*(S.dx(1) - s*Hmid.dx(1))

def dsdz(s):
    return -1./Hmid

# VERTICAL ANSATZ #
# TEMPERATURE DEPENDENT SHAPE FUNCTION

# TEST FUNCTION COEFFICIENTS

p0 = 12.0


# FULL ADAPTIVE p -- slow!!
p = Function(Q)
p.vector()[:] = p0
phat = Function(Q)
udhat = Function(Q)
p_test = Function(Q)
coef = [lambda s:1.0, lambda s:Expression('1.0/p*((p+1)*pow(s,p) - 1)',s=s,p=p,degree=1)]
dcoef = [lambda s:0, lambda s:Expression('(p+1)*pow(s,p-1)',s=s,p=p,degree=1)]

# REASONABLE BUT CONSTANT p -- fast!!
#p = p0
#coef = [lambda s:1.0, lambda s:1./p*((p+1)*s**p - 1)]
#dcoef = [lambda s:0, lambda s:(p+1)*s**(p-1)]

u_ = [ubar,udef]
v_ = [vbar,vdef]
phi_ = [phibar,phidef]
psi_ = [psibar,psidef]

u = VerticalBasis(u_,coef,dcoef)
v = VerticalBasis(v_,coef,dcoef)
phi = VerticalBasis(phi_,coef,dcoef)
psi = VerticalBasis(psi_,coef,dcoef)

T = VerticalFDBasis(T_,deltax,sigmas)
T0 = VerticalFDBasis(T0_,deltax,sigmas)

# TERMWISE STRESSES AND NONLINEARITIES
def A_v(T):
    return conditional(le(T,263.15),Bc*exp(-Qc/(Rc*T)),Bw*exp(-Qw/(Rc*T)))

# 2nd INVARIANT STRAIN RATE

def epsilon_dot(s):
    return ((u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
                +(v.dx(s,1) + v.ds(s)*dsdy(s))**2 \
                +(u.dx(s,0) + u.ds(s)*dsdx(s))*(v.dx(s,1) + v.ds(s)*dsdy(s)) \
                +0.25*((u.ds(s)*dsdz(s))**2 + (v.ds(s)*dsdz(s))**2 \
                + ((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))**2) \
                + eps_reg)

#def epsilon_dot(s):
#    return (0.25*((u.ds(s)*dsdz(s))**2 + (v.ds(s)*dsdz(s))**2) + eps_reg)

# VISCOSITY
def eta_v(s):
    return (A_v(T0.eval(s) + gamma*Hmid*s))**(-1./n)/2.*epsilon_dot(s)**((1.-n)/(2*n))

# MEMBRANE STRESSES
def membrane_xx(s):
    return (phi.dx(s,0) + phi.ds(s)*dsdx(s))*Hmid*eta_v(s)*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 2*(v.dx(s,1) + v.ds(s)*dsdy(s)))

def membrane_xy(s):
    return (phi.dx(s,1) + phi.ds(s)*dsdy(s))*Hmid*eta_v(s)*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

def membrane_yx(s):
    return (psi.dx(s,0) + psi.ds(s)*dsdx(s))*Hmid*eta_v(s)*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

def membrane_yy(s):
    return (psi.dx(s,1) + psi.ds(s)*dsdy(s))*Hmid*eta_v(s)*(2*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 4*(v.dx(s,1) + v.ds(s)*dsdy(s)))

# SHEAR STRESSES
def shear_xz(s):
    return dsdz(s)**2*phi.ds(s)*Hmid*eta_v(s)*u.ds(s)

def shear_yz(s):
    return dsdz(s)**2*psi.ds(s)*Hmid*eta_v(s)*v.ds(s)

# DRIVING STRESSES
def tau_dx():
    return rho*g*Hmid*S.dx(0)*Phi[0]

def tau_dy():
    return rho*g*Hmid*S.dx(1)*Phi[1]

# VERTICAL VELOCITY
def w(s):
    w_0 = (U[0].dx(0) + U[1].dx(1))*(s-1.)
    w_2 = (U[2].dx(0) + U[3].dx(1))*(s**(p0+1) - s)/p0 + (p0+1)/H*U[2]*(1./p0*(s**p0 - 1.)*S.dx(0) - 1./p0*(s**(p0+1) - 1.)*H.dx(0)) + (p0+1)/H*U[3]*(1./p0*(s**p0 - 1.)*S.dx(1) - 1./p0*(s**(p0+1) - 1.)*H.dx(1))
    return (u(1)*B.dx(0) + v(1)*B.dx(1)) - 1./dsdz(s)*(w_0 + w_2) 


# GET QUADRATURE POINTS (THIS SHOULD BE ODD: WILL GENERATE THE GAUSS-LEGENDRE RULE 
# POINTS AND WEIGHTS OF O(n), BUT ONLY THE POINTS IN [0,1] ARE KEPT< DUE TO SYMMETRY.
points,weights = half_quad(11)

# INSTANTIATE VERTICAL INTEGRATOR
vi = VerticalIntegrator(points,weights)

# FIRST ORDER EQUATIONS
R_x = - vi.intz(membrane_xx) - vi.intz(membrane_xy) - vi.intz(shear_xz) - phi(1)*beta2*u(1) - tau_dx()
R_y = - vi.intz(membrane_yx) - vi.intz(membrane_yy) - vi.intz(shear_yz) - psi(1)*beta2*v(1) - tau_dy()
#R_x = - vi.intz(shear_xz) - phi(1)*beta2*u(1) - tau_dx()
#R_y = - vi.intz(shear_yz) - psi(1)*beta2*v(1) - tau_dy()

R = (R_x + R_y)*dx
J = derivative(R,U,dU)

### MASS BALANCE ##

## SIA DIFFUSION COEFFICIENT INTEGRAL TERM.
def sia_int_bar(s):
    return A_v(T.eval(s) + gamma*H*s)*s**(n+1)

def sia_int_s(s):
    return A_v(T.eval(s) + gamma*H*s)*s**n

D = 2.*(rho*g)**n*H**(n+2)*dot(grad(S),grad(S))**((n-1.)/2.)*vi.intz(sia_int_bar) + rho*g*H**2/beta2

#D = 2*A/(n+2)*(rho*g)**n*dot(grad(S),grad(S))**((n-1)/2.)*H**(n+2) + (rho*g*H**2)/beta2

ubar_si = -D*S.dx(0)/H
vbar_si = -D*S.dx(1)/H

M = dH*xsi*dx
ubar_proj = (U[0] - ubar_si)*xsi*dx
vbar_proj = (U[1] - vbar_si)*xsi*dx

ubar_c = Function(Q)
vbar_c = Function(Q)
h = CellSize(mesh)

R_H = ((H-H0)/dt*xsi + D*dot(grad(xsi),grad(S)) - (xsi.dx(0)*Hmid*ubar_c + xsi.dx(1)*Hmid*vbar_c) + h/2.*sqrt(ubar_c**2 + vbar_c**2)*(xsi.dx(0)*Hmid.dx(0) + xsi.dx(1)*Hmid.dx(1)) - adot*xsi)*dx 
#R_H = ((H-H0)/dt*xsi + D*dot(grad(xsi),grad(S)) - (xsi.dx(0)*Hmid*ubar_c + xsi.dx(1)*Hmid*vbar_c) - adot*xsi)*dx 
#R_H = ((H-H0)/dt*xsi + D*dot(grad(xsi),grad(S)) - adot*xsi)*dx 
J_H = derivative(R_H,H)

### ENERGY BALANCE ### 
R_T = 0

Tm = as_vector([273.15 - gamma*sigma*H for sigma in sigmas])
Tmb = 273.15 - gamma*H
T_tol = 1.0

for i in range(N_layers):
    # SIGMA COORDINATE
    s = i/(N_layers-1.0)

    # EFFECTIVE VERTICAL VELOCITY
    w_eff = u(s)*dsdx(s) + v(s)*dsdy(s) + w(s)*dsdz(s) + 1./H*(1.-s)*((H-H0)/dt)
 
    # STRAIN HEAT
    #Phi_strain = (2*n)/(n+1)*2*eta_v(s)*epsilon_dot(s)
    Phi_strain = 2.*eta_v(s)*epsilon_dot(s)
 
    # STABILIZATION SCHEME
    Umag = sqrt(u(s)**2 + v(s)**2 + 1e-3)
    tau = h/(2*Umag)
    Psihat = Psi[i] + tau*(u(s)*Psi[i].dx(0) + v(s)*Psi[i].dx(1))

    # TIME DERIVATIVE
    dTdt = (T(i) - T0(i))/dt

    # SURFACE BOUNDARY
    if i==0:
        R_T += Psi[i]*(T(i) - T_s)*dx
    # BASAL BOUNDARY
    elif i==(N_layers-1):
        R_T += dTdt*Psi[i]*dx
        R_T += (u(s)*T.dx(i,0) + v(s)*T.dx(i,1))*Psihat*dx
        R_T += -Phi_strain/(rho*Cp)*Psi[i]*dx 
        f = (q_geo + beta2*(u(s)**2 + v(s)**2))/(rho*Cp*kappa*dsdz(s))
        #f = (q_geo)/(rho*Cp*kappa*dsdz(s))
        R_T += -w_eff*f*Psi[i]*dx
        R_T += -2.*kappa*dsdz(s)**2*((T(N_layers-2) - T(N_layers-1))/(deltax**2) - f/deltax)*Psi[i]*dx
    # INTERIOR
    else:
        R_T += dTdt*Psi[i]*dx
        R_T += -kappa*dsdz(s)**2.*T.d2s(i)*Psi[i]*dx
        R_T += w_eff*T.ds(i)*Psi[i]*dx
        R_T += (u(s)*T.dx(i,0) + v(s)*T.dx(i,1))*Psihat*dx
        R_T += -Phi_strain/(rho*Cp)*Psi[i]*dx 

# PRETEND THIS IS LINEAR (A GOOD APPROXIMATION IN THE TRANSIENT CASE)
R_T = replace(R_T,{T_:dT})

#####################################################################
#########################  I/O Functions  ###########################
#####################################################################

# For moving data between vector functions and scalar functions 
assigner_inv = FunctionAssigner([Q,Q,Q,Q],V)
assigner     = FunctionAssigner(V,[Q,Q,Q,Q])
assigner_vec = FunctionAssigner(Q2,[Q,Q])

#####################################################################
######################  Variational Solvers  ########################
#####################################################################
#Define variational solver for the momentum problem

momentum_problem = NonlinearVariationalProblem(R,U,J=J,form_compiler_parameters=ffc_options)
momentum_solver = NonlinearVariationalSolver(momentum_problem)
momentum_solver.parameters['nonlinear_solver'] = 'newton'

momentum_solver.parameters['newton_solver']['relaxation_parameter'] = 0.7
momentum_solver.parameters['newton_solver']['relative_tolerance'] = 1e-3
momentum_solver.parameters['newton_solver']['absolute_tolerance'] = 1e3
momentum_solver.parameters['newton_solver']['maximum_iterations'] = 20
momentum_solver.parameters['newton_solver']['error_on_nonconvergence'] = False
momentum_solver.parameters['newton_solver']['linear_solver'] = 'mumps'
#momentum_solver.parameters['newton_solver']['linear_solver'] = 'gmres'
#momentum_solver.parameters['newton_solver']['preconditioner'] = 'bjacobi'

mass_problem = NonlinearVariationalProblem(R_H,H,J=J_H,form_compiler_parameters=ffc_options)
mass_solver = NonlinearVariationalSolver(mass_problem)
mass_solver.parameters['nonlinear_solver'] = 'snes'

mass_solver.parameters['snes_solver']['method'] = 'vinewtonrsls'
mass_solver.parameters['snes_solver']['relative_tolerance'] = 1e-3
mass_solver.parameters['snes_solver']['absolute_tolerance'] = 1e-3
mass_solver.parameters['snes_solver']['maximum_iterations'] = 20
mass_solver.parameters['snes_solver']['error_on_nonconvergence'] = False
mass_solver.parameters['snes_solver']['linear_solver'] = 'mumps'

l_thick_bound = project(Constant(thklim),Q)
u_thick_bound = project(Constant(1e4),Q)

energy_problem = LinearVariationalProblem(lhs(R_T),rhs(R_T),T_,form_compiler_parameters=ffc_options)
energy_solver = LinearVariationalSolver(energy_problem)
energy_solver.parameters['linear_solver'] = 'mumps'
#energy_solver.parameters['preconditioner'] = 'bjacobi'

#####################################################################
##################  INITIAL CONDITIONS AND BOUNDS  ##################
#####################################################################

H0.vector()[:] = thklim + randn(H0.vector().size())
H.vector()[:] = thklim + randn(H0.vector().size())
T0_temp = project(as_vector([T_s]*N_layers),Z)
T0_.interpolate(T0_temp)
T_.interpolate(T0_)

#################################################
################  LOGGING  ######################
#################################################
results_dir = '/data/brinkerhoff/EISMINTS_II/H/warm_25km_b/'

Ufile = File(results_dir + 'U.pvd')
Hfile = File(results_dir + 'H.pvd')
Tbfile = File(results_dir + 'Tb.pvd')

Txfile = File(results_dir + 'T.xml')
Uxfile = File(results_dir + 'U.xml')
Hxfile = File(results_dir + 'H.xml')

beta2file = File(results_dir + 'beta2.pvd')

Us = Function(Q2)
Tb_ = Function(Q)

################
### Solution ###
################

for t in times:
    if t==7000:
        File('U_restart.xml') << U
        File('T_restart.xml') << T_
        File('H_restart.xml') << H
        momentum_solver.parameters['newton_solver']['relaxation_parameter'] = 0.7

    # COMMENT THIS TO DITCH ADAPTIVE p -- which you should if you want this to run fast.
    udbar  = project(1e15*vi.intz(sia_int_bar),Q)
    uds    = project(1e15*vi.intz(sia_int_s),Q)
    udhat = project(udbar/uds)
    p_temp = project(-udhat/(udhat-1))
    p_temp.vector()[p_temp.vector()<4] = 4.
    p_temp.vector()[p_temp.vector()>25] = 25.
    p_temp = p_temp.round()
    p.vector()[:] = p_temp.vector()[:]

    momentum_solver.solve() 
    # Find corrective velocities
    solve(M==ubar_proj,ubar_c,solver_parameters={'linear_solver':'mumps'},form_compiler_parameters=ffc_options)
    solve(M==vbar_proj,vbar_c,solver_parameters={'linear_solver':'mumps'},form_compiler_parameters=ffc_options)
    mass_solver.solve(l_thick_bound,u_thick_bound)

    energy_solver.solve()
    
    # Update temperature field
    Tb_m = project(Tmb,Q)
    Tb_temp = project(T_[N_layers-1],Q)
    
    beta2.vector()[Tb_temp.vector()>(Tb_m.vector() - T_tol)] = 1e3
    beta2.vector()[Tb_temp.vector()<=(Tb_m.vector() - T_tol)] = 1e6 

    # UPDATE PREVIOUS TIME STEP
    T_melt = project(Tm)
    T_.vector()[T_.vector()>T_melt.vector()] = T_melt.vector()[T_.vector()>T_melt.vector()]
    
    H0.interpolate(H)
    T0_.interpolate(T_)

    print t, H0.vector().max()

    # OUTPUT DATA
    Us_temp = project(as_vector([u(0.),v(0.0)]))
    Us.interpolate(Us_temp)

    Tb_temp = project(T_[N_layers-1],Q)
    Tb_.interpolate(Tb_temp)
    
    Ufile << (Us,t)
    Hfile << (H0,t)

    Tbfile << (Tb_,t)
    
    Hxfile << H0
    Txfile << T_
    Uxfile << U
    beta2file << beta2

    File(results_dir + 'Us_{:06d}.xml'.format(int(t)))<<Us
    File(results_dir + 'Tb_{:06d}.xml'.format(int(t)))<<Tb_
    File(results_dir + 'H_{:06d}.xml'.format(int(t)))<<H




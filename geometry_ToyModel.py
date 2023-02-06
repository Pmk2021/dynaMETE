'''
This file defines the functions needed for using the information geometry approach I looked at.
In practice it's very similar to the lambda dynamics version, but I want to see numerically how it shapes up.
It will get the needed matrix and vector for inversion, and define the necessary iteration scheme.
This version uses the flexible definitions for cov and mean, not the hard coded ones. This will
make it much slower, but it also means that we can change f much easier.
This file uses the toy model with only n and no e dependence.
'''

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import root #<- better than fsolve apparently.
import DynaMETE_Rfunctions_ToyModel as rf

def get_dl_matrix_vector(l,s,p,ds):
    '''
    Calculates the matrix and vector of covariances needed for the dynamics
    l are lambdas as an array, so l[0] is lambda_1 etc.
    s are state variables, call S, N
    p are parameters, call b0, d0, Nc
    ds are derivatives of state variables, call dN
    returns array and vector that need to be solved.
    '''
    # Calculate normalization
    z = rf.mean_pow(0,l,s,p)
    # Calculate rows of covariances
    # First define the covariances since many of them repeat and this will make the rows easier to read
    covnn = rf.cov_pow([1,1],l,s,p,z)
    covnf = rf.cov(lambda n,s,p: n,rf.f,l,s,p,z)
    covff = rf.cov(rf.f,rf.f,l,s,p,z)
    # Then put them in rows
    r1 = np.array([covnf,covff])
    r2 = np.array([covnn,covnf])

    # Also calculate and return vector for Ax = B
    # Covariances. 
    covndf = rf.cov(lambda n,s,p,ds: n,rf.dfdt,l,s,p,z,ds)
    # THIS HAS A TYPO
    covfdf = rf.cov(rf.dfdt,rf.dfdt,l,s,p,z,ds) 
    
    # Vectors, including normalization in proper spot.
    #v1 = - covndf*l[0] - covfdf*l[1] - rf.mean(rf.dfdt,l,s,p,ds,z=z)
    # approximating d^2N/dt^2 as <df/dt> <- Not sure what this is numerically.
    #v2 = -ds['dN']/s['S']-covndf*l[1]
    # Try vectors with just dN/dt and d^2N/dt^2
    v1 = 0#- rf.mean(rf.dfdt,l,s,p,ds,z=z) # Bad approximation of d^2N/dt^2
    v2 = - ds['dN']/s['S']
    
    return np.array([r1,r2]),np.array([v1,v2])

# Give defaults to l and ds so that by default we assume the lambdas and the derivatives have to be calculated from
# the corresponding state variable. Allow them as optional arguments if we want to iterate from a specific state.
def iterate(t,s,p,dt=0.2,l=np.array([]),ds=np.array([])):
    '''
    Iterates the lambda dynamics theory. This works by updating lambdas from the derivatives obtained from the
    covariance matrix calculated in get_dl_matrix_vector. The scheme is:
    1. Get derivatives of lambdas
    2. Update state variables s_t+1 = s_t + ds_t
    3. Update derivatives with new state variables, ie. dN/dt = S <f(X_t+1,lambda_t)>
    4. Update lambdas as l_t+1 = l_t + dt dl/dt
    
    THIS DEFINITION PART IS INCORRECT
    The first step is slightly different. We can either pass in only the state variables and parameters,
    in which case the theory assumes that we start in METE and calculates the corresponding lambdas and derivatives,
    or we can pass in lambdas and derivatives explicitly to iterate from anywhere.
    The former is basically the perturbation way, and the latter is a generic iteration.
    
    Inputs
    t is the number of iteration steps to take
    s are state variables, call S, N
    p are parameters, call b0, d0, Nc
    Optional:
    l are lambdas as an array, so l[0] is lambda_1 etc.
    ds are derivatives of state variables, call dN
    dt is the length of the time step, 0.2 by default.
    Outputs:
    lambdas, state variables, and their derivatives at each time t. 
    The arrays are length t+1 since it prints the initial values.
    '''

    # Set up dataframes for storage
    ld = np.zeros([t+1,2])
    sd = pd.DataFrame(np.zeros([t+1,2]),columns=['S','N'])
    dsd = pd.DataFrame(np.zeros([t+1,2]),columns=['dS','dN'])

    # Initialize zeroth element
    # Copy if present
    if bool(l.size):        
        ld[0] = l.copy()
    else:
        ld[0] = rf.lambda_i(s)
    # Same for ds
    if bool(ds.size):
        dsd.iloc[0] = ds.copy()
    else:
        dsd.iloc[0] = rf.get_dXdt(ld[0],s,p)
    # Now copy state variables
    sd.iloc[0] = s.copy()
    
    # Iterate t times
    for i in np.arange(t):
        print("Iteration {}/{}".format(i+1,t))
        # Get required matrix and vector
        dl_mat,dl_vec = get_dl_matrix_vector(ld[i],sd.iloc[i],p,dsd.iloc[i])
        # Solve for dlambdas/dt
        dl = linalg.solve(dl_mat,dl_vec)
        
        # Now update state variables
        sd.iloc[i+1] = sd.iloc[i] + dt*dsd.iloc[i].values
        # Update derivatives from means over f and h
        dsd.iloc[i+1] = rf.get_dXdt(ld[i],sd.iloc[i+1],p)
        # Update lambdas
        ld[i+1] = ld[i] + dt*dl
    return ld,sd,dsd

def stability_test(ld,sd,p):
    '''
    This function outputs <n> for each step to check that lambda dynamics is satisfying the
    constraints it has to at each time t. Note this is a slow function.
    ld are lambdas as an array with length t and 2 columns, so l[0,0] is lambda_1 at time 0 etc.
    ss are state variables at each time t, call S, N
    p are parameters, call b0, d0, Nc
    '''
    # Get total time
    t = len(sd)
    # Set up arrays
    N_S = np.zeros(t)
    for i in np.arange(t):
        # Get means at each time, normalized.
        z = rf.mean_pow(0,ld[i],sd.iloc[i],p)
        N_S[i] = rf.mean_pow(1,ld[i],sd.iloc[i],p,z)
    return N_S


def dXdt_min(pmin,l,s,p):
    '''
    This function is needed for get_ss_params as it changes which parameters are minimized over. It essentially
    just changes the order of the inputs from rf.get_dXdt.
    pmin is the parameters optimized over. Right now that is d0
    '''
    # Change p to incorporate the new pmin.
    pnew = p.copy()
    pnew['d0'] = pmin[0]
    # Just minimize dN since dS should already be 0
    return rf.get_dXdt(l,s,pnew)['dN']
    
def get_ss_params(s,pi):
    '''
    This function gets the steady state parameter values so that we can start from something roughly stable.
    It does this by fixing b0 and Nc, and solving the constraints for d0.
    Note that which parameters are optimized over can be changed according to the function dXdt_min, which is what
    we are minimizing in practice.
    Inputs state variables and initial parameters guess
    Outputs new parameters
    '''
    # Get initial lambdas
    li = rf.lambda_i(s)
    sol = root(dXdt_min,[pi['d0']],args=(li,s,pi))
    pnew = pi.copy()
    pnew['d0'] = sol.x[0]
    return pnew

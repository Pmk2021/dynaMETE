'''
This file defines three functions that get the needed means for lambda dynamics,
get the needed matrix and vector for inversion, and define the necessary iteration scheme.
This version uses the flexible definitions for cov and mean, not the hard coded ones. This will
make it much slower, but it also means that we can change f,h,q much easier.
This file uses the toy model with only n and no e dependence.
'''

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import root #<- better than fsolve apparently.
import DynaMETE_Rfunctions_ToyModel as rf

def get_dl_matrix_vector(l,s,p,ds):
    '''
    Calculates the matrix and vector of covariances needed for lambda dynamics
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
    # Covariances. Note here we also have to pass ds since the derivatives take 5 variables
    # It has to go after z because *args is after z
    covndf = rf.cov(lambda n,s,p,ds: n,rf.dfdt,l,s,p,z,ds)

    # Vectors, including normalization in proper spot.
    v2 = -ds['dN']/s['S']-covndf*l[1]

    return np.array([r1,r2]),np.array([0,v2])



##Return analytically derived D_Lambdas in form DLambda1, Dlambda2
def dlambda(l,s,p,ds, prev_ds, d_d):
    cur_d = p['d0']
    p['d0'] += -1 * d_d
    z = rf.mean_pow(0,l,s,p)
    mean_n = rf.mean_pow(1,l,s,p)/z
    covnn = rf.cov(lambda n,s,p: n,lambda n,s,p: n,l,s,p,z)
    covnn2 = rf.cov(lambda n,s,p: n*n,lambda n,s,p: n,l,s,p,z)
    covff = rf.cov(rf.f,rf.f,l,s,p,z)
    covnf = rf.cov(lambda n,s,p: n,rf.f,l,s,p,z)

    D_dndt = (ds['dN'] - prev_ds['dN'])
   
 
    D = covnn * covff - covnf * covnf

    #return (D_dndt/s['S'] - rf.cov(rf.dfdd, rf.dfdd, l,s,p,z) * d_d + l[1] * d_d * (covfdfdd - covndfdd*covff/covnf))*covnf/D
    DL1 = covnf/D * (D_dndt/s['S'] + d_d * (mean_n * s['N']/p['Nc'] + l[1] * covnn * s['N'] * covff/(covnf * p['Nc']) + l[1] * (p['b0'] * -1 * covnn * s['N']/p['Nc'] + 0.001 * covnn2 * s['N']/p['Nc'] + p['d0'] * covnn * s['N']**2 / p['Nc']**2)))
    
    DL2 = (l[1] * covnn * s['N'] * d_d/p['Nc'] - covnn * DL1)/covnf
    
    
    p['d0'] = cur_d
   
    return DL1, DL2

# Give defaults to l and ds so that by default we assume the lambdas and the derivatives have to be calculated from
# the corresponding state variable. Allow them as optional arguments if we want to iterate from a specific state.
def iterate(t,s,p,dt=0.2,l=np.array([]),ds=np.array([]), d_d = None, verbose = True):
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
        if verbose:
            print("Iteration {}/{}".format(i+1,t))
        # Get required matrix and vector

        # Now update state variables
        sd.iloc[i+1] = sd.iloc[i] + dt*dsd.iloc[i].values
        # Update derivatives from means over f and h
        dsd.iloc[i+1] = rf.get_dXdt(ld[i],sd.iloc[i+1],p)
        
        if(d_d != None):
            
            dl1, dl2 = dlambda(ld[i],sd.iloc[i],p,dsd.iloc[i+1],dsd.iloc[i],d_d)
            
            dl = np.array([dl1,dl2]) * 1/dt 
            #print(dl_new, dl)
            #dl = dl_new
            #print(dl * dt)


            
            d_d = None
        else:

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

def iterate_1(t,s,p,dt=0.2,l=np.array([]),ds=np.array([]), verbose=True):
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
        if(verbose):
            print("Iteration {}/{}".format(i+1,t))
        '''Step 1'''
        # Get required matrix and vector
        dl_mat,dl_vec = get_dl_matrix_vector(ld[i],sd.iloc[i],p,dsd.iloc[i])
        # Solve for dlambdas/dt
        dl_s_1 = linalg.solve(dl_mat,dl_vec)
        # Now update state variables
        state_s_1 = sd.iloc[i] + 1/2 * dt*dsd.iloc[i].values
        # Update derivatives from means over f and h
     
        dsd_s_1 = rf.get_dXdt(ld[i],state_s_1,p)
        # Update lambdas
        l_s_1 = ld[i] + 1/2 * dt*dl_s_1
        '''Step 2'''
        dl_mat_2,dl_vec_2 = get_dl_matrix_vector(l_s_1,state_s_1,p,dsd_s_1)
        dl_s_2 = linalg.solve(dl_mat_2,dl_vec_2)
        state_s_2 = sd.iloc[i] + 1/2 * dt*dsd_s_1.values
        dsd_s_2 = rf.get_dXdt(l_s_1,state_s_2,p)
        l_s_2 = ld[i] + 1/2 * dt*dl_s_2
        '''Step 3'''
        dl_mat_3,dl_vec_3 = get_dl_matrix_vector(l_s_2,state_s_2,p,dsd_s_2)
        dl_s_3 = linalg.solve(dl_mat_3,dl_vec_3)
        state_s_3 = sd.iloc[i] + dt*dsd_s_2.values
        dsd_s_3 = rf.get_dXdt(l_s_2,state_s_3,p)
        l_s_3 = ld[i] + dt*dl_s_3
        '''Step 4'''
        dl_mat_4,dl_vec_4 = get_dl_matrix_vector(l_s_3,state_s_3,p,dsd_s_3)
        dl_s_4 = linalg.solve(dl_mat_4,dl_vec_4)


        # Now update state variables
        sd.iloc[i+1] = sd.iloc[i] + dt*1/6 * (dsd.iloc[i].values + 2 * dsd_s_1.values + 2 * dsd_s_2.values + dsd_s_3.values)
        # Update derivatives from means over f and h
        dsd.iloc[i+1] = rf.get_dXdt(ld[i],sd.iloc[i+1],p)
        # Update lambdas
        ld[i+1] = ld[i] + dt*1/6 * (dl_s_1 + 2 * dl_s_2 + 2 * dl_s_3 + dl_s_4)
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

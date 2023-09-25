'''
This file takes the brute force approach to iterate the lambdas. It simply optimizes at each step.
'''

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy import integrate
import DynaMETE_Rfunctions_ToyModel as rf

def p():
    print('a')
# Now the constraints
def constraints(l,s,p,ds):
    '''Return all constraints in an array.
    l are lambdas
    s are state variables, call S, N
    p are parameters, call b0, d0, Nc
    ds are derivatives of state variables, call dS, dN
    '''
    # Calculate normalization
    z = rf.mean_pow(0,l,s,p)
    # Calculate <n> and <f>
    ns = rf.mean_pow(1,l,s,p,z=z)
    dns = rf.mean(rf.f,l,s,p,z=z)

    # n constraint
    ncon = s['N']/s['S'] - ns
    # dn constraint
    dncon = ds['dN']/s['S'] - dns

    return np.array([ncon,dncon])

# Iteration time!



def iterate(t,s0,p,new_N,dt,l0=np.array([]), ds0=np.array([]), verbose=False):
    '''
    This function will iterate DynaMETE t steps. Returns vectors of lambdas, state variables, and time derivatives.
    1. Update state variables using time derivatives
    2. Put new state variables into transition functions
    3. Update time derivatives
    4. Update structure function
    The first step is slightly different. We can either pass in only the state variables and parameters,
    in which case the theory assumes that we start in METE and calculates the corresponding lambdas and derivatives,
    or we can pass in lambdas and derivatives explicitly to iterate from anywhere.
    The former is basically the perturbation way, and the latter is a generic iteration.

    Inputs
    t is the integer number of steps
    l0 are initial lambdas
    s0 are initial state variables, call S, N
    p are parameters, call b0, d0, Nc
        Note that if we want to change this over time we have to put in an array for p, which isn't implemented here.
    ds0 are initial derivatives of state variables, call dS, dN
    dt is how much of one year is a time step. The default is 0.2 to make sure the steps are relatively small.'''

    # Make arrays of lambdas, state variables, and derivatives to store and return.
    # Note that we need +1 to keep the 0th element also
    lambdas = np.zeros([t+1,2])
    states = pd.DataFrame(np.zeros([t+1,2]),columns=['S','N'])
    dstates = pd.DataFrame(np.zeros([t+1,2]),columns=['dS','dN'])
    updated = False
    # Initialize zeroth element
    # Copy if present
    if bool(l0.size):
        lambdas[0] = l0.copy()
    else:
        lambdas[0] = rf.lambda_i(s0)
    # Same for ds
    if bool(ds0.size):
        dstates.iloc[0] = ds0.copy()
    else:
        dstates.iloc[0] = rf.get_dXdt(lambdas[0],s0,p)
    # Now copy state variables
    states.iloc[0] = s0.copy()

    # Iterate t times.
    for i in range(t):
        # Print out progress
        if(verbose == True):
            print('Iteration {:.0f}/{:.0f}'.format(i+1,t))

        # First update state variables with time derivatives, multiplied by how much of one year we want to step by
        states.iloc[i+1] = states.iloc[i] + dt*dstates.iloc[i].values
        if(updated == False):
            if(states.iloc[i]['N'] != new_N):
                states.iloc[i+1]['N'] = new_N
            updated = True
            for pl in range(100):
                dstates.iloc[i+1] = rf.get_dXdt(lambdas[i],states.iloc[i+1],p)
                # Now time for new lambdas. Use old l as starting point, new s and ds
                l = fsolve(constraints,lambdas[i]+0.00001,args=(states.iloc[i+1],p,dstates.iloc[i+1]))
                print(l, dstates.iloc[i+1])
                #print(l,j)
                lambdas[i] = l
            
        # Update derivatives from means over f and h
        
        dstates.iloc[i+1] = rf.get_dXdt(lambdas[i],states.iloc[i+1],p)

        # Now time for new lambdas. Use old l as starting point, new s and ds
        l = fsolve(constraints,lambdas[i]+0.00001,args=(states.iloc[i+1],p,dstates.iloc[i+1]))

        #print(l,j)
        lambdas[i+1] = l
        # Sanity check to make sure it worked.
        #print('Constraints (should be small!): {}'.format(constraints(l,states.iloc[i+1],p,dstates.iloc[i+1])))
        #print('Constraints (should be small!): {}'.format(constraints(j,states.iloc[i+1],p,dstates.iloc[i+1])))
        #print('Constraints (should be small!): {}'.format(constraints(k,states.iloc[i+1],p,dstates.iloc[i+1])))
        #print('')
        delta_D = 0
    return lambdas,states,dstates
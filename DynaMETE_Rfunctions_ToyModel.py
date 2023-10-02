'''
This file defines all of the necessary functions for DynaMETE, including the transition functions and
the structure function R. This function does NOT include sums over n, since it is designed to be a 
more flexible version incorporating different transition functions. This will be very slow for large N or E.
It also defines the METE constraint for beta, which is needed, and a function to obtain mete_lambdas.
This file uses the toy model with only n and no e dependence.
'''

# Import
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy import integrate

# Get initial lambda
def lambda_i(s):
    '''Initial lambda 1 given lambda 2 = 0 and state variables'''
    l1_guess = np.log(s['N']/(s['N']-1))
    nrange = np.arange(s['N'])+1
    l1_true = fsolve(lambda l1 : np.sum(nrange*np.exp(-l1*nrange))/np.sum(np.exp(-l1*nrange))-s['N']/s['S'],l1_guess)[0]
    li = np.array([l1_true,0])
    return li

# Transition functions
# The idea here is to make everything easy to change by changing only these functions.
# For f
# d0 = d2, d1 =  d1, b0 = r0

def fb0(s,p):
    return p['b0']
def fd0(s,p): # this one is d2
    return 0.02 * s['N']/p['Nc'] # - unperturbed (last parts need to be part of formula)
    # return p['d0']*s['N']/p['Nc'] - perturbed?
def fd1(s,p):
    # return 0.02 - unperturbed (0.001 originally)
    return p['d1']

# Nearly identical to dN/dt
# Last term is N/N0 * d0 * n^2 in code (fd0(s, p) returns p['d0'] * s['N'] / p['Nc'])
def f(n,s,p):
    '''Transition function for dN/dt. n is the microscopic variables.
    s are state variables, call S, N
    p are parameters, call b0, d0, Nc '''

    # d1*n*N/N0 in the code/Excel. And in the manuscript it d1*n*N/S_0 (from Pranav)
    # f(n, N) = b0 (birth rate) * n - d1 * n**2 - p['d2'] (d0) * n**2 * N/s (in draft)
    # f(n, N) = b0 (birth rate) * n - fd1(s, p) * n ** 2 - N/N0 * p['d2'] * n^2 (in code)
    # Not an issue: N0 and S0 are constants
    return fb0(s,p) * n - fd1(s,p) * n**2 - fd0(s,p) * n
    # fd1 proportional to n squared- increase number of people, competition increases quadratically
    #    - animals are killing each other and stuff
    #    - interspecies competition
    # fd0- related to like. natural death rate for 1 species
    # kinda similar to logistic equation; lower bound, upper bound in ecosystem (not below 0, carrying capacity)

# Also need derivatives for lambda dynamics. Note that these have to be manually editted for alternate f,h,q
def dfdt(n,s,p,ds):
    return -1 * fd0(s,p)/s['N']*ds['dN']*n

def dfdd(n, s, p):
    # return n*s['N']/p['Sc'] - for perturb d0
    return - n ** 2

def dfdd_sum(l,s,p,z):
    nrange = np.arange(s['N'])+1
    return np.sum(R(nrange,l,s,p) * nrange * dfdd(nrange, s, p))/z
# R itself
def R(n,l,s,p):
    '''Unnormalized struture function for toy model.
    n is microscopic variables.
    l are lambdas
    s are state variables, call S, N
    p are parameters, call b0, d0, Nc '''
    return np.exp(-l[0]*n-l[1]*f(n,s,p))

def Rd(n,l,s,p):
    '''Unnormalized struture function for toy model.
    n is microscopic variables.
    l are lambdas
    s are state variables, call S, N
    p are parameters, call b0, d0, Nc '''
    return np.exp(-l[0]*n-l[1]*dfdd(n,s,p))

# For calculating a single mean with specific powers of n and e
def mean_pow(npow,l,s,p,z=1):
    '''
    This function returns the mean of n^npow over the R function.
    It is NOT normalized, but it does take in z as an optional argument to normalize.
    This function sums numerically over n.
    Note that npow=0 corresponds to Z, so by default these are not normalized.
    l are lambdas
    s are state variables, call S, N
    p are parameters, call b0, d0, Nc
    '''
    nrange = np.arange(s['N'])+1
    return np.sum(nrange**npow*R(nrange,l,s,p))/z

#calculating derivatives for new lambda update
def mean_pow_d(npow,l,s,p,z=1):
    '''
    This function returns the mean of n^npow over the R function.
    It is NOT normalized, but it does take in z as an optional argument to normalize.
    This function sums numerically over n.
    Note that npow=0 corresponds to Z, so by default these are not normalized.
    l are lambdas
    s are state variables, call S, N
    p are parameters, call b0, d0, Nc
    '''
    nrange = np.arange(s['N'])+1
    return np.sum(nrange**npow*Rd(nrange,l,s,p))/z

# For calculating a covariance with specific powers of n and e for each function
def cov_pow(npow,l,s,p,z):
    '''
    This function returns the covariance of two functions with the form n^npow over the R function.
    You have to pass in the normalization so that things are faster than calculating normalization each time.
    npow should be a 2d array
    This function sums numerically over n.
    z is the normalization
    l are lambdas
    s are state variables, call S, N
    p are parameters, call b0, d0, Nc
    '''
    nrange = np.arange(s['N'])+1
    # Get integral over both functions
    ff = np.sum(nrange**np.sum(npow)*R(nrange,l,s,p))/z
    # Get integral over each function
    f1f2 = 1
    for nn in npow:
        f1f2 *= np.sum(nrange**nn*R(nrange,l,s,p))/z
    return ff-f1f2

# For calculating a single mean over an arbitrary function
# For these next two I could get clever and use *args etc to make it easier to pass in n and e
# But I just don't want to bother right now.
# Also, if you really want to do that, just use mean_pow since that's WAY easier
def mean(func,l,s,p,*args,z=1):
    '''
    This function returns the mean of an arbitrary function over the R function.
    It is NOT normalized, but it does take in z as an optional argument to normalize.
    Because I put *args first, you have to use z=z0 if you want to put in a normalization.
    The arbitrary function must take arguments of the form (n,s,p) for this to work.
    This is the form of the f function above.
    You can pass additional arguments as required for the function (ie. pass ds for df/dt)
    To pass in n, use lambda n,e,s,p: n or similar
    Alternatively, use mean_pow
    This function sums numerically over n.
    z is the normalization
    l are lambdas
    s are state variables, call S, N
    p are parameters, call b0, d0, d1, Nc
    '''
    nrange = np.arange(s['N'])+1
    # Below is to make this easier for lambda functions, but it isn't worth it. Just require s and p passed, 
    # and let other things be passed as args if needed.
    # Check if we need args by looking at function passed in
#    funcargs = func.__code__.co_argcount
#    if funcargs >= 4:
#        args = s,p,args
    return np.sum(R(nrange,l,s,p)*func(nrange,s,p,*args))/z

# For calculating a covariance
# Note if you want to do this with non-functions, use cov_pow
def cov(func1,func2,l,s,p,z,*args):
    '''
    This function returns the covariance of two arbitrary functions over the R function.
    You have to pass in the normalization so that things are faster than calculating normalization each time.
    The arbitrary functions must take arguments of the form (n,s,p) for this to work.
    This is the form of the f function above.
    You can pass additional arguments as required for the function (ie. pass ds for df/dt)
    To pass in n use lambda n,s,p: n
    This function sums numerically over n.
    z is the normalization
    l are lambdas
    s are state variables, call S, N
    p are parameters, call b0, d0, Nc
    '''
    nrange = np.arange(s['N'])+1
    # Get integral over both functions
    ffeint = R(nrange,l,s,p)*func1(nrange,s,p,*args)*func2(nrange,s,p,*args)
    ff = np.sum(ffeint)/z
    # Get integral over each function
    f1f2 = 1
    for func in [func1,func2]:
        feint = R(nrange,l,s,p)*func(nrange,s,p,*args)
        f1f2 *= np.sum(feint)/z
    return ff-f1f2
    
def get_dXdt(l,s,p):
    '''
    Returns the time derivatives of the state variables. 
    Inputs lambdas, state variables, and parameters.
    Outputs a pandas series of ds.
    '''
    # Create storage
    ds = pd.Series(np.zeros(2),index=['dS','dN'])
    # To normalize
    z = mean_pow(0,l,s,p)
    ds['dN'] = s['S']*mean(f,l,s,p,z=z)
    return ds


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python programs accompanying the article:

Vazquez, A. R. and Xu, H. (2022). An integer programming algorithm for constructing
maximin distance designs from good lattice point sets. Submitted to Statistica Sinica.

@author: Alan Roberto Vazquez
@email: alanrvazquez@gmail.com
"""

#%% Load required packages.---------------------------------------------------

from gurobipy import * # To solve optimization problems
import numpy as np # To perform matrix operations
from itertools import product, combinations # To use combinations and permutations
import operator as op
import math
import functools

#%% Python Classes.-----------------------------------------------------------
class DesignClass(object):
    """
    Python class for outputs of the IPAlgorithm function.
    """
    def __init__(self, design, Lqdistance, q):
        self.design = design
        self.Lqdistance = Lqdistance
        self.q = q
        
#%% Auxiliary Functions.------------------------------------------------------
def nchoosek(n,r): 
    """ Compute the total number of combinations of 'n' in 'r'.

    Input:
        n: number (int).
        r: number (int).
    Output:
        res: number of combinations (int).
    """
    
    if n >= r and n >= 0 and r >= 0:
        r = min(r, n-r)
        numer = functools.reduce(op.mul, range(n, n-r, -1), 1)
        denom = functools.reduce(op.mul, range(1, r+1), 1)
        res = numer / denom
    else :
        res = 0
    return int(res)


def Williams(x, N):
    """ Compute the Williams' transformation.

    Input:
        x: number (int).
        N: number (int).
    Output:
        number (int).
    """

    if x < N/2:
        return 2*x
    else :
        return 2*(N-x) - 1    

def generate_primes(start, end):
    """ Generate primes between two numbers.

    Input:
        start: left end of interval (int).
        end: right end of interval (int).
    Output:
        prime_numbers: prime numbers between 'start' and 'end' (list).
    """    
    
    prime_numbers = []
    for i in range(start, end+1):
        if i>1:
            for j in range(2,i):
                if(i % j==0):
                    break
            else:
                prime_numbers.append(i)

    return prime_numbers    
    

def smallest_largest_prime(N):
    """ Generate the smallest largest prime of a given number.

    Input:
        N: number (int).
    Output:
        pp: Smallest largest prime than N (int).
    """  

    my_cond = True
    c = 0
    while my_cond :
        pp = N + c
        
        for j in range(2, pp):
            # Check if pp is prime
            v = pp % j 
            if (v==0):
                break
            
        if v != 0:
            my_cond = False
        c = c + 1   
        
    return pp
    
def tolerance_integer(a, tol = 0.001):
    """ Transform a numeric value to a binary variable.
        This function corrects for possible misspecifications of the values
        of the integer decision variables resulting from the Gurobi Optimization.
    Input:
        a: variable value (int or float).
        tol: tolerance to be considered as integer (float).
    Output:
        zint: the integer value of a (int).
    """
    low_a = math.floor(a)
    up_a = math.ceil(a)
    
    if abs(a - up_a) < tol:
        zint = up_a
    elif abs(a - low_a) < tol:
        zint = low_a
    else :
        zint = a
        print('Decision variable z did not converge to an integer, ' + str(a))
    return(zint) 

def Generate_ElementWise_Distances(C, q):
    """ Generate all element-wise distances of the columns in an LHD.
    Input:
        C: N x n LHD (np array).
        q: The type of Lq-distance.
    Output:
        A: Nchk x n numpy array containing element-wise distances (numpy array).
    """  
    N, n = np.shape(C)
    # Allocate distances in each column
    Nchk = nchoosek(N,2)
    A = np.zeros((Nchk,n))
    # Create array of permutations.
    for i in range(n):         
        e = 0
        for eone, etwo in itertools.combinations(range(N), 2):
            A[e,i] = (abs(C[eone,i] - C[etwo,i]))**q
            e = e + 1 
    
    return A

#%% Functions to construct LHDs using good lattice point sets and Williams transformation.

def All_GLP_Williams(N, q = 1):
    """ Construct an LHD using good lattice point designs and the Williams'
        transformation.
    Input:
        N: number of runs (int).
        q: The type of Lq-distance (int).
    Output:
        C: N x n numpy array containing the best LHD (numpy array). 
        'n' equals phi(N). 
    """    
    # Construct Latin Hypercube Design
    h = list()
    for i in range(N):
        if math.gcd(i+1,N) == 1:
            h.append(i+1)
    n = len(h)
    # Generate good lattice point design.
    GLP = np.zeros((N,n))
    for i in range(N):
        for j in range(n):
            GLP[i,j] = np.mod((i+1)*h[j], N)
    # Generate permuted design.
    Designs = np.zeros((N,n,N))
    for b in range(N):
        Designs[:,:,b] = np.mod(GLP + b, N)
    # Apply William Transformation.
    WillDesigns = np.zeros((N,n,N))
    for d in range(N):
        for i in range(N):
            for j in range(n):
                WillDesigns[i,j,d] = Williams(Designs[i,j,d], N)
    min_dist_designs = list()
    
    for d in range(N):
        dist_design = list()
        for i, j in itertools.combinations(range(N), 2):
            t_dist = sum( abs(WillDesigns[i,:,d] - WillDesigns[j,:,d])**q ) 
            dist_design.append(t_dist)
        min_dist = min(dist_design)  
        min_dist_designs.append(min_dist)
     
    index, value = max(enumerate(min_dist_designs), key=op.itemgetter(1))
    C = WillDesigns[:,:,index]
    return C


def optimal_b_value(N):
    """ Generate optimal value of b. This function implements the 
    procedure in Section 3 of Wang, L, Xiao, Q., and Xu, H. (2018).
    Optimal maximin L1-distance Latin hypercube designs based on 
    good lattice point sets. The Annals of Statistics, 46:3741-3766.

    Input:
        N: number of runs (int). Must be an odd prime.
    Output:
        b: optimal value (int).   
    """      
    c_zero = int(np.sqrt((N**2 - 1)/12))
    def_poly = c_zero**2 + 2*((c_zero + 1)**2)
    if def_poly >= (N**2 - 1)/4:
        c = c_zero
    else :
        c = c_zero + 1
        
    y = (N-1)/2 - c # We adpot this one. - c
    if (y % 2) == 0: 
        b = int(y/2)
    else :
        aux_const = (2*N - y - 1)/2
        b = int(aux_const)
        
    return b    

def GLP_Williams(N):
    """ Construct a LHD using good lattice point designs and the Williams'
        transformation with optimal value of 'b.'
    Input:
        N: number of runs (int). Must be an odd prime.
    Output:
        C: N x n numpy array containing the best LHD (numpy array). 
        'n' equals phi(N).
    """    
    # Construct Latin Hypercube Design
    h = list()
    for i in range(N):
        if math.gcd(i+1,N) == 1:
            h.append(i+1)
    n = len(h)
    # Generate good lattice point design.
    GLP = np.zeros((N,n))
    for i in range(N):
        for j in range(n):
            GLP[i,j] = np.mod((i+1)*h[j], N)
     
    # Select optimal value for b.
    b = optimal_b_value(N)        
    # Generate permuted design.
    Design = np.mod(GLP + b, N)
    # Apply William Transformation.
    C = np.zeros((N,n))
    for i in range(N):
        for j in range(n):
            C[i,j] = Williams(Design[i,j], N)
    
    return C

#%% Integer Programming Algorithm.-------------------------------
def get_solution(model, N):
    """ Obtain the solutions from a gurobi model.
    Input:
        model: gurobi model.
        N: number of binary variables (int).
    Output:
        Z: Vector containing the optimal solutions of the  binary variables
        (numpy array).
    """
    Z = np.zeros(N)
    
    for i in range(N):
        zsol = model.getVarByName('z_'+ str(i))
        Z[i] = tolerance_integer(zsol.x)
            
    return Z  

def construct_design(z_sol, C, N, tol = 0.001):
    """ Construct design matrix from a solution to the IP problem.
    Input:
        z_sol: integer solution of IP program (numpy array).  
        C: candidate set (numpy array).
        N: run size (int).
        tol: tolerance to be considered as integer (float).
    Output:
        D: N x k LHD (numpy array). 'k' is the sum of all elements in z_sol.
    """
    nbin, = np.shape(z_sol)
    design = np.zeros((N,1))
    for i in range(nbin):
        candidate_column = C[:,i].reshape((N,1))
        if z_sol[i] > tol:
            design = np.hstack((design, candidate_column))    
    D = design[:,1:]
    return(D)

def IPmodel(A, k, q, max_time = 60, verbose = True, symmetry = -1, gurobi_log_name = 'IP_LHD.log'):
    """ IP problem formulation in Section 4.3 of the main text.
    Input:
        A: distance matrix of candidate set (numpy array).  
        k: number of factors in the LHD (int).
        q: The type of Lq-distance (int).
        max_time: maximum number of time for Gurobi in seconds (int).
        verbose: print the optimization progress? (boolean)
        symmetry: Gurobi parameter to control the symmetry detection (int).
                  See Gurobi manual for details. The default is -1. 
        gurobi_log_name: name of the Gurobi log which provides information 
                         on the solution of the problem (char).
    Output:
        z_sol: Vector with solution of binary decision variables (numpy array). 
        model.objval: Value of the objective function (int).
    """

    Nchk, nbin = np.shape(A)
    # Create IP model.
    model = Model('IP_for_LHD')
    model.params.outputflag = 0 
    if verbose:
        model.params.outputflag = 1
        model.params.LogFile = gurobi_log_name
    model.params.Symmetry = symmetry    
    
    #==CREATE VARIABLES====================================================
    # Binary variables.    
    z_vecvar = []
    for i in range(nbin):
        def_bin_var = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name = "z_{0}".format(i))
        z_vecvar.append(def_bin_var) 
    # Integer variable
    t = model.addVar(lb=0, vtype=GRB.INTEGER, name = "t")
    model.update()
    
    #==DEFINE AND ADD CONSTRAINTS==========================================
    for i in range(Nchk):
        model.addConstr(quicksum(z_vecvar[l]*A[i,l] for l in range(nbin) ) >= t, name = "dist_pair_"+str(i) )    
      
    # Run size of the design.
    model.addConstr(sum(z_vecvar) == k, name = 'number_factors')
    # Set first column.
    model.addConstr(z_vecvar[0] == 1, name = 'first_col')
    model.update()
    #==SET OBJECTIVE FUNCTION==============================================
    model.setObjective(t, GRB.MAXIMIZE)
            
    model.update() 
    
    #==SET TIME LIMIT====================================================== 
    model.params.timeLimit = max_time     
    
    #==OPTIMIZE MODEL======================================================
    model.optimize()       
    
    #==OBTAIN SOLUTION===================================================== 
    z_sol = get_solution(model, nbin)

    return z_sol, model.objval

#%% Implementation of leaeve-one-out method.----------------------------------
def create_intervals(x, N):
    """ Auxiliar function to arrange the levels of a column in an LHD in which
        a point or row has been removed.
    Input:
        x: number (int).  
        N: number  (int).
    Output:
        intervals: Set of intervals (list). 
    """    
    index = np.insert(x, 0, 0)
    index = np.append(index, N-1)
    index.sort()
    
    # Create intervals.
    intervals = []
    for i in range(len(index)-1):
        intervals.append([index[i], index[i+1]])

    return intervals


def remove_points(D, r):
    """ Remove points from an LHD.

    Input:
        D: N x n LHD (numpy array).  
        r: index of row or point to remove (int).
    Output:
        subD: (N-1) x n LHD without point r (numpy array). The columns of 
              subD are arranged so that it is an LHD too.

    """
    N, n = np.shape(D)
    rows_to_remove = D[r,:]
    subD = np.delete(D, r, axis = 0)
    
    Nsubdes, n = np.shape(subD)
    for i in range(n):
        def_set = rows_to_remove[:,i]
        
        # Create intervals.
        intervals = create_intervals(def_set, N)
        
        # Turn sub design into a latin hypercube design.
        c = 0
        for inter in intervals:
            location = np.where(np.logical_and(subD[:,i] > inter[0], subD[:,i] <= inter[1]))
            subD[location[0],i] = subD[location[0],i] - c
            c = c + 1
               
    return subD

def delete_one_method(D, q = 1):
    """ Systematic application of the leave-one-out method.

    Input:
        D: N x n LHD (numpy array).  
        q: The type of Lq-distance (int).
    Output:
        (N-1) x n LHD without point a point (numpy array). 
        The point that is removed is the one that leads to the
        best subdesign in terms of the Lq-distance.

    """
    nrows, ncols = np.shape(D)
    result_dist = list()
    subdesigns = np.zeros((nrows-1, ncols, nrows))
    for r in range(nrows):

        subdesigns[:,:,r] = remove_points(D, [r])
        A = Generate_ElementWise_Distances(subdesigns[:,:,r], q)
        subdistances = np.sum(A, axis = 1)
        min_dist = np.min(subdistances)
        result_dist.append(min_dist)
        
    best_r = np.argmax(result_dist)
    return subdesigns[:,:,best_r]

#%% Modified IP algorithm.----------------------------------

def ModifiedIP(N, n, q = 1, max_time = 300, verbose = True, gurobi_log_name = 'IP_Alg_LHD.log'):
    """ Standard and modified IP algorithm.
    Input:
        N: run size (int).  
        n: number of factors (int).
        q: The type of Lq-distance (int).
        max_time: maximum number of time for Gurobi in seconds (int).
        verbose: print the optimization progress? (boolean)
        gurobi_log_name: name of the Gurobi log which provides information 
                         on the solution of the problem (char).
    Output:
        Object of class 'DesignClass'.
    """  
    
    p = smallest_largest_prime(N)
    
    # Step 0. Generate parent maximin latin hypercube design.
    C = GLP_Williams(p)
    A = Generate_ElementWise_Distances(C, q)
    if verbose:
        print("Parent design with ", np.shape(C)[0], " rows and ", np.shape(C)[1], " columns. \n")
    
    # Step 1. Project C onto n columns. 
    subset_cols, minmax_distane = IPmodel(A, n, q, max_time, verbose, -1, gurobi_log_name)
    D = C[:, subset_cols == 1]
    
    # Step 2. Reduce design D to have N runs.
    if N < p:
        for i in range(N, p):
            D = delete_one_method(D, q) #Cut design
    
    # Step 3. Evaluate resulting design.        
    B = Generate_ElementWise_Distances(D, q)        
    minmax_dist = min(np.sum(B,1))
    
    return DesignClass(D, minmax_dist, q)
    
    
    
    
    

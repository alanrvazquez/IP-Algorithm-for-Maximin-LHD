#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples of Python programs accompanying the article:

Vazquez, A. R. and Xu, H. (2022). An integer programming algorithm for constructing
maximin distance designs from good lattice point sets. Submitted to Statistica Sinica.

@author: Alan Roberto Vazquez
@email: alanrvazquez@gmail.com
"""

import numpy as np
from IP_Functions import *

#%% Example 1. Generate the 16-run 8-factor LHD in Table 2 in the main text.
N = 16 # Number of runs.
k = 8 # Number of factors.
q = 1 # Lq-distance.

# STEP 1. Generate candidate set.---------------------------------
# Generate all LHDs from GLP sets and the Williams' transformation.
h = list()
for i in range(N):
    if math.gcd(i+1,N) == 1:
        h.append(i+1)
n = len(h)
# Generate good lattice point (GLP) set.
GLP = np.zeros((N,n))
for i in range(N):
    for j in range(n):
        GLP[i,j] = np.mod((i+1)*h[j], N)
# Apply linear permutations to GLP set.        
Designs = np.zeros((N,n,N))
for b in range(N):
    Designs[:,:,b] = np.mod(GLP + b, N)
# Apply Williams' transformation to each linearly permuted GLP set.
WillDesigns = np.zeros((N,n,N))
for d in range(N):
    for i in range(N):
        for j in range(n):
            WillDesigns[i,j,d] = Williams(Designs[i,j,d], N)
# Generate candidate set.
C = WillDesigns[:,:,0]
for i in range(1,N):
    C = np.concatenate( (C, WillDesigns[:,:,i]), axis = 1)
# Generate distance matrix.
A = Generate_ElementWise_Distances(C, q)
# Remove repeat columns.
B, s = np.unique(A, axis = 1, return_index = True) 
redC = C[:, s]
# Remove repeat rows.
B = np.unique(B, axis = 0)


# STEP 2. Build and solve IP problem formulation.---------------
subset_cols, minmax_distance = IPmodel(B, k, q)

# STEP 3. Print LHD.--------------------------------------------
print(redC[:, subset_cols == 1])

# Print value of Lq-distance.
print(minmax_distance)

#%% Example 2. Generate the 31-run 7-factor LHD in Table 3 in the main text.
LHD_two = ModifiedIP(N = 31, n = 20, q = 1)

# Print LHD.
print(LHD_two.design)

# Print value of Lq-distance.
print(LHD_two.Lqdistance)


#%% Example 3. Generate the 44-run 23-factor LHD in Table 4 in the main text.
LHD_three = ModifiedIP(N = 44, n = 23, q = 1)

# Print LHD.
print(LHD_three.design)

# Print value of Lq-distance.
print(LHD_three.Lqdistance)

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
from scipy.linalg import solve
import math


# define hermite polynomial function
def hmt(x,k):
    if k==1:
        result=[1]
    elif k==2:
        result=[1, 2*x]
    elif k==3:
        result=[1, 2*x, 4*(x**2)-2]
    else:
        result=[1, 2*x, 4*(x**2)-2, 8*(x**3)-12*x]
    return result


# define laguerre polynomial function
def lgr(x,k):
    if k==1:
        result=[np.exp(-x/2)]
    elif k==2:
        result=[np.exp(-x/2), (np.exp(-x/2))*(1-x)]
    elif k==3:
        result=[np.exp(-x/2), (np.exp(-x/2))*(1-x), (np.exp(-x/2))*(1-2*x+(x**2)/2)]
    else:
        result=[np.exp(-x/2), (np.exp(-x/2))*(1-x), (np.exp(-x/2))*(1-2*x+(x**2)/2), \
                (np.exp(-x/2))*(1-3*x+3*(x**2)/2-(x**3)/6)]
    return result

  
# define simple polynomial function
def spl(x,k):
    if k==1:
        result=[1]
    elif k==2:
        result=[1, x]
    elif k==3:
        result=[1, x, x**2]
    else:
        result=[1, x, x**2, x**3]
    return result
  

# define a function to compute realization of future St
def realization(j,i,n,ev,ind,r,dt):
    y = []
    for z in range(i+1,n,1):
        y.append(ev[j,z] * np.exp(-r*dt*(z-i)) * ind[j,z])
    return sum(y)


 ## Compute prices of American call options via Least Squares Monte Carlo Simulation

def lsmc(s0, x, r, t, sigma, k, method=):
    m = 50000 # path number
    dt = 1/np.sqrt(m)
    n = int(np.round(t/dt))

    # create stock price matrix
    St1 = np.zeros(m*n).reshape(m, n)
    St1[:,0] = 40
    St2 = np.zeros(m*n).reshape(m, n)
    St2[:,0] = 40
    for j in range(0, m):
        for i in range(0, n-1):
            Z1 = np.random.normal(0,1,n)
            Z2 = -Z1  # create antithetics variable
            St1[j,i+1] = St1[j,i] * (1 + r*dt + sigma*np.sqrt(dt)*Z1[i+1])
            St2[j,i+1] = St2[j,i] * (1 + r*dt + sigma*np.sqrt(dt)*Z2[i+1])
    
    # create blank index matrix & update the last column
    I1 = np.zeros(m*n).reshape(m, n)
    for j in range(0, m):
        if max(x-St1[j,i],0) > 0:
            I1[j,n-1] = 1
            
    I2 = np.zeros(m*n).reshape(m, n)
    for j in range(0, m):
        if max(x-St2[j,i],0) > 0:
            I2[j,n-1] = 1
    
    # create blank continuous value & exercise value matrix
    CV1 = np.zeros(m*n).reshape(m, n)
    EV1 = np.zeros(m*n).reshape(m, n)
    for i in range(n-1, 0, -1):
        for j in range(0, m):
            EV1[j,i] = max(x-St1[j,i],0)
            
    CV2 = np.zeros(m*n).reshape(m, n)
    EV2 = np.zeros(m*n).reshape(m, n)
    for i in range(n-1, 0, -1):
        for j in range(0, m):
            EV2[j,i] = max(x-St2[j,i],0)
    
    # create realization variable Yi
    Y1 = np.zeros(m*n).reshape(m, n)
    Y2 = np.zeros(m*n).reshape(m, n)
    
    # compute expected continuous value using non-linear least square method & update index matrix
    for i in range(n-1, 1, -1):
        
        L1 = np.zeros(m*k).reshape(m,k) # find L
        L2 = np.zeros(m*k).reshape(m,k)
        for j in range(0, m):
            if method == 'hermite':
                L1[j] = hmt(St1[j][i-1],k)
                L2[j] = hmt(St2[j][i-1],k)
            elif method == 'laguerre':
                L1[j] = lgr(St1[j][i-1],k)
                L2[j] = lgr(St2[j][i-1],k)
            elif method == 'simple':
                L1[j] = spl(St1[j][i-1],k)
                L2[j] = spl(St2[j][i-1],k)
            
        for j in range(0, m):
            Y1[j,i-1] = realization(j,i-1,n,EV1,I1,r,dt)
            Y2[j,i-1] = realization(j,i-1,n,EV2,I2,r,dt)
            
        A1 = np.dot(np.transpose(L1), L1)  # compute A
        b1 = np.dot(np.transpose(L1), Y1[:,i-1])  # compute b
        a1 = solve(A1,b1)  # find a
        A2 = np.dot(np.transpose(L2), L2)
        b2 = np.dot(np.transpose(L2), Y2[:,i-1])
        a2 = solve(A2,b2) 
        
        for j in range(0, m):
            CV1[j,i-1] = np.dot(a1, np.transpose(L1[j]))  # compute expected continuous value
            CV2[j,i-1] = np.dot(a2, np.transpose(L2[j]))
            
            # set index=1 if EV is larger than CV
            if EV1[j,i-1] >= CV1[j,i-1]:
                I1[j,i-1] = 1
            if EV2[j,i-1] >= CV2[j,i-1]:
                I2[j,i-1] = 1
            
            # set index to the right all 0 if index=1
            if I1[j,i-1] == 1:
                I1[j,i:] = 0
            if I2[j,i-1] == 1:
                I2[j,i:] = 0
                
    # update exercise value matrix
    E1 = np.zeros(m*n).reshape(m, n)
    E2 = np.zeros(m*n).reshape(m, n)
    for i in range(n-1, 0, -1):
        for j in range(0, m):
            E1[j,i] = EV1[j,i]*I1[j,i]
            E2[j,i] = EV2[j,i]*I2[j,i]
    
    # compute final value
    Yi1=[]
    Yi2=[]
    for j in range(0,m):
        Yi1.append(realization(j,0,n,E1,I1,r,dt))
        Yi2.append(realization(j,0,n,E2,I2,r,dt))
    
    return (np.mean(Yi1)+np.mean(Yi2))/2


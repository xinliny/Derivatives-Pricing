import pandas as pd
import numpy as np
import random
import scipy.stats
from scipy.stats import norm
from scipy.linalg import solve
import math


# COMPUTE PRICES OF FIX-STRIKE LOOKBACK OPTIONS

def lookback(s0, k, t, r, sig, type_):
    price = []
    for k in range(0,1000):
        n = 1000
        z = np.random.normal(0,1,n)
        dt = t/n
        
        s = np.zeros(n)
        s[0] = s0
        for i in range(1,n):
            s[i] = s[i-1] * (1+r*dt+sig*np.sqrt(dt)*z[i])
        s_max = max(s)
        s_min = min(s)
        
        if type_ == 'call':
          p = max(s_max-k, 0)
        elif type_ == 'put':
          p = min(k-s_min, 0)
        price.append(p * np.exp(-r*t))
    
    return np.mean(price)
  
  
# COMPUTE VALUE OF COLLATERAL WHICH FOLLOWS A JUMP DIFFUSION PROCESS

def collateral(lambda1, lambda2, T):
    path = 1
    pathnum = 1000
    DefOption = []
    Et = []
    
    # define all parameters
    mu = -0.1
    L0 = 22000
    sigma = 0.2
    gamma = -0.4
    r0 = 0.02
    delta = 0.25
    alpha = 0.7
    epsilon = 0.95
    R = r0+delta*lambda2
    r = R/12
    n = T*12
    pmt = L0*r/(1-(1+r)**(-n))
    a = pmt/r
    b = pmt/(r*(1+r)**n)
    c = 1+r
    beta = (epsilon-alpha)/T
    dt = 1/100
    num = int(T/dt) # steps in each path
    
    while path < pathnum:
    
        # simulate jump times
        Y = np.random.exponential(1/(lambda1*T),num)
        y = 0
        i = 0
        tao = []
        while y < T:
            y = Y[i] + y
            tao.append(y)
            i += 1
     
        # simulate the value of collateral
        V = np.zeros(num)
        V[0] = 20000
        z = np.random.normal(0,1,num)
        j = 0
        for i in range(0, num-1):
            if (i+1)*dt > tao[j]:
                V[i+1] = V[i]*(1+gamma)*(1+ r*dt + sigma*np.sqrt(dt)*z[i])
                j = j+1
            else:
                V[i+1] = V[i]*(1+ r*dt + sigma*np.sqrt(dt)*z[i])
         
        # simulate outstanding balance of the loan
        q = np.zeros(num)
        L = np.zeros(num)
        for i in range(0,num):
            q[i] = alpha + beta*(i*dt)
            L[i] = a - b*(c**(12*i*dt))
     
        # find stopping time Q
        qq = np.zeros(num)
        for i in range(0, num):
            if V[i] <= q[i]*L[i]:
                qq[i] = i
            else:
                qq[i] = num-1
        Qi = int(min(qq))
        Q = Qi*dt
        
        # find stopping time S
        N = np.random.poisson(lambda2*T,num)
        s = np.zeros(num)
        for i in range(0, num):
            if N[i] > 0:
                s[i] = i
            else:
                s[i] = num-1
        Si = int(min(s))
        S = Si*dt
    
        # compute default option value
        if Q < S:
            D = max((L[Qi]-epsilon*V[Qi]),0)*np.exp(-r0*Q)
        elif Q > S:
            D = abs(L[Si]-epsilon*V[Si])*np.exp(-r0*S)
        else:
            D = 0
        DefOption.append(D)
        
        # find expected default time
        if Q < S:
            et = Q
        elif Q > S:
            et = S
        else:
            et = 0
        Et.append(et)
        
        path = path+1
    
    DD = [x for x in DefOption if x!=0]
    ett = [x for x in Et if x !=0]
    value = np.mean(DD)
    prob = len(DD)/len(DefOption)
    time = np.mean(ett)
        
    return [value, prob, time]

import numpy as np
import random
import scipy.stats
from scipy.stats import norm
import math


def combos(n, i):
  return math.factorial(n) / (math.factorial(n-i)*math.factorial(i))

# Compute prices of European call options via Binomial Method
def binom_e_call_1(s0, k, r, t, sigma, n):
  dt = t/n
  
  # defined binomial methods parameters: upward & downward probabilities
  c = (np.exp(-r*dt) + np.exp((r+sigma**2)*dt)) / 2
  d = c - np.sqrt(c**2 - 1)
  u = 1/d
  p = (np.exp(r*dt)-d) / (u-d)
  
  value = 0
  for i in range(n+1):
      node_prob = combos(n,i) * p**i * (1-p)**(n-i)
      st = s0 * u**i * d**(n-i)
      value += max(st-k,0) * node_prob
  
  return value * np.exp(-r*t)

# alternaltive
def binom_e_call_2(s0, k, r, t, sigma, n):
  dt = t/n
   
  u = np.exp(r*dt) * (1 + np.sqrt(np.exp((sigma**2)*dt)-1))
  d = np.exp(r*dt) * (1 - np.sqrt(np.exp((sigma**2)*dt)-1))
  p = 1/2
  
  value = 0
  for i in range(n+1):
    node_prob = combos(n,i) * p**i * (1-p)**(n-i)
    st = s0 * u**i * d**(n-i)
    value += max(st-k,0) * node_prob
  
  return value * np.exp(-r*t)


# Compute prices of American options via Binomial Methods
def binom_a(s0, k, r, t, sigma, n, type_):
    dt = t/n
    c = (np.exp(-r*dt) + np.exp((r+sigma**2)*dt)) / 2
    d = c - np.sqrt(c**2 - 1)
    u = 1/d
    p = (np.exp(r*dt)-d) / (u-d)
    
    # binomial price tree
    stockvalue = np.zeros((n+1, n+1))
    stockvalue[0, 0] = s0
    for i in range(1,n+1):
        stockvalue[i,0] = stockvalue[i-1,0]*u
        for j in range(1,i+1):
            stockvalue[i,j] = stockvalue[i-1,j-1]*d
    
    # option value at final node   
    optionvalue = np.zeros((n+1,n+1))
    for j in range(n+1):
        if type_=="call":
            optionvalue[n,j] = max(0, stockvalue[n,j]-k)
        elif type_=="put":
            optionvalue[n,j] = max(0, k-stockvalue[n,j])
    
    # backward calculation for option price    
    for i in range(n-1,-1,-1):
        for j in range(i+1):
                if type_=="put":
                    optionvalue[i,j] = max(0, k-stockvalue[i,j], \
                                           np.exp(-r*dt)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1]))
                elif type_=="call":
                    optionvalue[i,j] = max(0, stockvalue[i,j]-k, \
                                           np.exp(-r*dt)*(p*optionvalue[i+1,j]+(1-p)*optionvalue[i+1,j+1]))
    
    return optionvalue[0,0]

  
def tri(n, i, j):
    return math.factorial(n) / (math.factorial(n-i-j)*math.factorial(i)*math.factorial(j))

# Compute prices of European call options via Trinomial Method
def trio_e_call(s0, k, t, r, sigma, n):
    dt = t/n
    d = np.exp(-sigma*np.sqrt(3*dt))
    u = 1/d
    m = 1 - d - u
    pd = (r*dt*(1-u) + (r*dt)**2 + (sigma**2)*dt) / ((u-d)*(1-d))
    pu = (r*dt*(1-d) + (r*dt)**2 + (sigma**2)*dt) / ((u-d)*(u-1))
    pm = 1 - pd - pu
    
    value = 0
    for i in range(n+1):
        for j in range(n-i+1):
            node_prob = tri(n, i, j)* pu**i * pm**(n-i-j) * pd**j
            st = s0 * u**i * m**(n-i-j) * d**j
            value += max(st-k,0) * node_prob
    
    return value * np.exp(-r*t)
  

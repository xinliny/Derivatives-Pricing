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
  
  # defined binomial methods parameters (can change)
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

  

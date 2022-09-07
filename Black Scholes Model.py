import scipy.stats
from scipy.stats import norm

# Compute prices of European call options via Black-Scholes Formula
def bs_call(r,sigma,s,T,K):
    d1 = (np.log(s/K) + (r+(sigma**2)/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    call = s*N_d1 - K*np.exp(-r*T)*N_d2
    return call

# Compute prices of European put options via Black-Scholes Formula 
def bs_put(r,sigma,s,T,K):
    d1 = (np.log(s/K) + (r+(sigma**2)/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    N_d1 = norm.cdf(-d1)
    N_d2 = norm.cdf(-d2)
    put = - s*N_d1 + K*np.exp(-r*T)*N_d2
    return put
  
  

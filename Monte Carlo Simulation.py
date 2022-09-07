# Function: generate radom numbers with normal distribution
def normal_box(n,i):
   rv_u = np.random.uniform(0,1,2*n)
   u1 = rv_u[:n]
   u2 = rv_u[n:]
   z1 = np.sqrt(-2*np.log(u1)) * np.cos(2*np.pi*u2)
   z2 = np.sqrt(-2*np.log(u1)) * np.sin(2*np.pi*u2)
   return z1, z2


# Function: compute price of a European call option via Monte Carlo Simulation
def mc_call(s0, k, t, r, sigma):
    n = 10000
    z1 = normal_box(n,10)[0]
    z2 = -z1   # variance-reduction techniques: antithetic variates
    w1 = np.sqrt(t) * z1
    w2 = np.sqrt(t) * z2
    
    st1 = s0 * np.exp(sigma*w1 + (r-(sigma**2)/2)*t)
    st2 = s0 * np.exp(sigma*w2 + (r-(sigma**2)/2)*t)
    q1 = st1-k
    q2 = st2-k
    
    call1 = []
    call2 = []
    for i in range(0,len(q1)):
        p1 = max(q1[i], 0)
        p2 = max(q2[i], 0)
        c1 = p1 * np.exp(-r*t)
        c2 = p2 * np.exp(-r*t)
        call1 = np.append(call1, c1)
        call2 = np.append(call2, c2)
    
    return (np.mean(call1)+np.mean(call2))/2
   

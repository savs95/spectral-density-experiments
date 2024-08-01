import numpy as np
import scipy as sp

class Distribution:
    def __init__(self):
        self.support = []
        self.weights = []
        
        return None
    
    def from_weights(self,support,weights):
        
        self.support = support
        self.weights = weights
        self.distr = np.cumsum(weights)
            
        #assert np.all(weights >= 0), 'distribution must be increasing'

        
    def from_distr(self,support,distr):
        
        self.support = support
        self.weights = np.diff(distr,prepend=0)
        
        assert distr[0] >= 0, 'distribution must start at zero'
        assert np.all(self.weights >= 0), 'distribution must be increasing'
        
        self.distr = distr
        
    def __call__(self,x):
        
        return np.sum(self.weights[self.support <= x])
    
    def get_distr(self):
        """
        get distribution for plotting, etc
        """
        
        return np.hstack([np.nextafter(self.support[0],-np.inf),\
                          self.support,\
                          np.nextafter(self.support[-1],np.inf)]),\
               np.hstack([0,self.distr,self.distr[-1]])

    def __add__(self,other):
        
        full_support = np.hstack([self.support,other.support])
        idx = np.argsort(full_support)
        
        support = full_support[idx]
        weights = np.hstack([self.weights,other.weights])[idx]
        
        out = Distribution()
        out.from_weights(support,weights)#,lower_bd,upper_bd)
        
        return out

    def __truediv__(self,c):
    
        out = Distribution()
            
        out.from_weights(self.support,np.array(self.weights)/c)
        
        return out
    
    def integrate(self,f):
        """
        integrate a function f against density
        """
        
        return np.sum(f(support)*weights)
        
        
def get_GQ_distr(a,b,norm2=1):
    """
    get distribution corresponding Gaussian quadrature
    """
    
    try:
        theta,S = sp.linalg.eigh_tridiagonal(a,b,tol=1e-30)
    except:
        T = np.diag(a) + np.diag(b,-1) + np.diag(b,1)
        theta,S = np.linalg.eigh(T)

    GQ = Distribution()
    GQ.from_weights(theta,S[0]**2*norm2)
    
    return GQ

def get_GQ_upper_bound(GQ,lower_bound,upper_bound):
    """
    get distribution corresponding to upper bounds for Gaussian quadrature
    """
    
    support = np.hstack([lower_bound,GQ.support,upper_bound])
    weights = np.hstack([GQ.weights,0,0])
    
    GQ_ub = Distribution()
    GQ_ub.from_weights(support,weights)
    
    return GQ_ub

def get_GQ_lower_bound(GQ,lower_bound,upper_bound):
    """
    get distribution corresponding to lower bounds for Gaussian quadrature
    """
    
    support = np.hstack([lower_bound,GQ.support,upper_bound])
    weights = np.hstack([0,0,GQ.weights])
    
    GQ_lb = Distribution()
    GQ_lb.from_weights(support,weights)
    
    return GQ_lb

def get_ave_distr(dists):
    
    k = len(dists)
    D = Distribution()

    for Di in dists:
        D = D + Di
    
    return D/k

def max_distribution(x1,y1,x2,y2):
    """
    return maximum distribution function

    Parameters
    ----------
    x1 : (k,) ndarray
    y1 : (k,) ndarray
    x2 : (k,) ndarray
    y2 : (k,) ndarray

    Returns
    -------
    x : (k,) ndarray
    y : (k,) ndarray
    """

    X = np.unique(np.hstack([x1,x2]))
    Y = np.zeros_like(X)

    for i,x in enumerate(X):
        
        if x > x1[-1]:
            y1_candidate = y1[-1]
        elif x < x1[0]:
            y1_candidate = -np.inf
        else:
            y1_candidate = y1[np.argmin(x1<=x)-1]
            
        if x > x2[-1]:
            y2_candidate = y2[-1]
        elif x < x2[0]:
            y2_candidate = -np.inf
        else:
            y2_candidate = y2[np.argmin(x2<=x)-1]
            
        Y[i] = np.max([y1_candidate,y2_candidate])
        
    return X,Y

def min_distribution(x1,y1,x2,y2):
    """
    return maximum distribution function

    Parameters
    ----------
    x1 : (k,) ndarray
    y1 : (k,) ndarray
    x2 : (k,) ndarray
    y2 : (k,) ndarray

    Returns
    -------
    x : (k,) ndarray
    y : (k,) ndarray
    """
    
    X = np.unique(np.hstack([x1,x2]))
    Y = np.zeros_like(X)

    for i,x in enumerate(X):
        
        
        if x > x1[-1]:
            y1_candidate = y1[-1]
        elif x < x1[0]:
            y1_candidate = 0
        else:
            y1_candidate = y1[np.argmin(x1<=x)-1]
            
        if x > x2[-1]:
            y2_candidate = y2[-1]
        elif x < x2[0]:
            y2_candidate = 0
        else:
            y2_candidate = y2[np.argmin(x2<=x)-1]
         
        Y[i] = np.min([y1_candidate,y2_candidate])
        
    return X,Y

def get_max_distr(dists):
    

    k = len(dists)
    D = Distribution()
    D.from_weights(dists[0].support,dists[0].weights)

    for j in range(1,k):
        X,Y = max_distribution(D.support,D.distr,dists[j].support,dists[j].distr)

        D = Distribution()
        D.from_distr(X,Y)
    
    return D

def get_min_distr(dists):
    

    k = len(dists)
    D = Distribution()
    D.from_weights(dists[0].support,dists[0].weights)

    for j in range(1,k):
        X,Y = min_distribution(D.support,D.distr,dists[j].support,dists[j].distr)

        D = Distribution()
        D.from_distr(X,Y)
    
    return D

def add_constant(D,c,lb,ub):
    
    D1 = Distribution()
    distr = D.distr + c
    
    distr[distr>1] = 1
    distr[distr<0] = 0
    
    D1.from_distr(np.hstack([lb,D.support,ub]),np.hstack([0,distr,1]))
    
    return D1

def d_KS(D1,D2):
    """
    return KS distance between two distributions
    
    Parameters
    ----------
    D1 : Distribution
    D2 : Distribution
    
    Returns
    -------
    dKS
    """
    
    _,ymin = min_distribution(D1.support,D1.distr,D2.support,D2.distr)
    _,ymax = max_distribution(D1.support,D1.distr,D2.support,D2.distr)
    
    return np.max(ymax-ymin)

def d_W(D1,D2):
    """
    return Wasserstein-1 distance between two distributions over interval a,b
    
    Parameters
    ----------
    D1 : Distribution
    D2 : Distribution
    
    Returns
    -------
    dW
    """
    
    x,ymin = min_distribution(D1.support,D1.distr,D2.support,D2.distr)
    _,ymax = max_distribution(D1.support,D1.distr,D2.support,D2.distr)
    
    return np.sum(np.diff(x)*(ymax-ymin)[:-1])

def d_Wbd(GQ,lb,ub):
    """
    ""
    return Wasserstein-1 distance between two distributions over interval a,b
    
    Parameters
    ----------
    GQ: Distribution
        gaussian quadrature for mu
    lb: float
        lower bound on support of mu
    ub: float
        upper bound on support of mu
    
    Returns
    -------
    bound for d_W(GQ,mu)
    """
    
    d = GQ.weights
    theta = GQ.support
    k = len(d)
    
    if lb == None:
        lb = theta[0]
    if ub == None:
        ub = theta[-1]

    return (theta[0]-lb)*d[0] + (ub-theta[-1])*d[-1] + (np.sum( np.diff(theta) * np.max(np.vstack([d[1:],d[:-1]]),axis=0) ) if k > 1 else 0)
         
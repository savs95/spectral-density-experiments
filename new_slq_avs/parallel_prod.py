import cmath
import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange
import random
import networkx as nx
from tqdm import tqdm
import pickle
from scipy.stats import wasserstein_distance as w1_dist
import scipy as sp
from multiprocessing import Pool


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

        
        
def get_GQ_distr(a,b,norm2=1):
    """
    get distribution corresponding Gaussian quadrature
    # NEED to add noise here io
    """
    
    try:
        theta,S = sp.linalg.eigh_tridiagonal(a,b,lapack_driver='stemr')
    except:
        #add noise here if it fails to converge
        theta,S = sp.linalg.eigh_tridiagonal(a,b,lapack_driver='stemr')

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
    # Takes in dictionary of distributions and returns the average distribution
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

def sample_spherical(ndim):
    vec = np.random.randn(ndim)
    vec /= np.linalg.norm(vec)
    return vec

def lanczos_new(A,k,reorth=False):
    """
    Lanczos algorithm

    Parameters
    ----------
    A : (n,n) matrix-like
        matrix
    v : (n,) ndarray
        starting vector
    k : int
                maximum iterations
    reoth : bool, default=True
                reorthogonalize or not

    Returns
    -------
    α : (k,) ndarray
        recurrence coefficients
    β : (k,) ndarray
        recurrence coefficients
    """

    n = A.shape[0]
    a = np.zeros(k,dtype=np.float64)
    b = np.zeros(k,dtype=np.float64)
    if reorth:
        Q = np.zeros((n,k+1),dtype=np.float64)

    q = sample_spherical(n)
    if reorth:
        Q[:,0] = q
    for i in range(0,k):
        print("iter",i)
        q__ = np.copy(q)
        q = A@q - b[i-1]*q_ if i>0 else A@q
        q_ = q__

        a[i] = q@q_
        q -= a[i]*q_

        # double Gram-Schmidt reorthogonalization
        if reorth:
            q -= Q@(Q.T@q)
            q -= Q@(Q.T@q)

        b[i] = np.sqrt(q.T@q)
        q /= b[i]

        if reorth:
            Q[:,i+1] = q

    else:
        return (a,b[:k-1])
    
def lanczos_GQ(A,k,reorth=False):
   a,b = lanczos_new(A,k)
   D = get_GQ_distr(a,b)
   print("lanczos_GQ done")
   return D

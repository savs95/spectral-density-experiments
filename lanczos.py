import numpy as np
import scipy as sp


def exact_lanczos(A,reorth=False):
    """
    run Lanczos with reorthogonalization
    
    Input
    -----
    A : entries of diagonal matrix A
    q0 : starting vector
    k : number of iterations
    B : entries of diagonal weights for orthogonalization
    """
    # Assuming A is a square matrix
    n = A.shape[0] 
    k = n
    # Initialize q0 as a random standard Gaussian vector
    q0 = np.random.randn(n)
    norm_q0 = np.linalg.norm(q0)
    # Normalize the vector
    normalized_q0 = q0 / norm_q0
    
    Q = np.zeros((n,k),dtype=A.dtype)
    a = np.zeros(k,dtype=A.dtype)
    b = np.zeros(k-1,dtype=A.dtype)
    
    Q[:,0] = q0 / np.sqrt(q0.T@q0)
    
    for i in range(1,k+1):
        # expand Krylov space
        qi = A@Q[:,i-1] - b[i-2]*Q[:,i-2] if i>1 else A@Q[:,i-1]
        
        a[i-1] = qi.T@Q[:,i-1]
        qi -= a[i-1]*Q[:,i-1]
        
        if reorth:
            qi -= Q@(Q.T@qi) # regular GS
            #for j in range(i-1): # modified GS (a bit too slow)
            #    qi -= (qi.T@Q[:,j])*Q[:,j]
            
        if i < k:
            b[i-1] = np.sqrt(qi.T@qi)
            Q[:,i] = qi / b[i-1]
                
    return Q,(a,b)
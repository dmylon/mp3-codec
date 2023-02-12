import numpy as np

def frameDCT(Y):

    # defining basic parameters
    N,M = Y.shape
    F = np.zeros((N,N))
    k = np.arange(0,N)
    l = np.arange(0,N)

    # creating meshgrids to account for every possible combination of k,l
    K,L = np.meshgrid(k,l)
    F[: ,:] = np.sqrt(2/N)*np.cos(np.pi/N*K*(L+1/2))
    F = np.transpose(F)
    F[0,:] = np.sqrt(1/N)
    
    # producing Ytrans as a matrix multiplication of F,Y
    Ytrans = np.matmul(F,Y)
    c = np.reshape(Ytrans,(M*N,1),order='F')

    return c.squeeze()


def iframeDCT(c,N,M):
    
    # initializations and parameters
    Ytrans = np.reshape(c,(N,M),order='F')
    F = np.zeros((N,N))
    k = np.arange(0,N)
    l = np.arange(0,N)
    K,L = np.meshgrid(k,l)
    F[: ,:] = np.sqrt(2/N)*np.cos(np.pi/N*K*(L+1/2))
    F = np.transpose(F)
    F[0,:] = np.sqrt(1/N)

    # inverting the above function
    Yh = np.matmul(np.linalg.inv(F),Ytrans)
    
    return Yh



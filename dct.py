import numpy as np

def frameDCT(Y):

    N,M = Y.shape
    F = np.zeros((N,N))
    k = np.arange(0,N)
    l = np.arange(0,N)
    K,L = np.meshgrid(k,l)
    F[: ,:] = np.sqrt(2/N)*np.cos(np.pi/N*K*(L+1/2))
    F = np.transpose(F)
    F[0,:] = np.sqrt(1/N)
    
    Ytrans = np.matmul(F,Y)
    c = np.reshape(Ytrans,(M*N,1),order='F')

    return c

def iframeDCT(c,N,M):
    
    Ytrans = np.reshape(c,(N,M),order='F')

    F = np.zeros((N,N))
    k = np.arange(0,N)
    l = np.arange(0,N)
    K,L = np.meshgrid(k,l)
    F[: ,:] = np.sqrt(2/N)*np.cos(np.pi/N*K*(L+1/2))
    F = np.transpose(F)
    F[0,:] = np.sqrt(1/N)

    Yh = np.matmul(np.linalg.inv(F),Ytrans)
    
    return Yh



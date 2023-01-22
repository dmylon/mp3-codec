import numpy as np

def DCTpower(c):
    
    P = 10*np.log10(np.square(np.absolute(c)))
    return P


def Dksparse(Kmax):

    D = np.zeros((Kmax,Kmax))

    for k in range(3,Kmax):
        
        if k > 2 and k < 282:
            Dk = [2,2]
        elif k >= 282 and k < 570:
            Dk = [2,13]
        else:
            Dk = [2,27]
        
        j = np.arange(0,Kmax)
        idx = np.where((j >= Dk[0]) & (j <= Dk[1]))
        D[k,idx] = 1
    
    return D


def STinit(c, D):

    P = DCTpower(c)
    Kmax,_ = D.shape
    
    ST = np.empty(0)
    for k in np.arange(3,Kmax-26):
      if P[k] > P[k+1] and P[k] > P[k-1]:
        Dk = np.where(D[k,:])
        if P[k] > np.max(np.r_[P[k-Dk],P[k+Dk]]) + 7:
           ST = np.r_[ST,k]
        
    return ST


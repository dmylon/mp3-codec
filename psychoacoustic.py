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


def MaskPower(c, ST):
  
  P  = DCTpower(c)
  PM = 10*np.log10(10**(0.1*(P[ST])) + 10**(0.1*(P[ST-1])) + 10**(0.1*(P[ST+1])))
  return PM


def Hz2Barks(f):
  
  z = 13*np.arctan(0.00076*f) + 3.5*np.arctan((f/7500)**2)
  return z


def STreduction(ST, c, Tq):
  
  fs = 44100
  MN = c.shape[0]
  PM = MaskPower(c, ST)

  ind = np.where(PM >= Tq[ST])
  STr, PMr = ST[ind], PM[ind]
  f = ST*fs/(2*MN)
  z = Hz2Barks(f)

  for i in np.arange(z.shape[0]-1):
    if np.absolute(z[i] - z[i+1]) < 0.5:
      r = i if PM[i] < PM[i+1] else i+1
      ind = np.where(PM[r]==PMr)[0]
      STr = np.delete(STr, ind)
      PMr = np.delete(PMr, ind)
  return STr, PMr


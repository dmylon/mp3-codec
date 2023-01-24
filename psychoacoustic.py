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
        
    return ST.astype(np.int32)


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
    
  while np.min(z[1:]-z[:-1]) < 0.5:
    i = np.argmin(z[1:]-z[:-1])
    if PMr[i+1] < PMr[i]:
      i = i + 1 
    z, STr, PMr = np.delete(z, i), np.delete(STr, i), np.delete(PMr, i)
    

  return STr, PMr


def SpreadFunc(ST, PM, Kmax):

  fs = 44100  
  SF = np.zeros((Kmax+1,len(ST)))


  for i in range(Kmax+1):
    for k in range(len(ST)):
      fi = fs/(2*(Kmax+1))*i
      fk = fs/(2*(Kmax+1))*k
      Dz = Hz2Barks(fi) - Hz2Barks(fk)

      if Dz >= -3 and Dz < -1:
        SF[i,k] = 17*Dz - 0.4*PM[k] + 11
      elif Dz >= -1 and Dz < 0:
        SF[i,k] = (0.4*PM[k] + 6)*Dz
      elif Dz >= 0 and Dz < 1:
        SF[i,k] = -17*Dz
      elif Dz >= 1 and Dz < 8:
        SF[i,k] = (0.15*PM[k] -17)*Dz - 0.15*PM[k]
      
  return SF


def Masking_Thresholds(ST, PM, Kmax):

  SF = SpreadFunc(ST,PM,Kmax)
  fs = 44100
  Ti = np.zeros((Kmax+1,len(ST)))

  for i in range(Kmax+1):
    for k in range(len(ST)):
      fi = fs/(2*(Kmax+1))*i
      fk = fs/(2*(Kmax+1))*k
      Ti[i, k] = PM[k] - 0.275*Hz2Barks(fk) + SF[i, k] - 6.025  

  return Ti


def Global_Masking_Thresholds(Ti, Tq):
  Tg = 10*np.log10(10**(0.1*Tq) +np.sum(10**(0.1*Ti),1))
  return Tg

def psycho(c,D):
  
  Kmax = len(c) - 1
  Tq = np.load("Tq.npy", allow_pickle=True).squeeze()
  ST = STinit(c,D)
  PM = MaskPower(c,ST)
  STr,PMr = STreduction(ST,c,Tq)
  Ti = Masking_Thresholds(STr,PMr,Kmax)
  Tg = Global_Masking_Thresholds(Ti,Tq)

  return Tg

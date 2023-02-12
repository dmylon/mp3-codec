import numpy as np

def DCTpower(c):
    
    # calculating power in db
    P = 10*np.log10(np.square(np.absolute(c)))
    return P


def Dksparse(Kmax):

    D = np.zeros((Kmax,Kmax))

    # creating the Dksparse matrix accoring to the given formulas
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

    # calling above function
    P = DCTpower(c)
    Kmax,_ = D.shape
    
    ST = np.empty(0)
    # calculating ST according to the given formula
    for k in np.arange(3,Kmax-26):
      if P[k] > P[k+1] and P[k] > P[k-1]:
        Dk = np.where(D[k,:])
        if P[k] > np.max(np.r_[P[k-Dk],P[k+Dk]]) + 7:
           ST = np.r_[ST,k]
        
    return ST.astype(np.int32)


def MaskPower(c, ST):
  
  P  = DCTpower(c)
  # calculating mask power according to formula
  PM = 10*np.log10(10**(0.1*(P[ST])) + 10**(0.1*(P[ST-1])) + 10**(0.1*(P[ST+1])))
  return PM


def Hz2Barks(f):
  
  # turning Hz into Barks
  z = 13*np.arctan(0.00076*f) + 3.5*np.arctan((f/7500)**2)
  return z


def STreduction(ST, c, Tq):
  
  # parameter defining and calling functions above
  fs = 44100
  MN = c.shape[0]
  PM = MaskPower(c, ST)

  # finding positions and values where PM >= Tq[ST]
  ind = np.where(PM >= Tq[ST])
  STr, PMr = ST[ind], PM[ind]
  f = STr*fs/(2*MN)
  z = Hz2Barks(f)
    
  # while the minimum distance between two consecutive barks is less than 0.5
  while np.min(z[1:]-z[:-1]) < 0.5:
    i = np.argmin(z[1:]-z[:-1])
    if PMr[i+1] < PMr[i]:
      i = i + 1 
    z, STr, PMr = np.delete(z, i), np.delete(STr, i), np.delete(PMr, i)
    

  return STr, PMr


def SpreadFunc(ST, PM, Kmax):
  
  # parameter defining and matrix initialization
  fs = 44100  
  SF = np.zeros((Kmax+1,len(ST)))
  i = np.arange(Kmax+1)

  fi = fs/(2*(Kmax+1))*i
  for k in np.arange(len(ST)):
    fk = fs/(2*(Kmax+1))*k
    # calculating distance in barks
    Dz = Hz2Barks(fi) - Hz2Barks(fk)
    
    # filling SF according to the given cases
    case1 = np.where((Dz>=-3)&(Dz<-1))
    case2 = np.where((Dz>=-1)&(Dz<0))
    case3 = np.where((Dz>=0)&(Dz<1))
    case4 = np.where((Dz>=1)&(Dz<8))
    SF[case1 ,k] = 17*Dz[case1] - 0.4*PM[k] + 11
    SF[case2, k] = (0.4*PM[k] + 6)*Dz[case2]
    SF[case3, k] = -17*Dz[case3]
    SF[case4, k] = (0.15*PM[k] - 17)*Dz[case4] - 0.15*PM[k]
  return SF


def Masking_Thresholds(ST, PM, Kmax):

  # calling previous function and defining parameters
  SF = SpreadFunc(ST,PM,Kmax)
  fs = 44100
  Ti = np.zeros((Kmax+1,len(ST)))
  i = np.arange(Kmax+1)
  for k in np.arange(len(ST)):
    fk = fs/(2*(Kmax+1))*k
    Ti[i,k] = PM[k] - 0.275*Hz2Barks(fk) + SF[i,k] - 6.025     
  return Ti


def Global_Masking_Thresholds(Ti, Tq):
  
  # returning Tg according to the given formula
  Tg = 10*np.log10(10**(0.1*Tq) +np.sum(10**(0.1*Ti),1))
  return Tg

def psycho(c,D):
  
  # defining Kmax and calling the previous functions one after another
  Kmax = len(c) - 1
  Tq = np.load("Tq.npy", allow_pickle=True).squeeze()
  ST = STinit(c,D)
  PM = MaskPower(c,ST)
  STr,PMr = STreduction(ST,c,Tq)
  Ti = Masking_Thresholds(STr,PMr,Kmax)
  Tg = Global_Masking_Thresholds(Ti,Tq)

  return Tg

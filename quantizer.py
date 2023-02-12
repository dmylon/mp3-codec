import numpy as np
def critical_bands(K):

    # defining the list of boundaries according to the critical bands bandwidth
    cbands = np.array([-np.inf,100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000
        ,15500,np.inf])

    fs = 44100
    cb = np.array([0 for _ in range(K)])
    k = np.arange(K)
    f = fs*k/(2*K)
    for i in np.arange(len(cbands)-1):
        # finding in which range the f is between
        idx = np.where((f>=cbands[i])&(f<cbands[i+1]))
        cb[idx] = i + 1 
    return cb      


def DCT_band_scale(c):

    # parameter defining and array initialization
    cb = critical_bands(len(c))
    sc = np.array([0.0 for _ in range(np.max(cb))])
    cs = np.array([0.0 for _ in range(len(cb))])

    for i in range(np.max(cb)):
        indexes = np.where(cb == i+1)
        sc[i] =  np.max(np.absolute(c[indexes])**(3/4))

    # applying the given formula
    cs = np.sign(c)*np.absolute(c)**(3/4) / sc[cb-1]
    
    return cs,sc


def quantizer(x, b):

    # defing decision level distance as 2/2^b
    wb = 2 / (2**(b))
    d = np.array([-np.inf])
    symb_index = np.array([0 for _ in range(len(x))])

    # creating decision levels
    for i in range(-(2**(b-1)-1),(2**(b-1)-1)+1):
        if i == 0:
            continue

        d = np.append(d,wb*i)

    d = np.append(d,np.inf)

    # creating symbols
    for i in np.arange(len(d)-1):
        idx = np.where((x>=d[i])&(x<d[i+1]))
        symb_index[idx] = i - 2**(b-1) + 1
    
    return symb_index



def dequantizer(symb_index, b):

    # defing decision level distance as 2/2^b
    wb = 2 / (2**(b))
    d = np.array([-1])
    xh = np.array([0.0 for _ in range(len(symb_index))])

    # creating decision levels
    for i in range(-(2**(b-1)-1),(2**(b-1)-1)+1):
        if i == 0:
            continue
        d = np.append(d,wb*i)
    
    d = np.append(d,1)
    # dequantizing symbols to values at the center of each interval
    for i in np.arange(len(d)-1):
        idx = np.where(symb_index==i - 2**(b-1) + 1)
        xh[idx] = (d[i+1]+d[i])/2

    return xh


def all_bands_quantizer(c,Tg):
    
    # calling previous functions
    cs,sc = DCT_band_scale(c)
    cb = critical_bands(len(c))
    
    # removing Nan values
    excl = np.where(~np.isnan(Tg))
    Tg = Tg[excl]

    # initial number of bits
    B = 1
    while True:
        
        # quantizing/dequantizing
        symb_index = quantizer(cs,B)
        csHat = dequantizer(symb_index,B)
        
        # producing cHat with the given formula
        cHat = np.sign(csHat)*np.array((np.abs(csHat)*sc[cb-1])**(4/3))
        
        # calculating absolute error
        eb = np.absolute(c - cHat)
        Pb = 10*np.log10(eb**2)        
        Pb = Pb[excl]

        # loop-break condition
        if np.count_nonzero(Pb >= Tg) == 0:
            break

        B = B + 1

    return symb_index,sc,B

def all_bands_dequantizer(symb_index, B, SF):

    # calling all functions one after another
    cb = critical_bands(len(symb_index))
    csHat = dequantizer(symb_index,B)
    cHat = np.sign(csHat) * np.array((np.abs(csHat)*SF[cb-1])**(4/3))
    return cHat
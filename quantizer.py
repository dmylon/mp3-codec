import numpy as np
def critical_bands(K):

    cbands = np.array([-np.inf,100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000
        ,15500,np.inf])

    fs = 44100
    cb = np.array([0 for _ in range(K)])
    for k in range(K):
        
        f = fs*k/(2*K)
        #print(f)

        i = 0 
        while True:
            if f >= cbands[i] and f <= cbands[i+1]:
                cb[k] = i + 1
                break
            i = i + 1 

    return cb      


def DCT_band_scale(c):

    cb = critical_bands(len(c))
    sc = np.array([0.0 for _ in range(np.max(cb))])
    cs = np.array([0.0 for _ in range(len(cb))])

    for i in range(np.max(cb)):
        indexes = np.where(cb == i+1)
        sc[i] =  np.max(np.absolute(c[indexes])**(3/4))

    cs = np.sign(c)*np.absolute(c)**(3/4) / sc[cb-1]
    
    return cs,sc


def quantizer(x, b):

    wb = 2 / 2**(b)
    d = np.array([-1])
    symb_index = np.array([0 for _ in range(len(x))])

    for i in range(-(2**(b-1)-1),(2**(b-1)-1)+1):
        if i == 0:
            continue

        d = np.append(d,wb*i)

    d = np.append(d,1)

    for k in range(len(x)):
        i = 0 
        while True:
            
            if x[k] >= d[i] and x[k] <= d[i+1]:
        
                symb_index[k] = i - 2**(b-1) + 1
                break
            
            i = i + 1

    return symb_index
        
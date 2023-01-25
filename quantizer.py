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
        
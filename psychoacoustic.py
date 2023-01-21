import numpy as np

def DCTpower(c):
    
    P = 10*np.log10(np.square(np.absolute(c)))
    return P
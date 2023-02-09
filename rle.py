import numpy as np

def RLE(symb_index,K):

    run_symbols = np.zeros((0,2))

    zero_count = 0
    for i in range(K):
        
        if symb_index[i] == 0:
            zero_count = zero_count + 1
            continue
        
        element = np.array([[symb_index[i],zero_count] for _ in range(1)])
        run_symbols = np.r_[run_symbols,element]
        #run_symbols = np.append(run_symbols,[symb_index[i],zero_count])
        zero_count = 0

    run_symbols = run_symbols.astype(np.int64)
    return run_symbols


def iRLE(run_symbols,K):
    
    R,_ = run_symbols.shape
    symb_index = np.zeros(K)

    k = 0
    for i in range(R):
        zeros_number = run_symbols[i,1]
        k = k + zeros_number
        symb_index[k] = run_symbols[i,0]
        k = k + 1
    
    return symb_index.astype(np.int64)
        
        


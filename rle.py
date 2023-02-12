import numpy as np

def RLE(symb_index,K):

    # matrix R X 2 to store the final symbols
    run_symbols = np.zeros((0,2))

    zero_count = 0
    for i in range(K):
        
        # counting a zero and going for the next loop
        if symb_index[i] == 0:
            zero_count = zero_count + 1
            continue
        
        # store a symbol with the zero count value and merge with the initial array
        element = np.array([[symb_index[i],zero_count] for _ in range(1)])
        run_symbols = np.r_[run_symbols,element]
        zero_count = 0

    run_symbols = run_symbols.astype(np.int64)
    return run_symbols


def iRLE(run_symbols,K):
    
    # extracting R and defining vector for symbols storing
    R,_ = run_symbols.shape
    symb_index = np.zeros(K)

    k = 0
    for i in range(R):
        # moving cursor for zeros_count positions in the vector
        zeros_number = run_symbols[i,1]
        k = k + zeros_number
        symb_index[k] = run_symbols[i,0]
        k = k + 1
    
    return symb_index.astype(np.int64)
        
        


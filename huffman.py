import numpy as np

def get_huff_symbols(tree):

    left = tree[0]
    right = tree[1]

    if type(left) is np.ndarray:
        left_symbols = np.array([[left, ""]],dtype=object)
    else: 
        left_symbols = get_huff_symbols(left)
    if type(right) is np.ndarray:
        right_symbols = np.array([[right, ""]],dtype=object)
    else:
        right_symbols = get_huff_symbols(right)

    left_symbols[:,1] = "0" + left_symbols[:,1]
    right_symbols[:,1] = "1" + right_symbols[:,1]
    huff_symbols = np.r_[left_symbols,right_symbols]


    return huff_symbols


def huff(run_symbols):

    unique_sumbols, counts = np.unique(run_symbols, return_counts=True, axis=0)
    prob = counts/np.sum(counts)
    frame_symbol_prob = np.c_[unique_sumbols, prob]
    order = frame_symbol_prob[:,2].argsort()
    
    N = np.array([(frame_symbol_prob[i,:2],frame_symbol_prob[i,2]) for i in order], dtype=object)

    while len(N) > 1:
        s1, p1 = N[0]
        s2, p2 = N[1]
        N = N[2:]
        new_s = (s1,s2)
        new_p = p1+p2
        N = np.r_[N,np.array([(new_s, new_p)], dtype=object)]
        N = N[N[:,1].argsort()]

    huff_symbols = get_huff_symbols(N[0][0])

    frame_stream = ""
    s = np.array([i for i in huff_symbols[:,0]]).astype(np.int64)
    for symbol in run_symbols:
        idx = np.where((s == symbol).all(axis=1))[0][0]
        frame_stream = frame_stream + huff_symbols[idx,1]

    return frame_stream, frame_symbol_prob

def ihuff(frame_stream, frame_symbol_prob):
    
    order = frame_symbol_prob[:,2].argsort()
    
    N = np.array([(frame_symbol_prob[i,:2],frame_symbol_prob[i,2]) for i in order], dtype=object)

    while len(N) > 1:
        s1, p1 = N[0]
        s2, p2 = N[1]
        N = N[2:]
        new_s = (s1,s2)
        new_p = p1+p2
        N = np.r_[N,np.array([(new_s, new_p)], dtype=object)]
        # probs = np.array([n[1] for n in N])
        N = N[N[:,1].argsort()]

    huff_symbols = get_huff_symbols(N[0][0])

    s = np.array([i for i in huff_symbols[:,1]]).astype(str)
    bit_stream = ""
    run_symbols = np.zeros((0,2))
    for bit in frame_stream:
        bit_stream += bit

        idx = np.where(bit_stream == s)

        if len(idx[0]) != 0:
            symbol = huff_symbols[idx,0]
            run_symbols = np.r_[run_symbols,np.array([symbol[0][0]])]
            bit_stream = ""

    return run_symbols.astype(np.int64)

    
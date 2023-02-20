import numpy as np

def get_huff_symbols(tree):
    '''
    recursively calculate each symbol's huffman encoded word
    Input: huffman tree
    Output: array containing each symbol in tree along with matching word in bits
    '''

    left = tree[0]
    right = tree[1]

    if type(left) is np.ndarray:
        # if left node is a leaf
        left_symbols = np.array([[left, ""]],dtype=object)
    else: 
        left_symbols = get_huff_symbols(left)
    if type(right) is np.ndarray:
        # if right node is a leaf
        right_symbols = np.array([[right, ""]],dtype=object)
    else:
        right_symbols = get_huff_symbols(right)

    # add 0 to each symbol's representation in left sub-tree
    left_symbols[:,1] = "0" + left_symbols[:,1]
    # add 1 to each symbol's representation in right sub-tree
    right_symbols[:,1] = "1" + right_symbols[:,1]
    huff_symbols = np.r_[left_symbols,right_symbols]


    return huff_symbols


def huff(run_symbols):

    unique_sumbols, counts = np.unique(run_symbols, return_counts=True, axis=0)
    # approximate probability density of each symbol 
    prob = counts/np.sum(counts)
    frame_symbol_prob = np.c_[unique_sumbols, prob]
    # sort by probability in ascending order
    order = frame_symbol_prob[:,2].argsort()
    
    # create node representation of each symbol
    N = np.array([(frame_symbol_prob[i,:2],frame_symbol_prob[i,2]) for i in order], dtype=object)

    while len(N) > 1:
        # join the two nodes with the lowest probability value
        s1, p1 = N[0]
        s2, p2 = N[1]
        N = N[2:]
        new_s = (s1,s2)
        new_p = p1+p2
        
        # store newly created sub-tree in N, retain sorting
        N = np.r_[N,np.array([(new_s, new_p)], dtype=object)]
        N = N[N[:,1].argsort()]

    # get huffman representation of each symbol, based on the tree that was created
    huff_symbols = get_huff_symbols(N.squeeze()[0])

    # create frame stream 
    frame_stream = ""
    s = np.array([i for i in huff_symbols[:,0]]).astype(np.int64)
    for symbol in run_symbols:
        idx = np.where((s == symbol).all(axis=1))[0][0]
        frame_stream = frame_stream + huff_symbols[idx,1]

    return frame_stream, frame_symbol_prob

def ihuff(frame_stream, frame_symbol_prob):

    # sort by probability in ascending order
    order = frame_symbol_prob[:,2].argsort()
    
    # create node representation of each symbol
    N = np.array([(frame_symbol_prob[i,:2],frame_symbol_prob[i,2]) for i in order], dtype=object)

    while len(N) > 1:
        # join the two nodes with the lowest probability value
        s1, p1 = N[0]
        s2, p2 = N[1]
        N = N[2:]
        new_s = (s1,s2)
        new_p = p1+p2

        # store newly created sub-tree in N, retain sorting
        N = np.r_[N,np.array([(new_s, new_p)], dtype=object)]
        N = N[N[:,1].argsort()]

    # get huffman representation of each symbol, based on the tree that was created
    huff_symbols = get_huff_symbols(N.squeeze()[0])

    # decode huffman bitstream
    s = np.array([i for i in huff_symbols[:,1]]).astype(str)
    bit_stream = ""
    run_symbols = np.zeros((0,2))
    for bit in frame_stream:
        bit_stream += bit

        idx = np.where(bit_stream == s)

        if len(idx[0]) != 0:
            # if bitstream matches to a symbol
            symbol = huff_symbols[idx,0]
            run_symbols = np.r_[run_symbols,np.array([symbol[0][0]])]
            bit_stream = ""\
        # else continue adding bits until bitstream matches to a symbol

    return run_symbols.astype(np.int64)

    
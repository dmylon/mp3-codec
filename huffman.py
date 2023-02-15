import numpy as np

def huff(run_symbols):

    unique_sumbols, counts = np.unique(run_symbols, return_counts=True, axis=0)
    prob = counts/np.sum(counts)
    frame_symbol_prob = np.c_[unique_sumbols, prob]
    frame_symbol_prob = frame_symbol_prob[frame_symbol_prob[:,2].argsort()]
    
    frame_stream = None
    return frame_stream, frame_symbol_prob

        


    
from subband import *
from dct import *
from psychoacoustic import *
from quantizer import * 
from rle import *
from huffman import *


def MP3codec(wavin, h, M, N):
    total_stream, total_frame_symbol_prob, stream_breakpoints, frame_symbol_prob_breakpoints, total_sc, total_B = \
        MP3cod(wavin,h,M,N)

    xhat = MP3decod(total_stream, total_frame_symbol_prob, stream_breakpoints, frame_symbol_prob_breakpoints, total_sc, total_B, h, M, N)

    return xhat, total_stream

def MP3cod(wavin, h, M, N):
    
    Ytot = coder0(wavin,h,M,N)
    
    D = Dksparse(M*N - 1)
    i = 0
    total_stream = ""
    total_frame_symbol_prob = np.zeros((0,3))
    stream_breakpoints = np.zeros(1)
    frame_symbol_prob_breakpoints = np.zeros(1)
    total_sc = np.zeros((0,max(critical_bands(M*N))))
    total_B = np.zeros(0)
    while (i+1)*N <= Ytot.shape[0]:
        Y = Ytot[i*N:(i+1)*N, :]
        c = frameDCT(Y)
        Tg = psycho(c,D)
        symb_index,sc,B = all_bands_quantizer(c, Tg)
        total_sc = np.r_[total_sc, np.array([sc])]
        total_B = np.append(total_B, B)
        # run length encoding
        run_symbols = RLE(symb_index,M*N) 
        
        frame_stream, frame_symbol_prob = huff(run_symbols)
        total_stream += frame_stream
        stream_breakpoints = np.append(stream_breakpoints, len(total_stream))
        total_frame_symbol_prob = np.r_[total_frame_symbol_prob, frame_symbol_prob]
        frame_symbol_prob_breakpoints = np.append(frame_symbol_prob_breakpoints, total_frame_symbol_prob.shape[0])
        print("Coding buffer {0}/{1}, stream length in bits: {2}".format(i+1, Ytot.shape[0]//N,len(total_stream)))
        i += 1

    return total_stream, total_frame_symbol_prob, stream_breakpoints.astype(np.int64), frame_symbol_prob_breakpoints.astype(np.int64), total_sc, total_B.astype(np.int64)

def MP3decod(total_stream, total_frame_symbol_prob, stream_breakpoints, frame_symbol_prob_breakpoints, total_sc, total_B, h, M, N):
    
    Ytot = np.empty((0,M))

    for i in np.arange(len(stream_breakpoints)-1):
        frame_stream = total_stream[stream_breakpoints[i]:stream_breakpoints[i+1]]
        frame_symbol_prob = total_frame_symbol_prob[frame_symbol_prob_breakpoints[i]:frame_symbol_prob_breakpoints[i+1],:]
        sc = total_sc[i,:]
        B = total_B[i]

        new_run_symbols = ihuff(frame_stream, frame_symbol_prob)
        symb_new = iRLE(new_run_symbols,M*N)
        xh = all_bands_dequantizer(symb_new, B, sc)
        Y = iframeDCT(xh, N, M)
        
        Ytot = np.r_[Ytot,Y]
        print("Decoding buffer {0}/{1}".format(i+1, len(stream_breakpoints) - 1))

    xhat = decoder0(Ytot, h, M, N)

    return xhat
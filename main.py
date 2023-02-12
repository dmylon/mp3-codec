from scipy.io import wavfile
import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from subband import *
from dct import *
from psychoacoustic import *
from quantizer import *
from rle import *

# loading the coefficients
data = np.load("h.npy", allow_pickle=True).tolist()
h = data['h'].squeeze()

# defining basic parameters
M = 32
fs = 44100
N = 36

# calculating analysis/synthesis filters matrices H, G
H = make_mp3_analysisfb(h, M)
G = make_mp3_synthesisfb(h, M)

# plotting analysis results
plot_frequency(H,fs)
fs, wavin = wavfile.read('myfile.wav')
xhat,Ytot = codec0(wavin,h,M,N)

# writing output file
wavfile.write('output.wav', fs, xhat.astype(np.int16))

# typecasting input and output files
wavin = wavin.astype(np.int64)
xhat = xhat.astype(np.int64)

# syncing input and output signals
L = H.shape[0]
lag = L-M
tmp_wavin, tmp_xhat = wavin[lag:], xhat[:-lag]

# calculating MSE
e = tmp_wavin - tmp_xhat 
mse = np.mean(np.square(e))

# calculate SNR in db
SNR = np.mean(np.square(wavin[lag:])) / mse
SNRdb = 10*np.log10(SNR)
print("SNR = {:.2f}".format(SNRdb))

# c = frameDCT(Ytot[:N,:])
# Yh = iframeDCT(c,N,M)
# P = DCTpower(c)
# D = Dksparse(M*N-1)

# Tg = psycho(c,D)
# Tq = np.load("Tq.npy", allow_pickle=True).squeeze()

# symb_index,sc,B = all_bands_quantizer(c,Tg)

# run_symbols = RLE(symb_index,M*N)
# symb_new = iRLE(run_symbols,M*N)

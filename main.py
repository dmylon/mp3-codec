from scipy.io import wavfile
import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from subband import *
from dct import *
from psychoacoustic import *

# calling the function
data = np.load("h.npy", allow_pickle=True).tolist()
h = data['h'].squeeze()
M = 32
fs = 44100
H = make_mp3_analysisfb(h, M)
G = make_mp3_synthesisfb(h, M)
#plot_frequency(H,fs)

N = 36
fs, wavin = wavfile.read('myfile.wav')
xhat,Ytot = codec0(wavin,h,M,N)
wavfile.write('output.wav', fs, xhat.astype(np.int16))

wavin = wavin.astype(np.int64)
xhat = xhat.astype(np.int64)
L = H.shape[0]
lag = L-M
tmp_wavin, tmp_xhat = wavin[lag:], xhat[:-lag]
e = tmp_wavin - tmp_xhat 
mse = np.mean(np.square(e))

SNR = np.mean(np.square(wavin[lag:])) / mse
SNRdb = 10*np.log10(SNR)

print("SNR = ",SNR)

c = frameDCT(Ytot[0:N,:])
Yh = iframeDCT(c,N,M)
P = DCTpower(c)


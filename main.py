from scipy.io import wavfile
import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from subband import *
from dct import *
from psychoacoustic import *
from quantizer import *

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

print("SNR = {:.2f}".format(SNR))

c = frameDCT(Ytot[0:N,:])
Yh = iframeDCT(c,N,M)
P = DCTpower(c)
D = Dksparse(M*N-1)

Tg = psycho(c,D)
Tq = np.load("Tq.npy", allow_pickle=True).squeeze()

# T1 = np.where(np.isnan(Tq))
# T1c = np.count_nonzero(T1)
# T2 = np.where(np.isnan(Tg))
# T2c = np.count_nonzero(T2)
# print("eimai alogo")
# print(Tg.shape)

# cs,sc = DCT_band_scale(c)
# b = 0
# for i in np.arange(1,11):
#     b = b + 1
#     symb_index = quantizer(cs, b)
#     cshat = dequantizer(symb_index, b)
#     e = cs - cshat
#     mse = np.mean(np.square(e))

#     SNR = 10*np.log10(np.mean(np.square(cshat)) / mse)
#     print("SNR = {0}, b = {1}".format(SNR,b))

symb_index,sc,B = all_bands_quantizer(c,Tg)
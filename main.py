from scipy.io import wavfile
import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb

from codec import *

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
xhat, total_stream = MP3codec(wavin, h, M, N)
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
print("Compression rate: {:.3f}".format(len(wavin)*16/len(total_stream)))

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from mp3 import make_mp3_analysisfb
from mp3 import make_mp3_synthesisfb
from nothing import donothing,idonothing
from frame import frame_sub_analysis,frame_sub_synthesis
import wave


def plot_frequency(H,fs):

    vals = np.zeros(H.shape)

    for i in np.arange(H.shape[1]):
        freq, Hf = signal.freqz(H[:,i],fs=fs)
        Hfabs = np.absolute(Hf)
        vals[:,i] = 10 *  np.log10(Hfabs*Hfabs)

    plt.figure()
    plt.plot(freq,vals)
    plt.xlabel("Hz")
    plt.ylabel("dB")
    plt.title("Μέτρο των συναρτήσεων μεταφοράς των φίλτρων στη συχνοτητα f")

    z = 13*np.arctan(0.00076*freq) + 3.5*np.arctan((freq/7500)**2)
    
    plt.figure()
    plt.plot(z,vals)
    plt.xlabel("Barks")
    plt.ylabel("dB")
    plt.title("Μέτρο των συναρτήσεων μεταφοράς των φίλτρων στη συχνοτητα z")
    plt.show()


def codec0(wavin, h, M, N):

    H = make_mp3_analysisfb(h, M)
    G = make_mp3_synthesisfb(h,M)

    L,_ = H.shape
    xbuffsize, ybuffsize = M*N, N
    i = 0
    Ytot = np.empty((0,M))

    while (i+1)*xbuffsize + L - M <= wavin.shape[0]:
        if (i+1)*xbuffsize + L - M == wavin.shape[0]:
            xbuff = np.r_[wavin[i*xbuffsize:(i+1)*xbuffsize],np.zeros(L-M)]
        else:
            xbuff = wavin[i*xbuffsize:(i+1)*xbuffsize + L - M]
        Y = frame_sub_analysis(xbuff,H,N)        
        Yc = donothing(Y)
        Ytot = np.r_[Ytot,Yc]
        i = i + 1
        
    i = 0
    Yhtot = np.empty((0,M))
    while (i+1)*ybuffsize <= Ytot.shape[0]:
        Yc = Ytot[i*ybuffsize:(i+1)*ybuffsize, :]
        Yh = idonothing(Yc)
        Yhtot = np.r_[Yhtot,Yh]
        i = i + 1
    
    i = 0
    xhat = np.empty(0)
    while (i+1)*ybuffsize + L//M - 1 <= Ytot.shape[0]:
        if (i+1)*ybuffsize + L//M - 1 == Ytot.shape[0]:
            ybuff = np.r_[Yhtot[i*ybuffsize:(i+1)*ybuffsize, :],np.zeros((L//M - 1,M))]
        else:
            ybuff = Yhtot[i*ybuffsize:(i+1)*ybuffsize + L//M - 1, :]
        xsynth = frame_sub_synthesis(ybuff,G)
        xhat = np.r_[xhat,xsynth]
        i = i + 1
     
    return xhat,Ytot

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

min_mse, min_i = np.mean(np.square(wavin-xhat)), 0
for i in np.arange(1,1000):
  tmp_wavin, tmp_xhat = wavin[i:], xhat[:-i]
  e = tmp_wavin - tmp_xhat 
  mse = np.mean(np.square(e))
  if mse < min_mse:
    min_mse, min_i = mse, i

SNR = np.mean(np.square(wavin[min_i:])) / min_mse
SNRdb = 10*np.log10(SNR)

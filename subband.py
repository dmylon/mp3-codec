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
    buffsize = M*(N-1)+L
    i = 0
    Ytot = np.empty((0,M))
    xhat = np.empty(0)
    yhbuff = np.empty((0,M))
    lines_encoded = 0
    ybuffsize = (N-1) + L//M

    while (i+1)*buffsize < wavin.shape[0]:

        xbuff = wavin[i*buffsize:(i+1)*buffsize]
        Y = frame_sub_analysis(xbuff,H,N)        
        Yc = donothing(Y)

        Ytot = np.r_[Ytot,Yc]
        Yh = idonothing(Yc)
        yhbuff = np.r_[yhbuff,Yh]

        
        while yhbuff.shape[0] - lines_encoded >= ybuffsize:
            ybuff = yhbuff[lines_encoded:lines_encoded+ ybuffsize, :]
            xsynth = frame_sub_synthesis(ybuff,G)
            #print(ybuff.shape)
            xhat = np.r_[xhat,xsynth]
            lines_encoded = lines_encoded + ybuff.shape[0]


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
print(wavin.shape)
xhat,Ytot = codec0(wavin,h,M,N)
print(xhat.shape)
print("Hello")






import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mp3 import make_mp3_analysisfb
from mp3 import make_mp3_synthesisfb

data = np.load("h.npy", allow_pickle=True).tolist()
h = data['h'].squeeze()
M = 32
fs = 44100

H = make_mp3_analysisfb(h, M)
G = make_mp3_synthesisfb(h, M)

vals = np.zeros(H.shape)

for i in np.arange(H.shape[1]):
    freq, Hf = signal.freqz(H[:,i])
    Hfabs = np.absolute(Hf)
    vals[:,i] = 10 *  np.log10(Hfabs*Hfabs)


plt.plot(freq*fs/(2*np.pi),vals)
plt.xlabel("Hz")
plt.ylabel("dB")
plt.title("Μέτρο των συναρτήσεων μεταφοράς των φίλτρων")
plt.show()

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
    freq, Hf = signal.freqz(H[:,i],fs=fs)
    Hfabs = np.absolute(Hf)
    vals[:,i] = 10 *  np.log10(Hfabs*Hfabs)

z = 13*np.arctan(0.00076*freq) + 3.5*np.arctan((freq/7500)**2)


plt.figure()
plt.plot(freq,vals)
plt.xlabel("Hz")
plt.ylabel("dB")
plt.title("Μέτρο των συναρτήσεων μεταφοράς των φίλτρων στη συχνοτητα f")

plt.figure()
plt.plot(z,vals)
plt.xlabel("Barks")
plt.ylabel("dB")
plt.title("Μέτρο των συναρτήσεων μεταφοράς των φίλτρων στη συχνοτητα ")
plt.show()





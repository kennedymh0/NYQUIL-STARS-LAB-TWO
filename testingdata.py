import ugradio
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
import math

son = np.load('/home/radiopi/Astro-121-Radio-Stars/LAB2/newdataa/secondtest/son_1.0.npz', allow_pickle=True)
soff = np.load('/home/radiopi/Astro-121-Radio-Stars/LAB2/newdataa/secondtest/off_1.0.npz', allow_pickle=True)

plt.plot(np.log10(son['p']))
plt.plot(np.log10(soff['p']))
plt.show()

plt.plot(son['p'])
plt.plot(soff['p'])
plt.show()

plt.plot(son['p']/soff['p'])
plt.show()

plt.plot(np.mean(son['p']))
plt.plot(np.mean(soff['p']))
plt.show()

plt.plot(np.median(son['p']))
plt.plot(np.median(soff['p']))
plt.show()

#meta = son['metadata'].item()
#f_lo= meta['lo']
#f_s = meta['sr']
#n_bins = meta['nsamples']
#f_rest = 1420e6
#c = 3e8

#def get_clean_spectrum(data_array):
    #while data_array.ndim >1:
        #data_array = np.mean(data_array, axis=0)
   # return data_array
##p_on = get_clean_spectrum(son['p'])
#p_off = get_clean_spectrum(soff['p'])

#T_sys = 150
#T_b = T_sys * (p_on - p_off) / p_off
#freqs = np.fft.fftshift(np.fft.fftfreq(n_bins, 1/f_s)) + f_lo

#velocities = c * (f_rest - freqs) / f_rest

#def hi_gaussian(v, amp, v0, sigma, offset):
    #return amp * np.exp(-(v - v0)**2 / (2*sigma**2)) +offset
    
#p0 = [ np.max(T_b), velocities[np.argmax(T_b)], 10.0, 0.0] #[amp, center_v, width, baseline]
#popt, pcov = curve_fit(hi_gaussian, velocities, T_b, p0=p0)



#plt.step(velocities, T_b, where='mid', color='black', label='Calibrated Data') 
#plt.plot(velocities, hi_gaussian(velocities, *popt), color='red', label='Gaussian Fit')
#plt.show()


print("on spectrum max: " + f"{np.log10(son['p']).max()}")
print("off spectrum max: " + f"{np.log10(soff['p']).max()}")

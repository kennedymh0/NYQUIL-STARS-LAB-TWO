import numpy as np
import matplotlib.pyplot as plt

s_on = np.load('data/son.npz')
s_off = np.load('data/soff.npz')
freqs_off = s_off['freqs_hz']
freqs_on = s_on['freqs_hz']
lo_on = s_on['lo_freq']

plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 1.5

plt.figure(1) 
plt.plot(freqs_on, s_on, label ='S_on (Signal)', color = 'plasma', alpha=0.9)
plt.plot(freqs_off, s_off, label ='S_off (Signal)', color = 'viridis', alpha=0.9)

plt.title('Average Power Spectra (Raw Data)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (arb)')
plt.legend()
plt.tight_layout()

plt.savefig('average_power_spectrum.png', dpi=300)

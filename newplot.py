import numpy as np
import matplotlib.pyplot as plt

# Load the collected data
d_on  = np.load("data/son.npz")
d_off = np.load("data/soff.npz")

# Average across blocks
s_on  = np.nanmean(d_on["spectra"], axis=0)
s_off = np.nanmean(d_off["spectra"], axis=0)

# Section 7.2 Analysis: Get the Line Shape
# Dividing s_on by s_off removes the systematic 'hump' of the SDR filters.
line_ratio = s_on / s_off

plt.figure(figsize=(10, 8))

# Subplot 1: Verification (Section 6.2)
plt.subplot(2, 1, 1)
plt.plot(d_on["freqs_hz"]/1e6, s_on, label="SDR tuned to RF-0.5MHz")
plt.plot(d_off["freqs_hz"]/1e6, s_off, label="SDR tuned to RF+0.5MHz")
plt.title("Experimental Verification (Section 6.2)")
plt.ylabel("Raw Power")
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Line Shape (Section 7.2)
plt.subplot(2, 1, 2)
plt.plot(d_on["freqs_hz"]/1e6, line_ratio, color='black', label="Line Shape (Ratio)")
plt.axhline(1.0, color='red', ls='--', alpha=0.5)
plt.title("Bandpass-Corrected Line Shape (s_on / s_off)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Normalized Intensity")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Load the data
d_on = np.load('data/son.npz')
d_off = np.load('data/soff.npz')

# Averages
s_on_z = np.nanmean(d_on['spectra'], axis=0)
s_off_z = np.nanmean(d_off['spectra'], axis=0)
f_on = d_on['freqs_hz']
f_off = d_off['freqs_hz']

# Constants for lab
HI_REST = 1420.405752e6
C = 299792.458 # km/s

# 1. APPLY DC ZAP (Manual fix for the 0Hz spike)
'''
def zap(data):
    d = data.copy()
    c = len(d)//2
    d[c-2:c+3] = np.mean([d[c-3], d[c+3]])
    return d

s_on_z = zap(s_on)
s_off_z = zap(s_off)
'''

# 2. RATIO & SMOOTHING
ratio = s_on_z / s_off_z
# Boxcar smoothing
def smooth(x, n=15):
    return np.convolve(x, np.ones(n)/n, mode='same')
r_smooth = smooth(ratio)

# 3. VELOCITY AXIS (Relative to Rest Freq)
# Convention: Positive velocity is moving away (Redshift)
vels = -C * (f_on - HI_REST) / HI_REST

# 4. STYLIZED PLOTTING
plt.style.use('dark_background') # "Cool" astronomer aesthetic
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Subplot 1: Raw Verification
ax1.plot(f_on/1e6, s_on_z, color='#00d4ff', label='On-Position (SDR shifted Left)', alpha=0.8)
ax1.plot(f_off/1e6, s_off_z, color='#ff007f', label='Off-Position (SDR shifted Right)', alpha=0.6)
ax1.axvline(HI_REST/1e6, color='white', linestyle='--', alpha=0.5, label='Rest Frequency')
ax1.set_title("Spectral Power Overlap (Verification Stage)", color='yellow', fontsize=14)
ax1.set_ylabel("Power [Arbitrary Units]")
ax1.set_xlabel("Frequency [MHz]")
ax1.legend(loc='upper right', frameon=True, facecolor='black')
ax1.grid(True, alpha=0.1)

# Subplot 2: The Scientific Result (Ratio)
ax2.fill_between(f_on/1e6, 1.0, r_smooth, where=(r_smooth > 1.0), color='#00ff41', alpha=0.3, label='Positive Peak (ON)')
ax2.fill_between(f_on/1e6, 1.0, r_smooth, where=(r_smooth < 1.0), color='#ff3131', alpha=0.3, label='Negative Dip (OFF)')
ax2.plot(f_on/1e6, ratio, color='white', alpha=0.15, label='Raw Ratio')
ax2.plot(f_on/1e6, r_smooth, color='#00ff41', linewidth=2, label='Smoothed Line Profile')

# Add a second x-axis for Velocity
ax2v = ax2.twiny()
ax2v.set_xlim(-C * (f_on[0] - HI_REST) / HI_REST, -C * (f_on[-1] - HI_REST) / HI_REST)
ax2v.set_xlabel("LSR Velocity [km/s]", color='orange')
ax2v.tick_params(axis='x', colors='orange')

ax2.axhline(1.0, color='white', linestyle='-', alpha=0.3)
ax2.set_title("Bandpass-Corrected Hydrogen Line Profile ($S_{on}/S_{off}$)", color='yellow', fontsize=14)
ax2.set_ylabel("Normalized Intensity")
ax2.set_xlabel("Frequency [MHz]")
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.1)

# Text Annotations for the Report
peak_val = np.max(r_smooth)
peak_f = f_on[np.argmax(r_smooth)]/1e6
ax2.annotate(f'Hydrogen Detection\nFreq: {peak_f:.2f} MHz', 
             xy=(peak_f, peak_val), xytext=(peak_f+0.2, peak_val+0.05),
             arrowprops=dict(arrowstyle='->', color='yellow'))

plt.tight_layout()
plt.savefig('cool_astronomy_plot.png')
print("Plot saved successfully.")

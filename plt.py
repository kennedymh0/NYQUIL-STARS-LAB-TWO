import numpy as np
import matplotlib.pyplot as plt

# Load data
son = np.load('son.npz')
soff = np.load('soff.npz')

spectra_on = son['spectra']
spectra_off = soff['spectra']
freqs_on = son['freqs_hz']
lo_on = son['lo_freq']

avg_on = np.nanmean(spectra_on, axis=0)
avg_off = np.nanmean(spectra_off, axis=0)

def zap_dc(spec):
    s = spec.copy()
    c = len(s) // 2
    s[c-2:c+3] = np.mean([s[c-3], s[c+3]])
    return s

ratio = zap_dc(avg_on) / zap_dc(avg_off)

def boxcar_smooth(x, n=15):
    return np.convolve(x, np.ones(n)/n, mode='same')

ratio_smooth = boxcar_smooth(ratio)

# Find the center index and frequency
center_idx = len(freqs_on) // 2
center_f = lo_on # The LO is the center of the window

# Find peaks on left and right side of center
left_half = ratio_smooth[:center_idx-50] # exclude zapped area
right_half = ratio_smooth[center_idx+50:]

p_left_idx = np.argmax(left_half)
p_right_idx = np.argmax(right_half) + (center_idx + 50)

f_left = freqs_on[p_left_idx]
f_right = freqs_on[p_right_idx]

offset_left = (f_left - center_f) / 1e3 # kHz
offset_right = (f_right - center_f) / 1e3 # kHz

# Plotting
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(freqs_on/1e6, ratio, color='gray', alpha=0.3, label='Raw Ratio')
ax.plot(freqs_on/1e6, ratio_smooth, color='#00ff41', linewidth=2, label='Smoothed Ratio')

# Label Center
ax.axvline(center_f/1e6, color='white', linestyle='--', alpha=0.5, label=f'LO Center: {center_f/1e6:.2f} MHz')

# Label Peaks
ax.scatter([f_left/1e6, f_right/1e6], [ratio_smooth[p_left_idx], ratio_smooth[p_right_idx]], 
           color='yellow', s=100, zorder=5)

ax.annotate(f"Left Peak\n{offset_left:.1f} kHz", (f_left/1e6, ratio_smooth[p_left_idx]), 
            textcoords="offset points", xytext=(0,10), ha='center', color='yellow')
ax.annotate(f"Right Peak (Mirror)\n{offset_right:.1f} kHz", (f_right/1e6, ratio_smooth[p_right_idx]), 
            textcoords="offset points", xytext=(0,10), ha='center', color='yellow')

ax.set_title("IQ Symmetry Proof: Identifying Instrumental Artifacts", fontsize=14, color='yellow')
ax.set_xlabel("Frequency (MHz)", fontsize=12)
ax.set_ylabel("Normalized Intensity", fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.1)

plt.tight_layout()
plt.savefig('symmetry_proof.png')

print(f"LO Center: {center_f/1e6:.4f} MHz")
print(f"Left Peak: {f_left/1e6:.4f} MHz (Offset: {offset_left:.2f} kHz)")
print(f"Right Peak: {f_right/1e6:.4f} MHz (Offset: {offset_right:.2f} kHz)")

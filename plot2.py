import numpy as np
import matplotlib.pyplot as plt

# Load data
son = np.load('son.npz')
soff = np.load('soff.npz')

# Extract spectra and metadata
spectra_on = son['spectra']
spectra_off = soff['spectra']
freqs_on = son['freqs_hz']
lo_on = son['lo_freq']
lo_off = soff['lo_freq']

print(f"ON LO Freq: {lo_on/1e6:.4f} MHz")
print(f"OFF LO Freq: {lo_off/1e6:.4f} MHz")

# Compute averages
avg_on = np.nanmean(spectra_on, axis=0)
avg_off = np.nanmean(spectra_off, axis=0)

# Zap the DC spike
def zap_dc(spec):
    s = spec.copy()
    c = len(s) // 2
    s[c-2:c+3] = np.mean([s[c-3], s[c+3]])
    return s

avg_on_zapped = zap_dc(avg_on)
avg_off_zapped = zap_dc(avg_off)

# Calculate ratio
ratio = avg_on_zapped / avg_off_zapped

# Smoothing
def boxcar_smooth(x, n=15):
    return np.convolve(x, np.ones(n)/n, mode='same')

ratio_smooth = boxcar_smooth(ratio)

# Find max/min in ratio
peak_idx = np.argmax(ratio_smooth[100:-100]) + 100
peak_freq = freqs_on[peak_idx]

print(f"Ratio Peak at: {peak_freq/1e6:.4f} MHz")

# Plotting
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(freqs_on/1e6, ratio, color='gray', alpha=0.3, label='Raw Ratio (1.5 MHz Shift)')
ax.plot(freqs_on/1e6, ratio_smooth, color='#00ff41', linewidth=2, label='Smoothed Ratio (1.5 MHz Shift)')
ax.axvline(peak_freq/1e6, color='yellow', linestyle='--', label=f'Measured Peak: {peak_freq/1e6:.2f} MHz')
ax.axvline(1419.79, color='magenta', linestyle='--', label='Previous Peak (0.5 MHz Shift)')
ax.axvline(1420.405, color='red', linestyle=':', label='Rest HI (1420.4 MHz)')

ax.set_title("Detection Shift Analysis: 1.5 MHz LO Shift", color='yellow')
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Normalized Intensity")
ax.legend()
ax.grid(True, alpha=0.1)

plt.tight_layout()
plt.savefig('frequency_shift_proof.png')

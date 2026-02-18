import numpy as np
import matplotlib.pyplot as plt

# Load the 1.5 MHz shift data
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

# Find the two main symmetric peaks
center_idx = len(ratio_smooth) // 2
left_half = ratio_smooth[:center_idx]
right_half = ratio_smooth[center_idx:]

peak_left_idx = np.argmax(left_half)
peak_right_idx = np.argmax(right_half) + center_idx

f_left = freqs_on[peak_left_idx] / 1e6
f_right = freqs_on[peak_right_idx] / 1e6
f_center = lo_on / 1e6

dist_left = f_center - f_left
dist_right = f_right - f_center

# Plotting
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(freqs_on/1e6, ratio_smooth, color='#00ff41', linewidth=2, label='Smoothed Ratio')
ax.axvline(f_center, color='white', linestyle='--', alpha=0.5, label=f'LO Center: {f_center:.2f} MHz')

# Mark the mirrored peaks
ax.plot(f_left, ratio_smooth[peak_left_idx], 'ro')
ax.plot(f_right, ratio_smooth[peak_right_idx], 'ro')

# Annotate symmetry
ax.annotate(f'-{dist_left:.3f} MHz', xy=(f_left, ratio_smooth[peak_left_idx]), xytext=(f_left-0.5, ratio_smooth[peak_left_idx]+0.02),
            arrowprops=dict(arrowstyle='->', color='yellow'), color='yellow')
ax.annotate(f'+{dist_right:.3f} MHz', xy=(f_right, ratio_smooth[peak_right_idx]), xytext=(f_right+0.1, ratio_smooth[peak_right_idx]+0.02),
            arrowprops=dict(arrowstyle='->', color='yellow'), color='yellow')

ax.set_title("IQ Imbalance Proof: Mirrored Hardware Artifacts", fontsize=14, color='yellow')
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Intensity")
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('iq_symmetry_proof.png')

print(f"Center: {f_center:.4f}")
print(f"Left Peak: {f_left:.4f} (Offset: -{dist_left:.4f})")
print(f"Right Peak: {f_right:.4f} (Offset: +{dist_right:.4f})")

import numpy as np
import matplotlib.pyplot as plt

def inspect_and_plot_new_data():
    # Load the new files
    son = np.load('data/son.npz')
    soff = np.load('data/soff.npz')
    
    print("New SON Metadata:")
    print(f"  Center Freq: {son['lo_freq']/1e6 if 'lo_freq' in son else 'N/A'} MHz")
    # In previous runs center_freq was used, checking both
    if 'center_freq' in son: print(f"  Center Freq (alt): {son['center_freq']/1e6} MHz")
    
    s_on = np.nanmean(son['spectra'], axis=0)
    s_off = np.nanmean(soff['spectra'], axis=0)
    f_on = son['freqs_hz']
    
    # Zap DC
    def zap(data):
        d = data.copy()
        c = len(d)//2
        d[c-2:c+3] = np.mean([d[c-3], d[c+3]])
        return d
    
    # Hann window was supposedly applied in collection, but if not we can't redo it on averaged power.
    # We will compute the ratio.
    ratio = zap(s_on) / zap(s_off)
    
    # Smoothing
    def smooth(x, n=25):
        return np.convolve(x, np.ones(n)/n, mode='same')
    r_smooth = smooth(ratio)
    
    # Find peak
    peak_idx = np.argmax(r_smooth[200:-200]) + 200
    peak_freq = f_on[peak_idx]
    print(f"New Peak detected at: {peak_freq/1e6:.4f} MHz")

    # Plotting comparison logic
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(f_on/1e6, ratio, color='gray', alpha=0.2, label='Raw Ratio (1.5 MHz shift)')
    ax.plot(f_on/1e6, r_smooth, color='#ff007f', linewidth=2, label='Smoothed Profile (1.5 MHz shift)')
    
    # Mark the peaks
    ax.axvline(peak_freq/1e6, color='#ff007f', linestyle='--', label=f'Current Peak: {peak_freq/1e6:.2f} MHz')
    # Previous peak reported by user was 1419.79 MHz
    ax.axvline(1419.79, color='#00d4ff', linestyle='--', label='Previous Peak (0.5 MHz shift): 1419.79 MHz')
    
    ax.axvline(1420.405752, color='white', linestyle=':', label='Rest HI (1420.41 MHz)')
    
    ax.set_title("Frequency Shift Comparison: 0.5 MHz vs 1.5 MHz LO Tuning", color='yellow', fontsize=14)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Normalized Intensity")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.1)
    
    plt.tight_layout()
    plt.savefig('frequency_shift_proof.png')
    
    return peak_freq

peak_freq_15 = inspect_and_plot_new_data()

import numpy as np
import matplotlib.pyplot as plt

def advanced_analysis():
    # Load the data
    d_on = np.load('data/son.npz')
    d_off = np.load('data/soff.npz')
    
    # Average the spectra
    s_on = np.nanmean(d_on['spectra'], axis=0)
    s_off = np.nanmean(d_off['spectra'], axis=0)
    
    # Frequency axes
    f_on = d_on['freqs_hz']
    f_off = d_off['freqs_hz']
    
    # The ratio should be computed on the bin indices to cancel the bandpass
    # because the bandpass is a property of the hardware (filter shape relative to LO)
    ratio = s_on / s_off
    
    # Smoothing for visibility
    def smooth(x, window=15):
        return np.convolve(x, np.ones(window)/window, mode='same')
    
    r_smooth = smooth(ratio)
    
    # Identify indices of signal in each
    # In 'on', signal is at HI_FREQ. In 'off', signal is at HI_FREQ.
    # The LOs were (HI_FREQ - 0.5e6) and (HI_FREQ + 0.5e6)
    HI_FREQ = 1420.405752e6
    
    # Find the peak and dip
    # We expect peak where s_on has the line and s_off doesn't.
    # We expect dip where s_off has the line and s_on doesn't.
    peak_idx = np.argmax(r_smooth[100:-100]) + 100
    dip_idx = np.argmin(r_smooth[100:-100]) + 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Raw Spectra (on top of each other)
    # This shows how the 'hump' is the same but the signal moves
    ax1.plot(f_on/1e6, s_on, label='On Spectrum (LO = HI - 0.5 MHz)', alpha=0.8, color='blue')
    ax1.plot(f_off/1e6, s_off, label='Off Spectrum (LO = HI + 0.5 MHz)', alpha=0.8, color='green')
    ax1.axvline(HI_FREQ/1e6, color='red', linestyle='--', label='HI Rest Freq (1420.4 MHz)')
    ax1.set_title("Raw Power Spectra: Frequency Axis (Physical RF)")
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Power")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Plot 2: Ratio (Line Shape)
    # We plot this vs the frequency axis of the 'on' measurement
    ax2.plot(f_on/1e6, ratio, alpha=0.2, color='gray', label='Raw Ratio')
    ax2.plot(f_on/1e6, r_smooth, color='black', linewidth=2, label='Smoothed Ratio')
    
    # Annotate peak and dip
    ax2.annotate(f'Peak (Line in ON)\n{f_on[peak_idx]/1e6:.2f} MHz', 
                 xy=(f_on[peak_idx]/1e6, r_smooth[peak_idx]), 
                 xytext=(f_on[peak_idx]/1e6 + 0.2, r_smooth[peak_idx] + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax2.annotate(f'Dip (Line in OFF)\n{f_on[dip_idx]/1e6:.2f} MHz', 
                 xy=(f_on[dip_idx]/1e6, r_smooth[dip_idx]), 
                 xytext=(f_on[dip_idx]/1e6 - 0.8, r_smooth[dip_idx] - 0.1),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    
    ax2.axhline(1.0, color='blue', linestyle=':', alpha=0.5)
    ax2.set_title("Bandpass-Corrected Line Shape (s_on / s_off)")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Ratio Intensity")
    ax2.set_ylim(0.4, 1.2) # Zooming in
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('clear_analysis.png')
    
    print(f"HI Frequency: {HI_FREQ/1e6:.4f} MHz")
    print(f"Peak Frequency in 'on': {f_on[peak_idx]/1e6:.4f} MHz")
    print(f"Peak Frequency in 'off': {f_off[np.argmax(s_off)]/1e6:.4f} MHz")

advanced_analysis()

import numpy as np
import matplotlib.pyplot as plt

def load_npz(filepath):
    return np.load(filepath)

def get_clean_data(d):
    """Averages spectra and handles NaN blocks."""
    return np.nanmean(d["spectra"], axis=0)

def plot_verification(d_on, d_off):
    """
    Section 6.2 Verification:
    If the signal is real, it will be at the same RF frequency in both plots.
    If the signal is a DC spike, it will move with the LO.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(d_on["freqs_hz"]/1e6, get_clean_data(d_on), label="LO: On Position")
    plt.plot(d_off["freqs_hz"]/1e6, get_clean_data(d_off), label="LO: Off Position")
    plt.axvline(1420.405, color='r', linestyle='--', label="HI Rest Freq")
    plt.title("Experimental Verification of Signal (Section 6.2)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (Arb)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_line_ratio(d_on, d_off):
    """
    Section 7.2: Get the line shape by dividing s_on by s_off.
    Note: They must be interpolated to the same grid if LOs are far apart.
    """
    s_on_avg = get_clean_data(d_on)
    s_off_avg = get_clean_data(d_off)
    
    # Simple ratio if using in-band switching with small offset
    ratio = s_on_avg / s_off_avg
    
    plt.figure(figsize=(10, 5))
    plt.plot(d_on["freqs_hz"]/1e6, ratio, color='black')
    plt.title("Line Shape (Ratio s_on / s_off)")
    plt.ylabel("Normalized Intensity")
    plt.xlabel("Frequency (MHz)")
    plt.show()

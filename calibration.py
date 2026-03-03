import numpy as np
import matplotlib.pyplot as plt

def load_npz_data(filename):
    """Loads arrays and metadata from an .npz file."""
    data = np.load(filename, allow_pickle=True)
    return data['frequencies'], data['power'], data['metadata'].item()

def calibrate_and_plot(cold_file, cal_file, T_cal=300):
    # 1. Load the data
    f_cold, p_cold, meta_cold = load_npz_data(cold_file)
    f_cal, p_cal, meta_cal = load_npz_data(cal_file)
    
    # 2. Calibration Logic
    # Calculate the spectral shape (S_line)
    # S_line is the ratio of cold sky power to calibration power
    s_line = p_cold / p_cal 
    
    # Calculate the gain factor
    sum_cold = np.sum(p_cold)
    sum_diff = np.sum(p_cal - p_cold)
    gain_factor = (T_cal / sum_diff) * sum_cold #
    
    # Calculate final intensity calibrated spectrum (T_line)
    t_line = s_line * gain_factor #
    
    # 3. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot Raw Powers
    axes[0].plot(f_cold/1e6, p_cold, label='Cold Sky (Target)')
    axes[0].plot(f_cal/1e6, p_cal, label='Calibrator (Blackbody)', alpha=0.7)
    axes[0].set_title("Raw Power Spectra")
    axes[0].set_ylabel("Power (ADC units^2)")
    axes[0].legend()

    # Plot Bandpass-Corrected Shape (S_line)
    axes[1].plot(f_cold/1e6, s_line, color='green')
    axes[1].set_title("Instrumental Bandpass Removed (S_line)")
    axes[1].set_ylabel("Relative Intensity")

    # Plot Final Temperature Calibrated Spectrum (T_line)
    axes[2].plot(f_cold/1e6, t_line, color='red')
    axes[2].set_title(f"Temperature Calibrated Spectrum (T_line)")
    axes[2].set_xlabel("Frequency Offset (MHz)")
    axes[2].set_ylabel("Temperature (K)")
    
    plt.tight_layout()
    plt.show()

# Example usage:
# calibrate_and_plot('Zenith_Cold.npz', 'Ambient_Load_Cal.npz', T_cal=300)
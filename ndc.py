import ugradio
import numpy as np
import matplotlib.pyplot as plt
import os

# --- OBSERVATION PARAMETERS ---
TRIAL_NAME = "Zenith_21cm_Survey"
CENTER_FREQ = 1420.395e6  # 1.420 GHz
SAMPLE_RATE = 2.2e6       #
NSAMPLES = 2048           #
NBLOCKS = 2000            # Total blocks to average for high SNR
GAIN = 10                 # SDR Gain

def run_observation():
    print(f"Starting collection for {TRIAL_NAME}...")
    
    # 1. Capture data from SDR
    # This captures NBLOCKS of data at once
    raw_data = ugradio.sdr.capture_data(
        direct=False, 
        center_freq=CENTER_FREQ, 
        nsamples=NSAMPLES, 
        nblocks=NBLOCKS, 
        sample_rate=SAMPLE_RATE, 
        gain=GAIN
    )

    # 2. Process into a Power Spectrum
    # average the raw data
    fourier = np.fft.fft(raw_data)
    fourier_freq = np.fft.fftfreq(NSAMPLES, 1/SAMPLE_RATE)
    
    # Shift zero-frequency component to the center
    f_shift = np.fft.fftshift(fourier_freq)
    p_shift = np.fft.fftshift(fourier)
    
    # Calculate average power (V^2) across all blocks
    avg_power_spectrum = np.mean(np.abs(p_shift)**2, axis=0)

    # 3. Quick Plot for Verification
    plt.figure(figsize=(10, 4))
    plt.plot(f_shift / 1e6, avg_power_spectrum)
    plt.title(f"Live Power Spectrum: {TRIAL_NAME}")
    plt.xlabel("Frequency Offset (MHz)")
    plt.ylabel("Power (Arb. Units)")
    plt.grid(True)
    plt.show()

    # 4. Save to .npz File
    save_to_npz(f_shift, avg_power_spectrum)

def save_to_npz(freqs, power):
    # Consolidate all metadata from SDR_gather.py logic
    metadata = {
        'trial': TRIAL_NAME,
        'center_freq_hz': CENTER_FREQ,
        'sample_rate_hz': SAMPLE_RATE,
        'gain': GAIN,
        'nblocks': NBLOCKS,
        'nsamples': NSAMPLES,
        'unix_time': ugradio.timing.unix_time(), #
        'local_time': ugradio.timing.local_time() #
    }
    
    filename = f"{TRIAL_NAME}.npz"
    # Save the arrays and the metadata dictionary together
    np.savez(filename, metadata=metadata, frequencies=freqs, power=power)
    print(f"Successfully saved data to {filename}")

if __name__ == "__main__":
    run_observation()
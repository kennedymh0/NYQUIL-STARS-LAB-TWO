import numpy as np
import ugradio
import ugradio.timing as timing
import os

# Parameters for Section 6.2
TARGET_RF   = 1420.405752e6 # The frequency of interest
SAMPLE_RATE = 2.4e6         #
NSAMPLES    = 2048
N_BLOCKS    = 500           # Integration blocks
OUT_DIR     = "data"

def capture_at(label, rf_freq):
    """Captures data at a specific RF tuning."""
    print(f"\n[{label}] Tuning SDR to {rf_freq/1e6:.3f} MHz...")
    s = ugradio.sdr.SDR(center_freq=rf_freq, sample_rate=SAMPLE_RATE, gain=40)
    
    spectra = np.zeros((N_BLOCKS, NSAMPLES))
    for i in range(N_BLOCKS):
        try:
            raw = s.capture_data(nblocks=1, nsamples=NSAMPLES)
            # Power spectrum calculation
            spectra[i] = np.abs(np.fft.fftshift(np.fft.fft(raw[0])))**2
            if i == 0:
                # Level check for Section 6.2
                print(f"  Levels: std={raw[0].real.std():.4f}, max={raw[0].real.max():.4f}")
        except:
            spectra[i] = np.nan
    s.close()

    # Create frequency axis centered on the tuning frequency
    freqs = np.fft.fftshift(np.fft.fftfreq(NSAMPLES, 1.0/SAMPLE_RATE)) + rf_freq
    
    fname = os.path.join(OUT_DIR, f"{label}.npz")
    np.savez(fname, spectra=spectra, freqs_hz=freqs, center_freq=rf_freq)
    print(f"  Saved to {fname}")
    return fname

# --- RUN EXPERIMENT ---
os.makedirs(OUT_DIR, exist_ok=True)

# Frequency Switching: Position 1 (Shifted down 0.5 MHz)
capture_at("son", TARGET_RF - 0.5e6)

# Frequency Switching: Position 2 (Shifted up 0.5 MHz)
capture_at("soff", TARGET_RF + 0.5e6)

print("\nData collection complete. Run visualize.py next.")

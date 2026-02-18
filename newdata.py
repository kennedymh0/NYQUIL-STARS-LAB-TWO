import numpy as np
import ugradio
import ugradio.timing as timing
import os

# Lab Constants
HI_FREQ     = 1420.405752e6
SAMPLE_RATE = 2.4e6
NSAMPLES    = 2048
N_BLOCKS    = 500
OUT_DIR     = "data"

def zap_dc(spec):
    """Removes the hardware DC spike at the center of the FFT."""
    s = spec.copy()
    c = len(s) // 2
    # Interpolate across the center 3 bins to remove the spike
    s[c-1:c+2] = (s[c-2] + s[c+2]) / 2
    return s

def power_spectrum(iq, nsamples=NSAMPLES):
    """Calculates power spectrum with Hann windowing and DC zapping."""
    # 1. Apply Hann Window to reduce spectral leakage
    w = np.hanning(len(iq))
    iq_windowed = iq * w
    
    # 2. Compute the FFT and shift
    spec = np.abs(np.fft.fftshift(np.fft.fft(iq_windowed, n=nsamples))) ** 2
    
    # 3. Zap the DC spike
    return zap_dc(spec)

def freq_axis(lo_freq, rate=SAMPLE_RATE, nsamples=NSAMPLES):
    """Generates the RF axis based on the SDR's Local Oscillator."""
    return np.fft.fftshift(np.fft.fftfreq(nsamples, 1.0/rate)) + lo_freq

def check_levels(iq):
    """Verifies gain to prevent clipping/quantization."""
    r = iq.real
    std = r.std()
    print(f"  Levels: std={std:.4f}, min={r.min():.4f}, max={r.max():.4f}")
    if np.mean(np.abs(r) > 0.95 * np.abs(r).max()) > 0.01:
        print("  !! WARNING: Clipping detected - Lower Gain")
    elif std < 0.005:
        print("  !! WARNING: Low Signal - Increase Gain")

def capture_at(label, lo_freq, nblocks=N_BLOCKS):
    """Captures data at a specific LO frequency."""
    print(f"\n[{label}] Tuning SDR (LO) to {lo_freq/1e6:.3f} MHz...")
    s = ugradio.sdr.SDR(center_freq=lo_freq, sample_rate=SAMPLE_RATE, gain=40)
    
    spectra = np.zeros((nblocks, NSAMPLES))
    for i in range(nblocks):
        try:
            raw = s.capture_data(nblocks=1, nsamples=NSAMPLES)
            spectra[i] = power_spectrum(raw[0])
            if i == 0: check_levels(raw[0])
        except Exception as e:
            print(f"  Error at block {i}: {e}")
            spectra[i] = np.nan
    s.close()

    freqs = freq_axis(lo_freq)
    
    fname = os.path.join(OUT_DIR, f"{label}.npz")
    np.savez(fname, spectra=spectra, freqs_hz=freqs, lo_freq=lo_freq)
    print(f"  → Saved to {fname}")
    return fname

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    # Perform Frequency Switching
    capture_at("son", HI_FREQ - .6e6)
    capture_at("soff", HI_FREQ + .6e6)
    print("\nDone. Use visualize.py to see the bandpass-corrected ratio.")

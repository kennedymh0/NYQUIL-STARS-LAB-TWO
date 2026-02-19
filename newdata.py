import numpy as np
import ugradio
import ugradio.timing as timing
import os

# Lab Constants
HI_FREQ     = 1420.405752e6
SAMPLE_RATE = 2.4e6
NSAMPLES    = 4096
N_BLOCKS    = 3
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
    
    #if iq.ndim ==2:
        #iq = iq[:,0] +1j * iq[:,1]
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
    #r = iq[:,0] if iq.ndim == 2 else iq.real
    std = r.std()
    print(f"  Levels: std={std:.4f}, min={r.min():.4f}, max={r.max():.4f}")
    if (r.min() = -128 or r.max() = 127):
        print("  !! WARNING: Clipping detected - Lower Gain")
    elif (r.max() < 10 and r.min() > -10):
        print("  !! WARNING: Low Signal - Increase Gain")

def capture_at(label, lo_freq, nblocks=N_BLOCKS):
    """Captures data at a specific LO frequency."""
    print(f"\n[{label}] Tuning SDR (LO) to {lo_freq/1e6:.3f} MHz...")
    s = ugradio.sdr.SDR(direct=False, center_freq=lo_freq, sample_rate=SAMPLE_RATE, gain=10)
    
    spectra = np.zeros((nblocks, NSAMPLES))
    _raw = s.capture_data(nblocks=nblocks+1, nsamples=NSAMPLES)
    raw = _raw[...,0]+1j * _raw[...,1]
    s.close()
    check_levels(raw[1])
    for i in range(nblocks):
        try:
            spectra[i] = power_spectrum(raw[i+1])
        except Exception as e:
            print(f"  Error at block {i}: {e}")
            spectra[i] = np.nan

    freqs = freq_axis(lo_freq)
    
    fname = os.path.join(OUT_DIR, f"{label}.npz")
    np.savez(fname, spectra=spectra, freqs_hz=freqs, lo_freq=lo_freq)
    print(f"  → Saved to {fname}")
    return fname

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    # Perform Frequency Switching
    for i in range(1, 10):
        shift = 0.5 + i*0.1
        capture_at(f"son_{shift}", HI_FREQ + shift*(1e6))
    print("\nDone. Use visualize.py to see the bandpass-corrected ratio.")

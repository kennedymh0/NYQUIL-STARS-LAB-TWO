import numpy as np
import ugradio
import ugradio.timing as timing
import os

# Constants from Lab Manual
HI_FREQ = 1420.405752e6  # 21cm Line
SAMPLE_RATE = 2.4e6      # Recommended stable rate
NSAMPLES = 2048
N_BLOCKS = 200

def make_sdr(target_freq, rate=SAMPLE_RATE, gain=40):
    """
    Tunes the R820T2 analog mixer to target_freq.
    The manual notes that 0-frequency (DC) spikes occur at this center.
    """
    return ugradio.sdr.SDR(freq=target_freq, rate=rate, gain=gain)

def power_spectrum(iq, nsamples=NSAMPLES):
    # Hann windowing removed per request; using raw FFT for birdie testing
    return np.abs(np.fft.fftshift(np.fft.fft(iq, n=nsamples))) ** 2

def freq_axis(center_freq, rate=SAMPLE_RATE, nsamples=NSAMPLES):
    """Returns the actual RF frequencies observed."""
    return np.fft.fftshift(np.fft.fftfreq(nsamples, 1.0/rate)) + center_freq

def check_levels(iq):
    """Experimental verification of signal levels."""
    r = iq.real
    std = r.std()
    print(f"  std={std:.4f}  min={r.min():.4f}  max={r.max():.4f}")
    if np.mean(np.abs(r) > 0.95 * np.abs(r).max()) > 0.01:
        print("  !! Clipping detected — reduce gain")
    elif std < 0.005:
        print("  !! Heavily quantized — increase gain")
    else:
        print("  Levels OK")

def measure(label, target_freq, nblocks=N_BLOCKS, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    jd_start = timing.julian_date()
    
    print(f"\n[{label}] Tuning SDR to {target_freq/1e6:.3f} MHz...")
    s = make_sdr(target_freq)
    spectra = np.zeros((nblocks, NSAMPLES))

    for i in range(nblocks):
        try:
            raw = s.capture_data(nblocks=1, nsamples=NSAMPLES)
            spectra[i] = power_spectrum(raw[0])
            if i == 0: check_levels(raw[0])
        except Exception as e:
            spectra[i] = np.nan

    s.close()
    
    fname = os.path.join(out_dir, f"{label}_{int(jd_start * 1e5)}.npz")
    np.savez(fname,
             spectra=spectra,
             freqs_hz=freq_axis(target_freq),
             center_freq=target_freq,
             jd_start=jd_start,
             sample_rate=SAMPLE_RATE)
    print(f"  → Saved: {fname}")
    return spectra, fname

def observe_frequency_switch(lo_on=1420.4e6, lo_off=1421.4e6, nblocks=500, out_dir="data"):
    """
    Performs Frequency Switching.
    Shifting the LO moves the 21cm line within the 2.4MHz window.
    """
    print("=== STARTING FREQUENCY SWITCHED DATA COLLECTION ===")
    s_on, f_on = measure("son", target_freq=lo_on, n

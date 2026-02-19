import ugradio
import numpy as np
import matplotlib.pyplot as plt
import os

SAMPLE_RATE=3.2e6
HI_FREQ = 1420.405e6
nblocks = 1
NSAMPLES=4096
GAIN = 10 # IN DBM
SG_GAIN = -70 # IN DBM

cwd = os.getcwd()

def check_levels(iq):
    """Verifies gain to prevent clipping/quantization."""
    r = iq.real
    #r = iq[:,0] if iq.ndim == 2 else iq.real
    std = r.std()
    print(f"  Levels: std={std:.4f}, min={r.min():.4f}, max={r.max():.4f}")
    if (r.min() == -128 or r.max() == 127):
        print("  !! WARNING: Clipping detected - Lower Gain")
    elif (r.max() < 10 and r.min() > -10):
        print("  !! WARNING: Low Signal - Increase Gain")

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
	
def capture_at(name, lo=1420e6):
	s = ugradio.sdr.SDR(direct=False, center_freq=lo, sample_rate=SAMPLE_RATE, gain=10)
	_raw = s.capture_data(nblocks=nblocks+1, nsamples=NSAMPLES)
	s.close()
	raw = _raw[...,0]+1j * _raw[...,1]
	raw = raw[1]
	check_levels(raw)
	plt.plot(raw.real)
	plt.plot(raw.imag)
	plt.show()
	metadata = {"lo": lo,
			"sr": SAMPLE_RATE,
			"HI FREQ": HI_FREQ,
			"nblocks": nblocks,
			"nsamples": NSAMPLES,
			"gain": GAIN,
			"SG GAIN": SG_GAIN}
			
	file_path = os.path.join("./newdataa", name)
	p = power_spectrum(raw, nsamples=NSAMPLES)
	np.savez(file_path, raw=raw, metadata=metadata, p=p)

for i in range(1, 10):
    shift = 0.5 + i*0.1
    capture_at(name=f"son_{shift}", lo=HI_FREQ + shift*(1e6))
    capture_at(name=f"off_{shift}", lo=HI_FREQ - shift*(1e6))
    print("\nDone.")

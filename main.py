import ugradio
import ugradio.timing as timing 
import numpy as np
import matplotlib.pyplot as plt
import os

SAMPLE_RATE=3.2e6
HI_FREQ = 1420.405e6
nblocks = 10
NSAMPLES=4096
GAIN = 10 # IN DBM
SG_GAIN = None # IN DBM
MODE = "HORN_gc" # modes: 'SIG_GEN', 'HORN', 'HORN_BB'
NOTE = "pointing at generator"

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
    w = np.hanning(len(iq[0]))
    p = []
    for i in iq:
        x = iq * w # applies window
        spec = np.abs(np.fft.fftshift(np.fft.fft(x, n=nsamples))) ** 2
        p.append(zap_dc(spec))
    
    # 3. Zap the DC spike
    return p
	
def capture_at(name, lo=1420e6):
    s = ugradio.sdr.SDR(direct=False, center_freq=lo, sample_rate=SAMPLE_RATE, gain=10)
    _raw = s.capture_data(nblocks=nblocks+1, nsamples=NSAMPLES)
    local_now = ugradio.timing.local_time()
    print(local_now)
    ut_now = ugradio.timing.unix_time()
    jd_now = ugradio.timing.julian_date()
    lst_now = ugradio.timing.lst()
    print(lst_now)
    s.close()
    raw = _raw[...,0]+1j * _raw[...,1]
    raw = raw[1:]
    check_levels(raw[0])
    plt.plot(raw[0].real)
    plt.plot(raw[0].imag)
    plt.show()
    metadata = {"lo": lo,
		"sr": SAMPLE_RATE,
		"HI FREQ": HI_FREQ,
		"nblocks": nblocks,
		"nsamples": NSAMPLES,
		"gain": GAIN,
		"SG GAIN": SG_GAIN,
		"lst" : lst_now,
		"jd" : jd_now,
		"local": local_now,
		"NOTE": NOTE}			
    file_path = os.path.join("./newdataa", name)
    
    p = power_spectrum(raw, nsamples=NSAMPLES)
    np.savez(file_path, raw=raw, metadata=metadata, p=p)
    
print("Experiment calibrated for the following settings:\n"
    + f"SAMPLE RATE: {SAMPLE_RATE}\n"
    + f"RADIO FREQUENCY: {HI_FREQ}\n"
    + f"NUMBER OF BLOCKS: {nblocks}\n"
    + f"NUMBER OF SAMPLES: {NSAMPLES}\n"
    + f"SIGNAL GENERATOR GAIN: {GAIN}\n"
    + f"SDR GAIN: {SG_GAIN}\n"
    + f"MODE: {MODE}\n")
print("If this is correct, press any key to continue.")
input()
if MODE == "HORN_gc":
	for i in range(1, 10):
		shift = 0.5 + i*0.1
		capture_at(name=f"{MODE}_son_{shift}", lo=HI_FREQ + shift*(1e6))
		capture_at(name=f"{MODE}_off_{shift}", lo=HI_FREQ - shift*(1e6))
		print("\nDone.")
		
shift = 1420e6
capture_at(name=f"{MODE}_{shift}", lo=shift)
#elif MODE == "HORN":
#	ASDFASDF
#else:
#	aasdfasd	

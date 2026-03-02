import ugradio
import ugradio.timing as timing 
import numpy as np
import matplotlib.pyplot as plt
import os

SAMPLE_RATE=2.4e6
HI_FREQ = 1420.405e6
nblocks = 10000
NSAMPLES= 4096
GAIN = 20 # IN DBM
SG_GAIN = None # IN DBM
MODE = "HORN_zenith" # modes: 'SIG_GEN', 'HORN', 'HORN_BB'
NOTE = "pointing zenith measuring 5k blocks first time"

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
    
    w = np.hanning(iq.shape[1])
    windowed_iq = iq * w
    fft_matrix = np.fft.fft(windowed_iq, nsamples, axis=1)
    spec_matrix = np.abs(np.fft.fftshift(fft_matrix, axes=1)) ** 2
    clean_specs = np.apply_along_axis(zap_dc, 1, spec_matrix)
    avg_spec = np.mean(clean_specs, axis=0)
    
    return avg_spec
	
def plot_hydrogen_line(p_avg, metadata):
	lo = metadata['lo']
	sr = metadata['sr']
	nsamples = metadata['nsamples']
	hi_freq = metadata['HI FREQ'] / 1e6
	freqs = np.fft.fftshift(np.fft.fftfreq(nsamples, 1/sr))
	freqs_mhz = (freqs + lo) / 1e6
	
	p_db = 10 * np.log10(p_avg)
	
	plt.figure(figsize=(12,6))
	plt.plot(freqs_mhz, p_db, label='Averages Spectrum', color='blue')
	
	plt.axvline(hi_freq, color='red', linestyle='--', alpha=0.7,
		label=f'HI LINE ({hi_freq:.3f} MHz)')
	plt.title(f"Hydrogen Line Observation (LO: {lo/1e6:.2f} MHz)")
	plt.xlabel("Frequency (MHz)")
	plt.ylabel("Relative Power (dB)")
	plt.grid(True, which='both', ls='-', alpha=0.5)
	plt.legend()
	
	plt.show()
	
def capture_at(name, lo=1420e6):
    all_spectra=[]
    s = ugradio.sdr.SDR(direct=False, center_freq=lo, sample_rate=SAMPLE_RATE, gain=GAIN)
    for i in range(nblocks//10):
        _raw = s.capture_data(nblocks=11)
        raw = _raw[...,0]+1j * _raw[...,1]
        raw = raw[1:] #drop first block
        spec = power_spectrum(raw)
        all_spectra.append(spec)
    s.close()
    
    final_avg = np.mean(all_spectra, axis=0)
    
    local_now = ugradio.timing.local_time()
    print(local_now)
    ut_now = ugradio.timing.unix_time()
    jd_now = ugradio.timing.julian_date()
    lst_now = ugradio.timing.lst()
    print(lst_now)

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
    
    plot_hydrogen_line(final_avg, metadata)
    
    # if there are lots of blocks discard the raw data
    np.savez(file_path, metadata=metadata, p=final_avg) 
		
		
    
print("Experiment calibrated for the following settings:\n"
    + f"SAMPLE RATE: {SAMPLE_RATE}\n"
    + f"RADIO FREQUENCY: {HI_FREQ}\n"
    + f"NUMBER OF BLOCKS: {nblocks}\n"
    + f"NUMBER OF SAMPLES: {NSAMPLES}\n"
    + f"SIGNAL GENERATOR GAIN: {SG_GAIN}\n"
    + f"SDR GAIN: {GAIN}\n"
    + f"MODE: {MODE}\n")
print("If this is correct, press any key to continue.")
input()
#if MODE == "HORN_gc":
#	for i in range(1, 10):
#		shift = 0.5 + i*0.1
#		capture_at(name=f"{MODE}_son_{shift}", lo=HI_FREQ + shift*(1e6))
#		capture_at(name=f"{MODE}_off_{shift}", lo=HI_FREQ - shift*(1e6))
#		print("\nDone.")
		
shift = 1420e6
capture_at(name=f"{MODE}_{shift}", lo=shift)
#elif MODE == "HORN":
#	ASDFASDF
#else:
#	aasdfasd	

import ugradio
import ugradio.timing as timing 
import numpy as np
import matplotlib.pyplot as plt
import os

SAMPLE_RATE=1.8e6
HI_FREQ = 1420.405e6
nblocks = 9999 
NSAMPLES=2048
BATCH_SIZE = 500 #blocks per SDR capture call; avoids memory/stability/runtime issues
GAIN = 10 # IN DBM
SG_GAIN = None # IN DBM
MODE = "HORN_zenith" # modes: 'SIG_GEN', 'HORN', 'HORN_BB'
NOTE = "pointing zenith"
OUTPUT_DIR = "./newerdata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
	
def capture_at(name, lo=1420e6, nblocks=nblocks, batch_size=BATCH_SIZE):
    s = ugradio.sdr.SDR(direct=False, center_freq=lo, sample_rate=SAMPLE_RATE, gain=10)
    
    local_now = ugradio.timing.local_time()
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
    accumulated_spec = np.zeros(NSAMPLES)
    n_batches = int(np.ceil(nblocks/batch_size))
    blocks_done = 0
    consecutive_erroes = 0 
    
    for batch_idx in range(n_batches):
        this_batch = min(batch_size, nblocks-blocks_done)
        if this_batch <= 0:
            break
            
        try:
            extra = 1 if batch_idx == 0 else 0
            _raw = s.capture_data(nblocks=this_batch + extra, nsamples=NSAMPLES)
            raw = _raw[...,0]+1j * _raw[...,1]
            raw = raw[1:] #drop first block
            
            if batch_idx == 0:
                _raw = _raw.astype(np.float32)
            if np.iscomplexobj(_raw):
                raw = _raw
            elif _raw.ndim == 3 and _raw.shape[-1] == 2:
                raw = _raw[...,0]+1j * _raw[...,1]
            elif _raw.ndim == 2:
                raw = _raw[:, 0::2]+1j * _raw[: , 1::2]
            else:
                raise ValueError(f"weird shape: {_raw.shape}")
            
            if batch_idx == 0:
                raw = raw[1:]
            if batch_idx == 0:
                check_levels(raw[0])
                
            batch_specs = np.array([power_spectrum(block) for block in raw])
            accumulated_spec  += batch_specs.sum(axis=0)
            blocks_done += len(raw)
        except Exception as e:
            print(f" error in batch {batch_idx} : {e} -skipping the batch")
        
    s.close()
    
    if blocks_done > 0:
        avg_spec = accumulated/spec / block_done
        final_path = os.path.join(OUTPUT_DIR, f"{name}_avgspec")
        np.savez(final_path, avg_spec=avg_spec, metadata=metadata, blocks_done=blocks_done)
        print("average spectra saved")
        
    else:
        print("where is our data???? what the helly???")
    return avg_spec if blocks_done > 0 else None 
    
    
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

shift = 1420e6
capture_at(name=f"{MODE}", lo=HI_FREQ)

print("collection complete")

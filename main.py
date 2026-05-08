import ugradio
import ugradio.timing as timing 
import numpy as np
import matplotlib.pyplot as plt
import os

SAMPLE_RATE=2.4e6
HI_FREQ = 1420.405e6
nblocks = 30000
NSAMPLES= 8192
GAIN = 20 # IN DBM
LO_LOWER = 1420e6
LO_UPPER = 1420.81e6
window = np.hanning(NSAMPLES)

cwd = os.getcwd()

def zap_dc(spec):
    """Removes the hardware DC spike at the center of the FFT."""
    s = spec.copy()
    c = len(s) // 2
    # Interpolate across the center 3 bins to remove the spike
    s[c-1:c+2] = (s[c-2] + s[c+2]) / 2
    return s

def power_spectrum(iq, nsamples=NSAMPLES):
    """Calculates power spectrum with Hann windowing and DC zapping."""
	iq -= np.mean(iq, axis=1, keepdims=True)
	iq *= window
    fft_matrix = np.fft.fft(iq, nsamples, axis=1)
    spec_matrix = np.abs(np.fft.fftshift(fft_matrix, axes=1)) ** 2
    
    avg_spec = np.mean(spec_matrix, axis=0)
    
    return avg_spec
	
def capture_at(name):
    all_spectra=[]
    s = ugradio.sdr.SDR(direct=False, center_freq=LO, sample_rate=SAMPLE_RATE, gain=GAIN)
	
	local_start = ugradio.timing.local_time()
    ut_start = ugradio.timing.unix_time()
    jd_start = ugradio.timing.julian_date()
    lst_start = ugradio.timing.lst()

    for i in range(nblocks//10):
        _raw = s.capture_data(nblocks=11)
        raw = _raw[...,0]+1j * _raw[...,1]
        raw = raw[1:] #drop first block
        spec = power_spectrum(raw)
        all_spectra.append(spec)
    s.close()
    
    final_avg = np.mean(all_spectra, axis=0)
    
    local_end = ugradio.timing.local_time()
    ut_end = ugradio.timing.unix_time()
    jd_end = ugradio.timing.julian_date()
    lst_end = ugradio.timing.lst()

    metadata = {"lo": lo,
		"sr": SAMPLE_RATE,
		"HI FREQ": HI_FREQ,
		"nblocks": nblocks,
		"nsamples": NSAMPLES,
		"gain": GAIN,
		"SG GAIN": SG_GAIN,
		"lst_start" : lst_start,
		"jd_start" : jd_start,
		"local_start": local_start
		"ut_start": ut_start,
		"lst_end" : lst_end,
		"jd_end" : jd_end,
		"local_end": local_end,
		"ut_end": ut_end,
		"DATA": final_avg}			
    file_path = os.path.join("data_today", name)
    
    # if there are lots of blocks discard the raw data
    np.savez(file_path, metadata=metadata) 
	return file_path
		
		
    
print("Experiment calibrated for the following settings:\n"
    + f"SAMPLE RATE: {SAMPLE_RATE}\n"
    + f"RADIO FREQUENCY: {HI_FREQ}\n"
    + f"NUMBER OF BLOCKS: {nblocks}\n"
    + f"NUMBER OF SAMPLES: {NSAMPLES}\n"
    + f"SIGNAL GENERATOR GAIN: {SG_GAIN}\n"
    + f"SDR GAIN: {GAIN}\n")
print("If this is correct, press any key to continue.")
input()

fi = capture_at("MEASUREMENT")

print("DONE!")
dat = np.load(fi, allow_pickle=True)['metadata'].item()
plt.plot(dat['DATA'])
plt.show()



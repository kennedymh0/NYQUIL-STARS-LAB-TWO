import numpy as np
import ugradio 

sdr = ugradio.sdr.SDR(direct=False, center_freq=1420.405e6, sample_rate=3.2e6, gain=10)
_raw= sdr.capture_data(nblocks=2, nsamples=2048)
print("shape:", _raw.shape, "dtype:", _raw.dtype)
sdr.close()

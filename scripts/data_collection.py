import numpy as np
import ugradio
import ugradio.timing as timing
import ugradio.doppler as doppler
import os
import time

HI_FREQ     = 1400e6
SAMPLE_RATE = 2.4e6
NSAMPLES    = 2048
N_BLOCKS    = 200
SDR_CENTER  = 10e6

def make_sdr(center_freq=HI_FREQ, sample_rate=SAMPLE_RATE, gain=40):
    s = ugradio.sdr.SDR(center_freq=center_freq, sample_rate=SAMPLE_RATE, gain=gain)
    return s


def power_spectrum(iq, nsamples=NSAMPLES):
    w = np.hanning(len(iq))
    return np.abs(np.fft.fftshift(np.fft.fft(iq, n=nsamples))) ** 2

def freq_axis(center_freq=0, rate=SAMPLE_RATE, nsamples=NSAMPLES):
    return np.fft.fftshift(np.fft.fftfreq(nsamples, 1.0/rate)) + center_freq

def check_levels(iq):
    r = iq.real
    print(f"  std={r.std():.4f}  min={r.min():.4f}  max={r.max():.4f}")
    if np.mean(np.abs(r) > 0.95 * np.abs(r).max()) > 0.01:
        print("  !! Clipping detected — reduce gain")
    elif r.std() < 0.005:
        print("  !! Heavily quantized — increase gain")
    else:
        print("  Levels OK")


def measure(label, nblocks=N_BLOCKS, out_dir="data", lo_freq=1400e6):
    os.makedirs(out_dir, exist_ok=True)

    jd_start  = timing.julian_date()
    lst_start = timing.lst()
    ut_start  = timing.utc()

    print(f"\n[{label}] UTC={ut_start}  LST={lst_start:.4f}h  JD={jd_start:.6f}")

    s = make_sdr(center_freq=lo_freq)
    spectra = np.zeros((nblocks, NSAMPLES))

    for i in range(nblocks):
        try:
            raw = s.capture_data(nblocks=1, nsamples=NSAMPLES)
            spectra[i] = power_spectrum(raw[0])
            if i == 0:
                check_levels(raw[0])
        except Exception as e:
            print(f"  Block {i} error: {e} — NaN inserted")
            spectra[i] = np.nan

    s.close()
    jd_end = timing.julian_date()

    fname = os.path.join(out_dir, f"{label}_{int(jd_start * 1e5)}.npz")
    np.savez(fname,
             spectra     = spectra,
             freqs_hz    = freq_axis(),
             jd_start    = jd_start,
             jd_end      = jd_end,
             jd_mid      = 0.5 * (jd_start + jd_end),
             lst_start   = lst_start,
             center_freq = lo_freq,
             sample_rate = SAMPLE_RATE,
             nblocks     = nblocks,
             nsamples    = NSAMPLES)

    print(f"  → Saved: {fname}")
    return spectra, fname


def observe_frequency_switch(nblocks=500, out_dir="data"):
    print("=== FREQUENCY SWITCHED OBSERVATION ===")
    print("Set upstream LO to POSITION 1 (line in upper half). Type LO frequency (hz):")
    lo1 = float(input())
    s_on, f_on   = measure("son",  nblocks=nblocks, out_dir=out_dir, lo_freq=lo1)

    print("\nSwitch upstream LO to POSITION 2 (line in lower half). Type LO frequency (hz):")
    lo2 = float(input())
    s_off, f_off = measure("soff", nblocks=nblocks, out_dir=out_dir, lo_freq=lo2)

    return s_on, s_off

def observe_calibration(nblocks=50, out_dir="data"):
    print("\n=== CALIBRATION: COLD SKY ===")
    print("Horn at zenith, aperture clear. Press Enter.")
    input()
    s_cold, f_cold = measure("scold", nblocks=nblocks, out_dir=out_dir)

    print("\n=== CALIBRATION: BLACKBODY ===")
    print("Fill horn aperture with people (~300K). Press Enter.")
    input()
    s_cal, f_cal   = measure("scal",  nblocks=nblocks, out_dir=out_dir)

    return s_cold, s_cal


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["check", "line", "cal", "all"], default="check")
    p.add_argument("--nblocks",     type=int, default=500)
    p.add_argument("--nblocks_cal", type=int, default=50)
    p.add_argument("--outdir",      default="data")
    args = p.parse_args()

    if args.mode in ("check", "all"):
        print("Opening SDR for level check...")
        try:
            s = make_sdr()
            raw = s.capture_data(nblocks=1, nsamples=NSAMPLES)
            check_levels(raw[0])
            s.close()
            print("Hardware check passed.")
        except Exception as e:
            print(f"HARDWARE ERROR: {e}")
            raise SystemExit(1)

    if args.mode in ("line", "all"):
        observe_frequency_switch(nblocks=args.nblocks, out_dir=args.outdir)

    if args.mode in ("cal", "all"):
        observe_calibration(nblocks=args.nblocks_cal, out_dir=args.outdir)
import numpy as np
import matplotlib.pyplot as plt
import os

HI_FREQ    = 1420.405752e6
C_LIGHT    = 3e5             # km/s

def load_npz(filepath):
    d = np.load(filepath)
    print(f"Loaded: {filepath}  shape={d['spectra'].shape}  "
          f"LST={float(d['lst_start']):.3f}h")
    return d

def average_spectra(spectra):
    return np.nanmean(spectra, axis=0), np.nanmedian(spectra, axis=0)

def smooth(spectrum, nchan=10):
    return np.convolve(spectrum, np.ones(nchan)/nchan, mode="same")

def freq_to_velocity(freqs, rest_freq=HI_FREQ):
    return -C_LIGHT * (freqs - rest_freq) / rest_freq

#raw average spectra (no analysis AT ALL except for averaging)
#def plot_raw_data(d_on, d_off, d_cold, d_cal, smooth_n=10):
    #fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    #fig.suptitle("Raw Averaged Power Spectra", fontsize=14)

    #datasets = [d_on,   d_off,   d_cold,        d_cal]
    #titles   = ["s_on", "s_off", "s_cold (sky)", "s_cal (blackbody)"]

    #for ax, d, title in zip(axes.flat, datasets, titles):
        #freqs        = d["freqs_hz"]
        #mean, median = average_spectra(d["spectra"])

        #ax.plot(freqs/1e6, smooth(mean,   smooth_n), label="mean")
        #ax.plot(freqs/1e6, smooth(median, smooth_n), label="median", ls="--")
        #ax.set_title(title)
        #ax.set_xlabel("Frequency (MHz)")
        #ax.set_ylabel("Power (arb. units)")
        #ax.legend(fontsize=8)
        #ax.grid(True, alpha=0.3)

    #plt.tight_layout()
    #plt.savefig("plot_raw_data.png", dpi=150)
    #plt.show()
    #print("Saved: plot_raw_data.png")
    
def plot_raw(d_on, d_off, smooth_n=10):
    fig, axes = plt.subplots(2, 1, figsize=(10,5))
    fig.suptitle("Raw Averaged Power Spectra", fontsize=14)

    datasets = [d_on,   d_off]
    titles   = ["s_on", "s_off"]

    for ax, d, title in zip(axes.flat, datasets, titles):
        freqs        = d["freqs_hz"]
        mean, median = average_spectra(d["spectra"])

        ax.plot(freqs/1e6, smooth(mean,   smooth_n), label="mean")
        ax.plot(freqs/1e6, smooth(median, smooth_n), label="median", ls="--")
        ax.set_title(title)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Power (arb. units)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_raw.png", dpi=150)
    plt.show()
    print("Saved: plot_raw.png")

# line shape plots 

def plot_line_shape(d_on, d_off, smooth_n=10):
    freqs      = d_on["freqs_hz"]
    s_on_m, _  = average_spectra(d_on["spectra"])
    s_off_m, _ = average_spectra(d_off["spectra"])

    r_smooth = smooth(s_on_m / s_off_m, smooth_n)
    vels     = freq_to_velocity(freqs)

    fig, (ax1, ax2) = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle("Bandpass-Corrected Line Shape  (r = s_on / s_off)", fontsize=14)

    ax1.plot(freqs/1e6, r_smooth, color="steelblue")
    ax1.axhline(1.0, color="gray", ls="--", lw=0.8, label="r=1 (no line)")
    ax1.set_xlabel("Baseband Frequency (MHz)")
    ax1.set_ylabel("r_line (unitless)")
    ax1.set_title("vs. frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_line_shape.png", dpi=150)
    plt.show()
    print("Saved: plot_line_shape.png")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--son",   required=True, help="path to son .npz file")
    p.add_argument("--soff",  required=True, help="path to soff .npz file")
    #p.add_argument("--scold", required=True, help="path to scold .npz file")
    #p.add_argument("--scal",  required=True, help="path to scal .npz file")
    p.add_argument("--smooth", type=int, default=10)
    args = p.parse_args()

    d_on   = load_npz(args.son)
    d_off  = load_npz(args.soff)
    #d_cold = load_npz(args.scold)
    #d_cal  = load_npz(args.scal)

    #plot_raw_data(d_on, d_off, d_cold, d_cal, smooth_n=args.smooth)
    plot_raw(d_on, d_off, smooth_n=args.smooth)
    plot_line_shape(d_on, d_off, smooth_n=args.smooth)
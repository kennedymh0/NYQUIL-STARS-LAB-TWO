import numpy as np

# Data points from your signal generator test
injected_volts = np.array([0.1, 0.2, 0.3, 0.4]) # Example values
sdr_measured_amps = np.array([120, 245, 360, 485]) # Example digital units

# Fit a linear relationship [cite: 36, 240]
slope, intercept = np.polyfit(injected_volts, sdr_measured_amps, 1)

def sdr_to_volts(digital_value):
    """Convert digital SDR units back to physical Volts"""
    return (digital_value - intercept) / slope

import ugradio

# The coefficients provided in Section 9.2.1 [cite: 310]
fir_coeffs = np.array([-54, -36, -41, -40, -32, -14, 14, 53, 101, 156, 215, 273, 
                       327, 372, 404, 421, 421, 404, 372, 327, 273, 215, 156, 
                       101, 53, 14, -14, -32, -40, -41, -36, -54])

# Calculate the frequency response [cite: 312, 313]
_, response = ugradio.dft.dft(fir_coeffs)
power_correction = np.abs(response)**2 # Square of the DFT 

# To apply: corrected_spectrum = raw_spectrum / power_correction

def get_velocity_factor(known_length, delta_t):
    """
    known_length: length of the test cable in meters
    delta_t: time between input and reflection in seconds (from cursor)
    """
    # Signal travels 2 * length [cite: 297]
    v_cable = (2 * known_length) / delta_t 
    c = 299792458 
    return v_cable / c # Should be ~0.66 per your friend

def get_cable_length(v_cable, delta_t_roof):
    """Use the calculated v_cable to find the unknown roof cable length [cite: 324]"""
    return (v_cable * delta_t_roof) / 2

def calculate_db_gain(v_in, v_out):
    """Calculates gain/loss in dB using the voltage ratio [cite: 325]"""
    return 20 * np.log10(v_out / v_in)

# Example Application:
# amplifier_gain = calculate_db_gain(v_before_amp, v_after_amp)
# cable_loss_per_m = calculate_db_gain(v_start, v_end) / total_length
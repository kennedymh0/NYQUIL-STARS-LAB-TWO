import ugradio
import ugradio.timing as timing
import time 
import numpy as np
import matplotlib.pyplot as plt
import os

l_rad = np.radians(120.0)
b_rad = np.radians(0.0)

x = np.array([0.,0.,0.])
x[0] = np.cos(b_rad)*np.cos(l_rad)
x[1] = np.cos(b_rad)*np.sin(l_rad)
x[2] = np.sin(b_rad)

lst_rad = ugradio.timing.lst()
print(lst_rad)
phi_rad= np.radians(37.8732)
R_eq_to_gal = np.array([
    [-0.054876, -0.873437, -0.483835],
    [0.494109, -0.444830, 0.746982],
    [-0.867666, -0.198076, 0.455984]
    ])
R_gal_to_eq = np.transpose(R_eq_to_gal)

R_eq_to_ha = np.array([
    [ np.cos(lst_rad), np.sin(lst_rad), 0],
    [ np.sin(lst_rad), -np.cos(lst_rad), 0],
    [ 0, 0, 1]
    ])
    
R_ha_to_azalt = np.array([
    [-np.sin(phi_rad), 0, np.cos(phi_rad)],
    [0, -1, 0],
    [np.cos(phi_rad), 0, np.sin(phi_rad)]
    ])
    
R_temp = np.dot(R_eq_to_ha, R_gal_to_eq)
R_total = np.dot(R_ha_to_azalt, R_temp)

x_prime = np.dot(R_total, x)

az_rad = np.arctan2(x_prime[1], x_prime[0])
alt_rad = np.arcsin(x_prime[2])

azimuth = np.degrees(az_rad)
altitude = np.degrees(alt_rad)

if azimuth < 0:
    azimuth += 360
    
print(f"Point the horn to Azimuth: {azimuth:.2f} deg, Altitude: {altitude:.2f} deg")

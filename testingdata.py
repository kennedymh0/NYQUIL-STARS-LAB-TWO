import ugradio
import numpy as np
import matplotlib.pyplot as plt
import math

son = np.load('newdataa/secondtest/son_0.8.npz', allow_pickle=True)
soff = np.load('newdataa/secondtest/off_0.8.npz', allow_pickle=True)

plt.plot(np.log10(son['p']))
plt.plot(np.log10(soff['p']))
plt.show()

plt.plot(son['p'])
plt.plot(soff['p'])
plt.show()

print("on spectrum max: " + f"{np.log10(son['p']).max()}")
print("off spectrum max: " + f"{np.log10(soff['p']).max()}")

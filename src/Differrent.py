# here is an example of plotting with matplotlib
import numpy as np
import matplotlib.pyplot as plt

def sinusoid():
    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()

sinusoid()